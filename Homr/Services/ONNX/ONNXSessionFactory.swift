import Foundation
import OnnxRuntimeBindings

/// Creates ONNX Runtime sessions with CoreML acceleration on Apple devices.
enum ONNXSessionFactory {
    enum SessionError: LocalizedError {
        case modelMissing(String)
        case sessionCreationFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelMissing(let name):
                return "Model not found: \(name)"
            case .sessionCreationFailed(let detail):
                return "ONNX session failed: \(detail)"
            }
        }
    }

    /// Shared ORT environment — one per process is recommended by ONNX Runtime.
    private static let environment: ORTEnv = {
        try! ORTEnv(loggingLevel: .warning)
    }()

    // Per-model intra-op thread counts. These interact with the *cross-staff*
    // parallelism in `parseIndividualStaves`: the encoder/decoder sessions are
    // driven from several staff threads at once, so we keep their intra-op pools
    // small to avoid oversubscribing the CPU (coreCount × coreCount threads).
    // Segnet runs once at the top level, so it gets ORT's default (all cores).
    private static let segnetIntraOpThreads: Int32 = 0   // 0 = ORT default (≈ core count)
    private static let encoderIntraOpThreads: Int32 = 0  // brief conv-heavy pass per staff
    private static let decoderIntraOpThreads: Int32 = 1  // tiny per-step matmuls; parallelised across staves

    /// Cached segnet session + a lock guarding its one-time construction.
    private static var cachedSegnetSession: ORTSession?
    private static let segnetLock = NSLock()

    /// Builds (once) and reuses a session for segnet (CoreML EP when available).
    ///
    /// The session is cached and reused across pages/scans. This is correct —
    /// segnet runs sequentially, once per page, and a single `ORTSession` can be
    /// `run` repeatedly — and it avoids RE-CREATING the CoreML session in a tight
    /// multi-page loop, which compiled the MLProgram over and over and could
    /// corrupt memory (the `_setContext:` abort) when scanning 2+ pages.
    static func makeSegnetSession() throws -> ORTSession {
        segnetLock.lock()
        defer { segnetLock.unlock() }
        if let session = cachedSegnetSession { return session }

        let modelPath = HomrModelPaths.path(for: "\(HomrModelPaths.segnetBaseName).onnx")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw SessionError.modelMissing(modelPath.lastPathComponent)
        }
        let session = try makeSession(
            modelPath: modelPath.path, useCoreML: true, intraOpThreads: segnetIntraOpThreads
        )
        cachedSegnetSession = session
        return session
    }

    /// Encoder session — CoreML MLProgram can accelerate the encoder on device.
    static func makeEncoderSession(useCoreML: Bool = true) throws -> ORTSession {
        let fileName = "encoder_\(HomrModelPaths.transformerBaseName).onnx"
        let modelPath = HomrModelPaths.path(for: fileName)
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw SessionError.modelMissing(fileName)
        }
        return try makeSession(modelPath: modelPath.path, useCoreML: useCoreML, intraOpThreads: encoderIntraOpThreads)
    }

    /// Decoder must run on CPU EP (homr's dynamic KV-cache is incompatible with CoreML EP).
    static func makeDecoderSession() throws -> ORTSession {
        let fileName = "decoder_\(HomrModelPaths.transformerBaseName).onnx"
        let modelPath = HomrModelPaths.path(for: fileName)
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw SessionError.modelMissing(fileName)
        }
        return try makeSession(modelPath: modelPath.path, useCoreML: false, intraOpThreads: decoderIntraOpThreads)
    }

    /// Builds tuned session options: full graph optimisation plus an explicit
    /// intra-op thread count (`0` lets ORT pick ≈ core count).
    private static func makeTunedOptions(intraOpThreads: Int32) throws -> ORTSessionOptions {
        let options = try ORTSessionOptions()
        // `.all` enables every graph rewrite/fusion ORT supports (faster inference,
        // slightly slower session creation — fine since sessions are reused).
        try options.setGraphOptimizationLevel(.all)
        try options.setIntraOpNumThreads(intraOpThreads)
        return options
    }

    private static func makeSession(modelPath: String, useCoreML: Bool, intraOpThreads: Int32) throws -> ORTSession {
        // CoreML EP is unstable on the Simulator (can abort with obscure ObjC
        // selector errors); use plain CPU there. On device we still try CoreML
        // when requested and fall back to CPU if compilation fails.
        //
        // We also skip CoreML when a debugger is attached: compiling the
        // MLProgram for the Neural Engine raises/handles Mach signals internally,
        // and LLDB stops on those benign signals — which looks exactly like the
        // app "hanging on launch" while tethered to Xcode (it runs fine once
        // detached). Falling back to the CPU EP under the debugger keeps stepping
        // through the pipeline smooth; shipping/untethered runs still get CoreML.
        #if targetEnvironment(simulator)
        let preferCoreML = false
        #else
        let preferCoreML = useCoreML && !isBeingDebugged()
        #endif
        if preferCoreML, let session = try? makeCoreMLSession(modelPath: modelPath, intraOpThreads: intraOpThreads) {
            return session
        }
        // Plain CPU session (also the fallback when the CoreML EP is unavailable
        // or fails to compile the MLProgram).
        let options = try makeTunedOptions(intraOpThreads: intraOpThreads)
        return try ORTSession(env: environment, modelPath: modelPath, sessionOptions: options)
    }

    private static func makeCoreMLSession(modelPath: String, intraOpThreads: Int32) throws -> ORTSession {
        let options = try makeTunedOptions(intraOpThreads: intraOpThreads)
        // Mirror homr's CoreML MLProgram settings for real GPU/ANE execution.
        // The provider name is case-sensitive ("CoreML").
        let coreMLOptions: [String: String] = [
            "MLComputeUnits": "CPUAndGPU",
            "ModelFormat": "MLProgram",
        ]
        try options.appendExecutionProvider("CoreML", providerOptions: coreMLOptions)
        return try ORTSession(env: environment, modelPath: modelPath, sessionOptions: options)
    }

    /// True when the process is running under a debugger (LLDB/Xcode).
    ///
    /// Uses the documented `sysctl(KERN_PROC)` `P_TRACED` check (App Store safe).
    /// We cache the result: attach state doesn't change mid-session for our use,
    /// and this is read on every session creation.
    private static let debuggerAttached: Bool = {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        let result = mib.withUnsafeMutableBufferPointer { ptr -> Int32 in
            sysctl(ptr.baseAddress, UInt32(ptr.count), &info, &size, nil, 0)
        }
        guard result == 0 else { return false }
        return (info.kp_proc.p_flag & P_TRACED) != 0
    }()

    private static func isBeingDebugged() -> Bool { debuggerAttached }
}
