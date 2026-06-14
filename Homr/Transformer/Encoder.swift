import Foundation
import OnnxRuntimeBindings

/// Port of `homr/transformer/encoder_inference.py` (`Encoder`).
///
/// Runs the vision encoder ONNX model. It takes the normalized staff image
/// tensor (`[1, 1, maxHeight, maxWidth]` = `[1,1,256,1280]`) and produces the
/// `context` tensor (`[1, seq, decoderDim]`) that conditions the autoregressive
/// decoder.
///
/// Differences from Python:
/// - Python may run an fp16 encoder on CUDA/CoreML via `io_binding`. On iOS we
///   always use the fp32 encoder (CPU, or CoreML EP when `useCoremlEncoder` is
///   set) and the plain `session.run(...)` API (no io-binding convenience in
///   `OnnxRuntimeBindings`). The returned context is fp32.
final class Encoder {
    /// Result of an encoder pass: the flat context buffer plus its tensor shape.
    /// The shape is returned so the decoder can re-bind the context as an
    /// `ORTValue` and slice `context[:, :1]` on later steps.
    struct Context {
        let values: [Float]
        let shape: [Int]
    }

    private let session: ORTSession
    private let inputName: String
    private let outputName: String

    /// Serialises encoder `Run` calls across threads.
    ///
    /// ONNX Runtime's CoreML execution provider is NOT thread-safe for concurrent
    /// `Run` on a single session: overlapping runs trample the session's shared
    /// MLMultiArray IO buffers, which surfaces as
    /// "CoreMLExecutionProvider … Status Message: mlmultiarray_buffer has no data"
    /// and corrupts the (cached, singleton) session so every later run fails too.
    /// `parseIndividualStaves` decodes staves in parallel, so we serialise the
    /// (brief, conv-heavy) encoder pass here. The expensive 608-step decode runs on
    /// the CPU EP — which IS concurrency-safe — and stays parallel, so we keep both
    /// CoreML acceleration and the multi-staff speed-up.
    private static let runLock = NSLock()

    /// Builds the encoder session via the shared factory.
    ///
    /// The input/output tensor names are read from the session when available
    /// (mirrors Python's `get_inputs()[0].name` / `get_outputs()[0].name`),
    /// falling back to the conventional "input"/"output".
    init(config: TransformerConfig) throws {
        self.session = try ONNXSessionFactory.makeEncoderSession(useCoreML: config.useCoremlEncoder)
        self.inputName = (try? session.inputNames())?.first ?? "input"
        self.outputName = (try? session.outputNames())?.first ?? "output"
    }

    /// Runs the encoder over a normalized image tensor.
    ///
    /// - Parameters:
    ///   - x: row-major fp32 image data of length `shape.reduce(1, *)`.
    ///   - shape: `[1, 1, H, W]` (H = `maxHeight`, W = `maxWidth`).
    /// - Returns: the `context` tensor as a flat fp32 buffer plus its `[1, seq, dim]` shape.
    func generate(_ x: [Float], shape: [Int]) throws -> Context {
        // Wrap the image buffer as an fp32 ONNX tensor (same Data(buffer:) +
        // ORTValue(tensorData:elementType:shape:) pattern as SegnetInference).
        let inputData = x.withUnsafeBufferPointer { Data(buffer: $0) }
        let nsShape = shape.map { NSNumber(value: $0) }
        let inputValue = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .float,
            shape: nsShape
        )

        // Hold the lock across the run AND the output read: the output `ORTValue`
        // can alias session-internal buffers that a concurrent run would overwrite,
        // so we copy the context out (into `values`) before releasing.
        Encoder.runLock.lock()
        defer { Encoder.runLock.unlock() }

        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            outputNames: [outputName],
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw EncoderError.missingOutput
        }

        // Read the context shape (e.g. [1, seq, 512]) and its raw fp32 bytes.
        let outShape = try output.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        let outData = try output.tensorData() as Data
        let values = outData.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self))
        }
        return Context(values: values, shape: outShape)
    }

    enum EncoderError: LocalizedError {
        case missingOutput
        var errorDescription: String? {
            switch self {
            case .missingOutput: return "Encoder produced no output tensor"
            }
        }
    }
}
