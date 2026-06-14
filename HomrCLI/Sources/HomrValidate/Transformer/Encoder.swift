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

    /// Serialises encoder `Run` calls — ORT's CoreML EP is not concurrency-safe
    /// (concurrent runs corrupt the session: "mlmultiarray_buffer has no data").
    /// Harmless in the serial CLI; kept in sync with the app's parallel decode.
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
