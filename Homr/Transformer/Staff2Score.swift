import Foundation

/// Port of `homr/transformer/staff2score.py` (`Staff2Score` + `ConvertToArray`).
///
/// Ties the transformer pipeline together: normalize image → encode → greedily
/// decode into `EncodedSymbol`s.
final class Staff2Score {
    private let config: TransformerConfig
    private let encoder: Encoder
    private let decoder: ScoreDecoder

    /// Builds the encoder and (fp32 CPU) decoder sessions.
    init(config: TransformerConfig) throws {
        self.config = config
        self.encoder = try Encoder(config: config)
        self.decoder = try get_decoder(config: config)
    }

    /// Port of `Staff2Score.predict`.
    ///
    /// IMPORTANT assumption (matches the Python pipeline): `image` is ALREADY the
    /// centered `maxHeight × maxWidth` (256×1280) staff canvas produced by the
    /// staff-parsing stage. `_transform` here therefore only normalizes it into
    /// the `[1, 1, 256, 1280]` fp32 tensor the encoder expects — it does NOT crop,
    /// pad or resize.
    ///
    /// Flow: transform → `encoder.generate` → `decoder.generate(startToken: 1,
    /// nonoteToken: 0, context:)`. The context is fp32 (the iOS decoder always
    /// runs fp32 on CPU), so no dtype cast is needed.
    func predict(_ image: GrayscaleImage) -> [EncodedSymbol] {
        let (x, shape) = ConvertToArray.transform(image)
        do {
            let context = try encoder.generate(x, shape: shape)
            return decoder.generate(
                startToken: config.bosToken,
                nonoteToken: config.nonoteToken,
                context: context.values,
                contextShape: context.shape
            )
        } catch {
            eprint("Staff2Score.predict failed:", error.localizedDescription)
            return []
        }
    }
}

/// Port of `ConvertToArray` (`_transform`).
///
/// `arr = image / 255`, add two leading axes → `[1, 1, H, W]`, then normalize
/// `(arr - mean) / std` with `mean = 0.7931`, `std = 0.1738`, output fp32.
///
/// `GrayscaleImage.pixels` is row-major (`row * width + col`), which is exactly
/// the flat layout of numpy's `arr[np.newaxis, np.newaxis, :, :]` (H then W), so
/// no reordering is required.
enum ConvertToArray {
    private static let mean: Float = 0.7931
    private static let std: Float = 0.1738

    /// Normalizes a grayscale image into the encoder input tensor.
    /// - Returns: the flat fp32 buffer and its `[1, 1, H, W]` shape.
    static func transform(_ image: GrayscaleImage) -> (values: [Float], shape: [Int]) {
        let height = image.height
        let width = image.width
        var values = [Float](repeating: 0, count: width * height)
        image.pixels.withUnsafeBufferPointer { src in
            values.withUnsafeMutableBufferPointer { dst in
                for i in 0..<(width * height) {
                    // (pixel/255 - mean) / std
                    dst[i] = (Float(src[i]) / 255 - mean) / std
                }
            }
        }
        return (values, [1, 1, height, width])
    }
}
