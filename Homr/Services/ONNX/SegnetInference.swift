import CoreGraphics
import Foundation
import OnnxRuntimeBindings

/// Segmentation inference ported from homr/segmentation/inference_segnet.py.
///
/// Runs the UNet segnet model over a sliding 320×320 window, merges the
/// overlapping per-patch argmax class maps, and splits the result into the five
/// binary masks homr consumes downstream.
struct SegnetInference {
    /// Class indices produced by the segnet model (channel argmax).
    private enum SegClass {
        static let background = 0
        static let stemsRests = 1
        static let noteheads = 2
        static let clefsKeys = 3
        static let staff = 4
        static let symbols = 5
        static let count = 6
    }

    struct SegmentationMaps {
        let staff: [UInt8]
        let symbols: [UInt8]
        let stemsRests: [UInt8]
        let noteheads: [UInt8]
        let clefsKeys: [UInt8]
        let width: Int
        let height: Int
    }

    enum InferenceError: LocalizedError {
        case inferenceFailed(String)

        var errorDescription: String? {
            switch self {
            case .inferenceFailed(let detail):
                return "Segmentation failed: \(detail)"
            }
        }
    }

    private let session: ORTSession
    private let winSize = 320
    private let stepSize = 320
    // Larger batches amortise the per-run overhead of the segnet pass; 8 fits a
    // typical page's 320×320 tiling comfortably within memory.
    private let batchSize = 8

    init(session: ORTSession) {
        self.session = session
    }

    /// Runs sliding-window segnet over a preprocessed grayscale image.
    func run(on image: GrayscaleImage) throws -> SegmentationMaps {
        let width = image.width
        let height = image.height
        var merged = [Float](repeating: 0, count: height * width)
        var weights = [Float](repeating: 0, count: height * width)

        var batchPatches: [[Float32]] = []
        var batchOrigins: [(y: Int, x: Int)] = []

        for yLoop in stride(from: 0, to: max(height, winSize), by: stepSize) {
            let y = min(yLoop, height - winSize)
            for xLoop in stride(from: 0, to: max(width, winSize), by: stepSize) {
                let x = min(xLoop, width - winSize)
                batchPatches.append(extractPatch(from: image, y: y, x: x))
                batchOrigins.append((y, x))

                if batchPatches.count == batchSize {
                    try runBatch(&batchPatches, &batchOrigins, &merged, &weights, width: width, height: height)
                }
            }
        }
        if !batchPatches.isEmpty {
            try runBatch(&batchPatches, &batchOrigins, &merged, &weights, width: width, height: height)
        }

        // Average overlapping patch contributions, then round to a class index.
        var classMap = [UInt8](repeating: 0, count: width * height)
        for index in classMap.indices {
            let w = weights[index]
            classMap[index] = w > 0 ? UInt8((merged[index] / w).rounded()) : 0
        }

        return SegmentationMaps(
            staff: mask(classMap, equals: SegClass.staff),
            symbols: mask(classMap, equals: SegClass.symbols),
            stemsRests: mask(classMap, equals: SegClass.stemsRests),
            noteheads: mask(classMap, equals: SegClass.noteheads),
            clefsKeys: mask(classMap, equals: SegClass.clefsKeys),
            width: width,
            height: height
        )
    }

    private func mask(_ classMap: [UInt8], equals value: Int) -> [UInt8] {
        let target = UInt8(value)
        return classMap.map { $0 == target ? 255 : 0 }
    }

    /// Extracts a (winSize × winSize) patch, white-padded past the image edges.
    /// homr replicates the single grayscale channel into the model's 3 inputs.
    private func extractPatch(from image: GrayscaleImage, y: Int, x: Int) -> [Float32] {
        var values = [Float32](repeating: 255, count: winSize * winSize)
        let width = image.width
        let height = image.height
        let y0 = max(y, 0), x0 = max(x, 0)
        let y1 = min(y + winSize, height), x1 = min(x + winSize, width)

        image.pixels.withUnsafeBufferPointer { src in
            values.withUnsafeMutableBufferPointer { dst in
                for row in y0..<y1 {
                    let srcRow = row * width
                    let dstRow = (row - y) * winSize - x
                    for col in x0..<x1 {
                        dst[dstRow + col] = Float32(src[srcRow + col])
                    }
                }
            }
        }
        return values
    }

    private func runBatch(
        _ patches: inout [[Float32]],
        _ origins: inout [(y: Int, x: Int)],
        _ merged: inout [Float],
        _ weights: inout [Float],
        width: Int,
        height: Int
    ) throws {
        let outputs = try infer(patches: patches)
        for (output, origin) in zip(outputs, origins) {
            mergePatch(output, y: origin.y, x: origin.x, into: &merged, weights: &weights, width: width, height: height)
        }
        patches.removeAll(keepingCapacity: true)
        origins.removeAll(keepingCapacity: true)
    }

    private func infer(patches: [[Float32]]) throws -> [[UInt8]] {
        // CRITICAL: always run a FIXED batch size. The segnet session is cached and
        // reused across pages with the CoreML execution provider; if the input's
        // batch dimension changes between runs (e.g. a full batch of 8 then a
        // 3-patch remainder, or a different remainder on the next page), ORT's
        // CoreML EP recompiles the MLProgram mid-session, which corrupts memory and
        // aborts ("-[OS_dispatch_mach_msg _setContext:] unrecognized selector")
        // when scanning 2+ pages. We pad the final partial batch with white patches
        // up to `batchSize` so the shape is constant, then drop the padded outputs.
        let realCount = patches.count
        let runCount = batchSize
        let patchFloats = 3 * winSize * winSize
        var flatInput = [Float32](repeating: 255, count: runCount * patchFloats)

        // Layout: [batch, channel(3), H, W]; the 3 channels are identical. Only the
        // first `realCount` slots are filled; the rest stay white padding.
        flatInput.withUnsafeMutableBufferPointer { dst in
            for (batchIndex, patch) in patches.enumerated() {
                patch.withUnsafeBufferPointer { src in
                    let planeSize = winSize * winSize
                    for channel in 0..<3 {
                        let base = batchIndex * patchFloats + channel * planeSize
                        for i in 0..<planeSize {
                            dst[base + i] = src[i]
                        }
                    }
                }
            }
        }

        let shape: [NSNumber] = [runCount, 3, winSize, winSize].map { NSNumber(value: $0) }
        let inputData = flatInput.withUnsafeBufferPointer { Data(buffer: $0) }
        let inputValue = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .float,
            shape: shape
        )

        let outputs = try session.run(
            withInputs: ["input": inputValue],
            outputNames: ["output"],
            runOptions: nil
        )
        guard let output = outputs["output"] else {
            throw InferenceError.inferenceFailed("Missing output tensor")
        }
        let outputData = try output.tensorData() as Data
        // Decode the whole padded batch, then keep only the real patches' maps.
        let decoded = decodeClassMaps(from: outputData, batchCount: runCount)
        return realCount == runCount ? decoded : Array(decoded.prefix(realCount))
    }

    /// Argmax over the class channel for every pixel: output is [batch, classes, H, W].
    private func decodeClassMaps(from data: Data, batchCount: Int) -> [[UInt8]] {
        let classCount = SegClass.count
        let planeSize = winSize * winSize
        let floatsPerBatch = classCount * planeSize

        return data.withUnsafeBytes { raw -> [[UInt8]] in
            let floats = raw.bindMemory(to: Float32.self)
            return (0..<batchCount).map { batch in
                var result = [UInt8](repeating: 0, count: planeSize)
                let batchBase = batch * floatsPerBatch
                result.withUnsafeMutableBufferPointer { out in
                    for pixel in 0..<planeSize {
                        var bestClass = 0
                        var bestScore = floats[batchBase + pixel]
                        for classIndex in 1..<classCount {
                            let score = floats[batchBase + classIndex * planeSize + pixel]
                            if score > bestScore {
                                bestScore = score
                                bestClass = classIndex
                            }
                        }
                        out[pixel] = UInt8(bestClass)
                    }
                }
                return result
            }
        }
    }

    private func mergePatch(
        _ patch: [UInt8],
        y: Int,
        x: Int,
        into merged: inout [Float],
        weights: inout [Float],
        width: Int,
        height: Int
    ) {
        let y1 = min(y + winSize, height)
        let x1 = min(x + winSize, width)
        patch.withUnsafeBufferPointer { src in
            merged.withUnsafeMutableBufferPointer { mrg in
                weights.withUnsafeMutableBufferPointer { wgt in
                    for row in 0..<(y1 - y) {
                        let srcRow = row * winSize
                        let dstRow = (y + row) * width + x
                        for col in 0..<(x1 - x) {
                            mrg[dstRow + col] += Float(src[srcRow + col])
                            wgt[dstRow + col] += 1
                        }
                    }
                }
            }
        }
    }
}

extension SegnetInference.SegmentationMaps {
    /// Renders a colour overlay of the detected classes for visual verification.
    /// Each class gets a distinct tint; background stays transparent.
    func makeOverlay() -> CGImage? {
        let pixelCount = width * height
        guard pixelCount > 0 else { return nil }

        // RGBA, premultiplied.
        var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
        let tints: [([UInt8], UInt8)] = [
            (staff, 0),        // staff lines  → blue
            (stemsRests, 1),   // stems/rests  → green
            (clefsKeys, 2),    // clefs/keys   → orange
            (symbols, 3),      // misc symbols → purple
            (noteheads, 4),    // noteheads    → red (drawn last, highest priority)
        ]

        for (mask, colorIndex) in tints {
            let color = Self.overlayColors[Int(colorIndex)]
            for i in 0..<pixelCount where mask[i] > 0 {
                let o = i * 4
                rgba[o] = color.0
                rgba[o + 1] = color.1
                rgba[o + 2] = color.2
                rgba[o + 3] = 255
            }
        }

        return rgba.withUnsafeMutableBytes { raw -> CGImage? in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return nil }
            return context.makeImage()
        }
    }

    private static let overlayColors: [(UInt8, UInt8, UInt8)] = [
        (40, 120, 255),   // blue
        (40, 200, 90),    // green
        (255, 150, 30),   // orange
        (170, 70, 220),   // purple
        (240, 40, 40),    // red
    ]

    /// Counts the foreground pixels per class for a quick textual summary.
    var summary: String {
        func count(_ mask: [UInt8]) -> Int { mask.lazy.filter { $0 > 0 }.count }
        return """
        Noteheads: \(count(noteheads))px · Staff: \(count(staff))px
        Stems/rests: \(count(stemsRests))px · Clefs/keys: \(count(clefsKeys))px
        """
    }
}
