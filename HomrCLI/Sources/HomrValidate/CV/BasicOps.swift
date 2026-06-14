import Foundation

// MARK: - BasicOps
//
// Native Swift re-implementation of the simple per-pixel / statistical OpenCV
// calls used by `homr`:
//
//   * cv2.threshold(THRESH_BINARY | THRESH_BINARY_INV)
//   * cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, ...)
//   * cv2.subtract           (saturating per-pixel difference)
//   * cv2.bitwise_and        (with an optional mask)
//   * cv2.calcHist           (256-bin grayscale histogram)
//
// All behaviours match OpenCV 4.x so the parity fixtures reproduce exact bytes.
extension CV {

    /// Thresholding mode, mirroring the OpenCV flags homr uses.
    enum ThresholdType {
        case binary    // THRESH_BINARY
        case binaryInv // THRESH_BINARY_INV
    }

    // MARK: threshold

    /// Global fixed-level threshold, mirroring `cv2.threshold` for 8-bit images.
    ///
    /// OpenCV floors the threshold for 8-bit input (`ithresh = cvFloor(thresh)`)
    /// and then compares strictly greater-than:
    ///   * `.binary`     → `pixel > ithresh ? maxValue : 0`
    ///   * `.binaryInv`  → `pixel > ithresh ? 0 : maxValue`
    ///
    /// `maxValue` is saturate-cast to a byte just like OpenCV.
    static func threshold(
        _ image: GrayscaleImage,
        thresh: Double,
        maxValue: Double = 255,
        type: ThresholdType
    ) -> GrayscaleImage {
        // OpenCV's 8-bit path floors the threshold before comparing.
        let ithresh = thresh.rounded(.down)
        // saturate_cast<uchar>(maxValue): round-to-nearest then clamp to 0...255.
        let maxByte = saturatingByte(maxValue)

        var out = [UInt8](repeating: 0, count: image.pixels.count)
        image.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for i in 0..<src.count {
                    // Strict greater-than comparison against the floored threshold.
                    let isAbove = Double(src[i]) > ithresh
                    switch type {
                    case .binary:
                        dst[i] = isAbove ? maxByte : 0
                    case .binaryInv:
                        dst[i] = isAbove ? 0 : maxByte
                    }
                }
            }
        }
        return GrayscaleImage(pixels: out, width: image.width, height: image.height)
    }

    // MARK: adaptiveThresholdGaussian

    /// Adaptive Gaussian threshold, mirroring
    /// `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, ...)`.
    ///
    /// OpenCV's algorithm:
    ///   1. Blur the source with a separable Gaussian of size `blockSize`,
    ///      using `BORDER_REPLICATE`. The blur runs in float and the result is
    ///      rounded back to a byte per pixel — that rounded value is the local
    ///      `mean`.
    ///   2. An integer delta is derived from `c`:
    ///        `.binary`    → `idelta = ceil(c)`
    ///        `.binaryInv` → `idelta = floor(c)`
    ///   3. Per pixel (`d = pixel - mean`):
    ///        binary      → `d >  -idelta ? maxValue : 0`
    ///        binaryInv   → `d <= -idelta ? maxValue : 0`
    ///
    /// - Parameter blockSize: odd neighbourhood size (e.g. homr uses 21).
    /// - Parameter invert: `true` selects `THRESH_BINARY_INV`.
    static func adaptiveThresholdGaussian(
        _ image: GrayscaleImage,
        maxValue: Double,
        blockSize: Int,
        c: Double,
        invert: Bool
    ) -> GrayscaleImage {
        let width = image.width
        let height = image.height
        // Guard against degenerate sizes / even block sizes; fall back to a copy.
        guard width > 0, height > 0, blockSize >= 1, blockSize % 2 == 1 else {
            return image
        }

        let maxByte = saturatingByte(maxValue)
        // Integer delta: ceil for BINARY, floor for BINARY_INV (OpenCV exact).
        let idelta = invert ? Int(c.rounded(.down)) : Int(c.rounded(.up))

        // Local Gaussian mean, rounded to a byte per pixel (see helper).
        let mean = gaussianMean(image, blockSize: blockSize)

        var out = [UInt8](repeating: 0, count: width * height)
        image.pixels.withUnsafeBufferPointer { src in
            mean.withUnsafeBufferPointer { mptr in
                out.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<src.count {
                        // d = pixel - mean, matching OpenCV's lookup-table index.
                        let d = Int(src[i]) - Int(mptr[i])
                        let on: Bool = invert ? (d <= -idelta) : (d > -idelta)
                        dst[i] = on ? maxByte : 0
                    }
                }
            }
        }
        return GrayscaleImage(pixels: out, width: width, height: height)
    }

    // MARK: subtract

    /// Saturating per-pixel difference, mirroring `cv2.subtract(a, b)`:
    /// `result = max(0, a - b)` (the upper clamp at 255 can never trigger because
    /// both operands are already bytes). Used by brace/dot detection to remove
    /// the staff mask from the symbol mask.
    static func subtract(_ a: GrayscaleImage, _ b: GrayscaleImage) -> GrayscaleImage {
        // Operate over the overlapping pixel count to stay memory-safe even if
        // the two buffers somehow differ in length.
        let count = min(a.pixels.count, b.pixels.count)
        var out = [UInt8](repeating: 0, count: count)
        a.pixels.withUnsafeBufferPointer { ap in
            b.pixels.withUnsafeBufferPointer { bp in
                out.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<count {
                        let diff = Int(ap[i]) - Int(bp[i])
                        dst[i] = diff > 0 ? UInt8(diff) : 0 // clamp negatives to 0
                    }
                }
            }
        }
        return GrayscaleImage(pixels: out, width: a.width, height: a.height)
    }

    // MARK: bitwiseAnd

    /// Bitwise AND of two images with an optional mask, mirroring
    /// `cv2.bitwise_and(a, b, mask=mask)`.
    ///
    /// Where `mask` is provided, output pixels are `0` wherever `mask == 0`;
    /// elsewhere the result is `a & b`. homr always passes the *same* image for
    /// `a` and `b` together with a noise mask, so this effectively keeps the
    /// image only inside the mask and zeroes everything else.
    static func bitwiseAnd(
        _ a: GrayscaleImage,
        _ b: GrayscaleImage,
        mask: GrayscaleImage?
    ) -> GrayscaleImage {
        let count = min(a.pixels.count, b.pixels.count)
        var out = [UInt8](repeating: 0, count: count)
        a.pixels.withUnsafeBufferPointer { ap in
            b.pixels.withUnsafeBufferPointer { bp in
                out.withUnsafeMutableBufferPointer { dst in
                    if let mask = mask {
                        // Masked variant: gate the AND result on mask > 0.
                        mask.pixels.withUnsafeBufferPointer { mp in
                            let mCount = mp.count
                            for i in 0..<count {
                                // A short mask is treated as 0 (masked out) past its end.
                                if i < mCount, mp[i] != 0 {
                                    dst[i] = ap[i] & bp[i]
                                } else {
                                    dst[i] = 0
                                }
                            }
                        }
                    } else {
                        // Unmasked variant: plain per-pixel AND.
                        for i in 0..<count {
                            dst[i] = ap[i] & bp[i]
                        }
                    }
                }
            }
        }
        return GrayscaleImage(pixels: out, width: a.width, height: a.height)
    }

    // MARK: calcHist

    /// 256-bin intensity histogram, mirroring `cv2.calcHist([img], [0], None, [256], [0,256])`.
    /// Returns counts indexed by pixel value `0...255`.
    static func calcHist(_ image: GrayscaleImage) -> [Int] {
        var hist = [Int](repeating: 0, count: 256)
        image.pixels.withUnsafeBufferPointer { src in
            for i in 0..<src.count {
                hist[Int(src[i])] += 1 // each pixel value increments its bin
            }
        }
        return hist
    }

    // MARK: - Private helpers

    /// `saturate_cast<uchar>` for a Double: round-half-to-even, then clamp to
    /// `0...255`. Matches how OpenCV converts floating scalars to bytes.
    private static func saturatingByte(_ value: Double) -> UInt8 {
        let rounded = value.rounded(.toNearestOrEven)
        if rounded <= 0 { return 0 }
        if rounded >= 255 { return 255 }
        return UInt8(rounded)
    }

    /// Computes the per-pixel Gaussian-weighted local mean used by
    /// `adaptiveThresholdGaussian`, reproducing OpenCV's `GaussianBlur` with
    /// `BORDER_REPLICATE`.
    ///
    /// The blur is separable: it convolves horizontally then vertically with the
    /// same 1-D kernel, carrying full Double precision between passes and only
    /// rounding to a byte at the very end (OpenCV keeps the intermediate in
    /// `CV_32F` and rounds once via `convertTo`).
    private static func gaussianMean(_ image: GrayscaleImage, blockSize: Int) -> [UInt8] {
        let width = image.width
        let height = image.height
        let kernel = gaussianKernel1D(blockSize)        // normalized, sum == 1
        let radius = blockSize / 2                       // odd size → symmetric

        // --- Horizontal pass: byte source → Double intermediate. ---
        var horizontal = [Double](repeating: 0, count: width * height)
        image.pixels.withUnsafeBufferPointer { src in
            horizontal.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let rowBase = y * width
                    for x in 0..<width {
                        var acc = 0.0
                        for k in 0..<blockSize {
                            // BORDER_REPLICATE: clamp the sample index to the row.
                            var sx = x + k - radius
                            if sx < 0 { sx = 0 } else if sx >= width { sx = width - 1 }
                            acc += kernel[k] * Double(src[rowBase + sx])
                        }
                        dst[rowBase + x] = acc
                    }
                }
            }
        }

        // --- Vertical pass: Double intermediate → rounded byte output. ---
        var out = [UInt8](repeating: 0, count: width * height)
        horizontal.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    for x in 0..<width {
                        var acc = 0.0
                        for k in 0..<blockSize {
                            // BORDER_REPLICATE: clamp the sample index to the column.
                            var sy = y + k - radius
                            if sy < 0 { sy = 0 } else if sy >= height { sy = height - 1 }
                            acc += kernel[k] * src[sy * width + x]
                        }
                        // convertTo CV_8U: round-half-to-even + clamp.
                        dst[y * width + x] = saturatingByte(acc)
                    }
                }
            }
        }
        return out
    }

    /// Reproduces OpenCV's `getGaussianKernel(n, sigma = 0)` as a normalized
    /// `[Double]`.
    ///
    /// OpenCV behaviour replicated here:
    ///   * For an odd `n <= 7` with `sigma <= 0`, a hard-coded small-kernel table
    ///     is used (the adaptiveThreshold call passes `sigma = 0`, so these sizes
    ///     would hit the table inside OpenCV).
    ///   * Otherwise `sigma = 0.3 * ((n - 1) * 0.5 - 1) + 0.8` and
    ///     `kernel[i] = exp(-0.5 * x^2 / sigma^2)` with `x = i - (n - 1) / 2`,
    ///     finally normalized so the weights sum to 1.
    private static func gaussianKernel1D(_ n: Int) -> [Double] {
        // OpenCV's fixed tables for small odd kernels (sigma <= 0).
        let smallGaussianTab: [[Double]] = [
            [1.0],
            [0.25, 0.5, 0.25],
            [0.0625, 0.25, 0.375, 0.25, 0.0625],
            [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
        ]
        if n % 2 == 1, n <= 7 {
            // n >> 1 selects the table row matching this kernel size.
            return smallGaussianTab[n >> 1]
        }

        // Derived sigma exactly as in OpenCV when the caller passes sigma <= 0.
        let sigma = 0.3 * ((Double(n) - 1.0) * 0.5 - 1.0) + 0.8
        let scale2X = -0.5 / (sigma * sigma)
        let center = (Double(n) - 1.0) * 0.5

        var kernel = [Double](repeating: 0, count: n)
        var sum = 0.0
        for i in 0..<n {
            let x = Double(i) - center
            let t = exp(scale2X * x * x)
            kernel[i] = t
            sum += t
        }
        // Normalize so the weights sum to 1 (a true averaging blur).
        let inv = 1.0 / sum
        for i in 0..<n { kernel[i] *= inv }
        return kernel
    }
}
