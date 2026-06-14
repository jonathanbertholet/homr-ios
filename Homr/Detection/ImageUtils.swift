import CoreGraphics
import Foundation

// MARK: - ImageUtils
//
// Port of `homr/image_utils.py` plus a couple of small numpy / Python numeric
// stand-ins that the rest of the detection pipeline (StaffParsing, StaffDewarping)
// needs to reproduce CPython's exact truncation / rounding behaviour.
//
// The Python module slices a numpy array with `image[y1:y2, x1:x2]`, clamping the
// requested coordinates first via `_limit_x` / `_limit_y` (each does
// `max(0, min(shape-1, int(round(value))))`). We reproduce that here on a
// `GrayscaleImage`, returning the new top-left corner so callers can re-express
// symbol coordinates relative to the crop.

// MARK: Python / numpy numeric helpers

/// Python `int(x)` — truncation toward zero (NOT floor for negatives).
@inline(__always)
func homrInt(_ value: Double) -> Int {
    Int(value.rounded(.towardZero))
}

/// Python 3 `round(x)` / numpy `np.round(x)` — round half to even (banker's),
/// returning an `Int` (Python's `round(float)` yields an `int`).
@inline(__always)
func homrRound(_ value: Double) -> Int {
    Int(value.rounded(.toNearestOrEven))
}

/// numpy `np.round(x)` keeping a floating result (used when the rounded value is
/// then multiplied / subtracted again before any `int()` cast).
@inline(__always)
func homrRoundToDouble(_ value: Double) -> Double {
    value.rounded(.toNearestOrEven)
}

/// Python floor division `a // b` for integers (rounds toward negative infinity).
@inline(__always)
func homrFloorDiv(_ a: Int, _ b: Int) -> Int {
    Int((Double(a) / Double(b)).rounded(.down))
}

/// Arithmetic mean of a slice of pixel intensities (numpy `np.mean`).
@inline(__always)
func homrMean(_ values: [Double]) -> Double {
    if values.isEmpty { return Double.nan }
    return values.reduce(0, +) / Double(values.count)
}

// MARK: - GrayscaleImage exact-size resize

extension GrayscaleImage {
    /// Bilinear resize to an EXACT `(width, height)`, mirroring
    /// `cv2.resize(image, (width, height))` (which takes a `(w, h)` dsize and
    /// does NOT preserve aspect ratio). The existing `resized(toWidth:)` only
    /// supports aspect-preserving scaling, so `center_image_on_canvas` /
    /// `prepare_staff_image` need this variant.
    ///
    /// The sampling math matches the existing `resized(toWidth:)`: half-pixel
    /// centre alignment, clamped neighbour indices, bilinear blend, then
    /// round-half-up clamp to a byte.
    func resized(toWidth targetWidth: Int, height targetHeight: Int) -> GrayscaleImage {
        guard targetWidth > 0, targetHeight > 0 else {
            // Degenerate target size → empty image (matches a 0-size numpy resize
            // being unusable downstream; callers guard on this).
            return GrayscaleImage(pixels: [], width: max(0, targetWidth), height: max(0, targetHeight))
        }
        guard width > 0, height > 0 else {
            return GrayscaleImage(pixels: [UInt8](repeating: 0, count: targetWidth * targetHeight),
                                  width: targetWidth, height: targetHeight)
        }
        if targetWidth == width && targetHeight == height { return self }

        var output = [UInt8](repeating: 0, count: targetWidth * targetHeight)
        let xScale = Double(width) / Double(targetWidth)
        let yScale = Double(height) / Double(targetHeight)

        pixels.withUnsafeBufferPointer { src in
            output.withUnsafeMutableBufferPointer { dst in
                for ty in 0..<targetHeight {
                    // Map the destination centre back into source space.
                    let srcY = (Double(ty) + 0.5) * yScale - 0.5
                    let y0 = max(0, min(height - 1, Int(srcY.rounded(.down))))
                    let y1 = min(height - 1, y0 + 1)
                    let wy = max(0, min(1, srcY - Double(y0)))
                    for tx in 0..<targetWidth {
                        let srcX = (Double(tx) + 0.5) * xScale - 0.5
                        let x0 = max(0, min(width - 1, Int(srcX.rounded(.down))))
                        let x1 = min(width - 1, x0 + 1)
                        let wx = max(0, min(1, srcX - Double(x0)))

                        let p00 = Double(src[y0 * width + x0])
                        let p01 = Double(src[y0 * width + x1])
                        let p10 = Double(src[y1 * width + x0])
                        let p11 = Double(src[y1 * width + x1])
                        let top = p00 * (1 - wx) + p01 * wx
                        let bot = p10 * (1 - wx) + p11 * wx
                        let value = top * (1 - wy) + bot * wy
                        dst[ty * targetWidth + tx] = UInt8(min(255, max(0, value.rounded())))
                    }
                }
            }
        }
        return GrayscaleImage(pixels: output, width: targetWidth, height: targetHeight)
    }

    /// Raw rectangular slice `image[y1:y2, x1:x2]` (numpy semantics: the upper
    /// bounds are exclusive and an inverted/empty range yields a 0-sized image).
    /// Unlike `cropped(to:)`, this does NOT force a minimum 1x1 size, because the
    /// Python crop relies on possibly-empty slices.
    fileprivate func slice(x1: Int, y1: Int, x2: Int, y2: Int) -> GrayscaleImage {
        let w = max(0, x2 - x1)
        let h = max(0, y2 - y1)
        if w == 0 || h == 0 {
            return GrayscaleImage(pixels: [], width: w, height: h)
        }
        var output = [UInt8](repeating: 0, count: w * h)
        for row in 0..<h {
            let srcStart = (y1 + row) * width + x1
            let dstStart = row * w
            output.replaceSubrange(dstStart..<(dstStart + w), with: pixels[srcStart..<(srcStart + w)])
        }
        return GrayscaleImage(pixels: output, width: w, height: h)
    }
}

// MARK: - crop_image / crop_image_and_return_new_top

/// Clamp helper for the x axis (`_limit_x`): round-half-to-even, then clamp to
/// `0 ... width - 1`. `image.shape[1]` is the width.
private func limitX(_ image: GrayscaleImage, _ x: Double) -> Int {
    max(0, min(image.width - 1, homrRound(x)))
}

/// Clamp helper for the y axis (`_limit_y`): round-half-to-even, then clamp to
/// `0 ... height - 1`. `image.shape[0]` is the height.
private func limitY(_ image: GrayscaleImage, _ y: Double) -> Int {
    max(0, min(image.height - 1, homrRound(y)))
}

/// Port of `crop_image`: crop to the (clamped) rectangle, discarding the new
/// top-left corner.
func cropImage(_ image: GrayscaleImage, x1: Double, y1: Double, x2: Double, y2: Double) -> GrayscaleImage {
    cropImageAndReturnNewTop(image, x1: x1, y1: y1, x2: x2, y2: y2).0
}

/// Port of `crop_image_and_return_new_top`.
///
/// Sorts the two corners, clamps each coordinate to the image, slices, and
/// returns both the cropped image and the clamped top-left corner
/// (`np.array([x1_limited, y1_limited])`) so callers can shift coordinates.
func cropImageAndReturnNewTop(
    _ image: GrayscaleImage, x1: Double, y1: Double, x2: Double, y2: Double
) -> (GrayscaleImage, CV.Point) {
    let xMin = min(x1, x2)
    let xMax = max(x1, x2)
    let yMin = min(y1, y2)
    let yMax = max(y1, y2)
    let x1Limited = limitX(image, xMin)
    let y1Limited = limitY(image, yMin)
    let x2Limited = limitX(image, xMax)
    let y2Limited = limitY(image, yMax)
    let newTop = CV.Point(x1Limited, y1Limited)
    let cropped = image.slice(x1: x1Limited, y1: y1Limited, x2: x2Limited, y2: y2Limited)
    return (cropped, newTop)
}
