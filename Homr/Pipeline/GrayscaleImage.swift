import CoreGraphics
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// A single-channel 8-bit grayscale image stored as a flat row-major buffer.
///
/// Used as the shared raster representation throughout preprocessing and
/// segmentation so the image is only rasterized once (mirrors how homr keeps a
/// single numpy array through `autocrop → resize → apply_clahe → segnet`).
struct GrayscaleImage {
    var pixels: [UInt8]
    let width: Int
    let height: Int

    init(pixels: [UInt8], width: Int, height: Int) {
        self.pixels = pixels
        self.width = width
        self.height = height
    }

    #if canImport(UIKit)
    /// Rasterizes a UIImage into a grayscale buffer (luminance, no alpha).
    init?(uiImage: UIImage) {
        guard let cgImage = uiImage.cgImage else { return nil }
        self.init(cgImage: cgImage)
    }
    #endif

    init?(cgImage: CGImage) {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        var buffer = [UInt8](repeating: 0, count: width * height)
        let success = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return false }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard success else { return nil }

        self.pixels = buffer
        self.width = width
        self.height = height
    }

    /// Bilinearly resizes to a new width, preserving aspect ratio.
    func resized(toWidth targetWidth: Int) -> GrayscaleImage {
        guard targetWidth > 0, targetWidth != width else { return self }
        let ratio = Double(targetWidth) / Double(width)
        let targetHeight = max(1, Int((Double(height) * ratio).rounded()))

        var output = [UInt8](repeating: 0, count: targetWidth * targetHeight)
        let xScale = Double(width) / Double(targetWidth)
        let yScale = Double(height) / Double(targetHeight)

        pixels.withUnsafeBufferPointer { src in
            output.withUnsafeMutableBufferPointer { dst in
                for ty in 0..<targetHeight {
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

    /// Returns a cropped copy. The rect is clamped to image bounds.
    func cropped(to rect: CGRect) -> GrayscaleImage {
        let x0 = max(0, Int(rect.minX))
        let y0 = max(0, Int(rect.minY))
        let x1 = min(width, Int(rect.maxX))
        let y1 = min(height, Int(rect.maxY))
        let w = max(1, x1 - x0)
        let h = max(1, y1 - y0)

        var output = [UInt8](repeating: 0, count: w * h)
        for row in 0..<h {
            let srcStart = (y0 + row) * width + x0
            let dstStart = row * w
            output.replaceSubrange(dstStart..<(dstStart + w), with: pixels[srcStart..<(srcStart + w)])
        }
        return GrayscaleImage(pixels: output, width: w, height: h)
    }

    /// Renders the buffer back into a displayable CGImage.
    func makeCGImage() -> CGImage? {
        var buffer = pixels
        return buffer.withUnsafeMutableBytes { raw -> CGImage? in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return nil }
            return context.makeImage()
        }
    }
}
