import CoreGraphics
import Foundation
import UIKit

/// Image preprocessing ported from homr/autocrop.py, resize.py, and color_adjust.py.
///
/// Pipeline: rasterize → autocrop → resize to 1920px width → CLAHE.
/// Everything operates on a single shared `GrayscaleImage` buffer.
enum ImagePreprocessor {
    private static let targetWidth = 1920

    static func preprocess(_ image: UIImage) -> GrayscaleImage? {
        guard var gray = GrayscaleImage(uiImage: image) else { return nil }
        gray = autocrop(gray)
        gray = gray.resized(toWidth: targetWidth)
        CLAHE.apply(to: &gray)
        return gray
    }

    /// Crops to the sheet-music page when borders are visible.
    ///
    /// Mirrors `autocrop.py`: threshold around the dominant (background) gray
    /// value, take the bounding box of the largest bright region, and skip the
    /// crop when the box already starts near the top-left (a full-page scan).
    static func autocrop(_ image: GrayscaleImage) -> GrayscaleImage {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return image }

        // Dominant gray value (the paper/background peak in the histogram).
        var histogram = [Int](repeating: 0, count: 256)
        for value in image.pixels { histogram[Int(value)] += 1 }
        let dominant = histogram.firstIndex(of: histogram.max() ?? 0) ?? 0
        let threshold = UInt8(max(0, dominant - 30))

        // Bounding box of all foreground (brighter-than-threshold) pixels.
        var minX = width, minY = height, maxX = 0, maxY = 0
        var found = false
        image.pixels.withUnsafeBufferPointer { src in
            for y in 0..<height {
                let row = y * width
                for x in 0..<width where src[row + x] > threshold {
                    found = true
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }

        guard found, maxX > minX, maxY > minY else { return image }

        // Skip the crop for full-page views (box hugs the top-left corner).
        let isFullPageView = minX < Int(Double(width) * 0.25) || minY < Int(Double(height) * 0.25)
        if isFullPageView { return image }

        let rect = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        return image.cropped(to: rect)
    }
}
