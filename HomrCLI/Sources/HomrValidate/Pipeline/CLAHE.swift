import Foundation

/// Contrast Limited Adaptive Histogram Equalization.
///
/// Faithful re-implementation of OpenCV's CLAHE (the algorithm homr applies in
/// `color_adjust.apply_clahe` with `clipLimit=1.0, tileGridSize=(8, 8)`).
/// Works in place on a row-major 8-bit grayscale buffer:
///   1. split the image into a grid of tiles,
///   2. build a clipped, redistributed histogram per tile,
///   3. turn each into a CDF lookup table,
///   4. bilinearly interpolate the four neighbouring tile LUTs per pixel.
enum CLAHE {
    static func apply(
        to image: inout GrayscaleImage,
        clipLimit: Double = 1.0,
        tilesX: Int = 8,
        tilesY: Int = 8
    ) {
        let width = image.width
        let height = image.height
        guard width >= tilesX, height >= tilesY else { return }

        let histSize = 256
        let tileW = width / tilesX
        let tileH = height / tilesY
        let tileArea = tileW * tileH

        // OpenCV clip-limit semantics: counts, scaled to tile area, min 1.
        var clipCount = Int(clipLimit * Double(tileArea) / Double(histSize))
        if clipCount < 1 { clipCount = 1 }

        // Flat LUT table: [tileY][tileX][256] laid out contiguously for speed.
        var luts = [UInt8](repeating: 0, count: tilesX * tilesY * histSize)

        image.pixels.withUnsafeBufferPointer { src in
            luts.withUnsafeMutableBufferPointer { lut in
                var hist = [Int](repeating: 0, count: histSize)
                for ty in 0..<tilesY {
                    let y0 = ty * tileH
                    let y1 = (ty == tilesY - 1) ? height : y0 + tileH
                    for tx in 0..<tilesX {
                        let x0 = tx * tileW
                        let x1 = (tx == tilesX - 1) ? width : x0 + tileW

                        for i in 0..<histSize { hist[i] = 0 }
                        for y in y0..<y1 {
                            let row = y * width
                            for x in x0..<x1 {
                                hist[Int(src[row + x])] += 1
                            }
                        }

                        // Clip the histogram and gather the excess.
                        var excess = 0
                        for i in 0..<histSize where hist[i] > clipCount {
                            excess += hist[i] - clipCount
                            hist[i] = clipCount
                        }
                        // Redistribute the excess uniformly across all bins.
                        let increment = excess / histSize
                        var remainder = excess % histSize
                        for i in 0..<histSize { hist[i] += increment }
                        if remainder > 0 {
                            let step = max(1, histSize / remainder)
                            var idx = 0
                            while remainder > 0 && idx < histSize {
                                hist[idx] += 1
                                remainder -= 1
                                idx += step
                            }
                        }

                        // CDF → normalized LUT for this tile.
                        let actualArea = (x1 - x0) * (y1 - y0)
                        let scale = Double(histSize - 1) / Double(max(1, actualArea))
                        let lutBase = (ty * tilesX + tx) * histSize
                        var cdf = 0
                        for i in 0..<histSize {
                            cdf += hist[i]
                            let mapped = Int((Double(cdf) * scale).rounded())
                            lut[lutBase + i] = UInt8(min(255, max(0, mapped)))
                        }
                    }
                }
            }
        }

        // Bilinearly blend the surrounding tile LUTs for every pixel.
        let halfTileW = Double(tileW) / 2.0
        let halfTileH = Double(tileH) / 2.0
        image.pixels.withUnsafeMutableBufferPointer { px in
            luts.withUnsafeBufferPointer { lut in
                for y in 0..<height {
                    let (ty0, ty1, wy) = tileSpan(
                        coord: Double(y), half: halfTileH, tileSize: tileH, tileCount: tilesY)
                    let rowBase = y * width
                    for x in 0..<width {
                        let (tx0, tx1, wx) = tileSpan(
                            coord: Double(x), half: halfTileW, tileSize: tileW, tileCount: tilesX)
                        let value = Int(px[rowBase + x])

                        let v00 = Double(lut[(ty0 * tilesX + tx0) * histSize + value])
                        let v01 = Double(lut[(ty0 * tilesX + tx1) * histSize + value])
                        let v10 = Double(lut[(ty1 * tilesX + tx0) * histSize + value])
                        let v11 = Double(lut[(ty1 * tilesX + tx1) * histSize + value])

                        let top = v00 * (1 - wx) + v01 * wx
                        let bot = v10 * (1 - wx) + v11 * wx
                        let out = top * (1 - wy) + bot * wy
                        px[rowBase + x] = UInt8(min(255, max(0, out.rounded())))
                    }
                }
            }
        }
    }

    /// Maps a pixel coordinate to the two bracketing tile indices and a weight.
    /// Pixels before the first tile centre or after the last clamp to one tile.
    private static func tileSpan(
        coord: Double, half: Double, tileSize: Int, tileCount: Int
    ) -> (Int, Int, Double) {
        let g = (coord - half) / Double(tileSize)
        if g <= 0 {
            return (0, 0, 0)
        }
        if g >= Double(tileCount - 1) {
            return (tileCount - 1, tileCount - 1, 0)
        }
        let lower = Int(g)
        return (lower, lower + 1, g - Double(lower))
    }
}
