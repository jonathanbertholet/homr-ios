import Foundation

// MARK: - Morphology
//
// Native Swift re-implementation of the subset of OpenCV's morphology API that
// the `homr` Python pipeline relies on:
//
//   * cv2.getStructuringElement(MORPH_RECT | MORPH_ELLIPSE | MORPH_CROSS, ...)
//   * cv2.erode  / cv2.dilate          (grayscale min / max filters)
//   * cv2.morphologyEx(MORPH_OPEN | MORPH_CLOSE | MORPH_ERODE | MORPH_DILATE)
//
// Everything matches OpenCV 4.x semantics so the golden-fixture parity tests
// reproduce the exact byte output of `opencv-python`. The key OpenCV behaviours
// reproduced here are documented inline next to the code that implements them.
extension CV {

    // MARK: getStructuringElement

    /// Builds a morphology kernel, mirroring `cv2.getStructuringElement`.
    ///
    /// - Parameters:
    ///   - shape: `.rect` (full mask), `.ellipse` (rasterized disk/ellipse) or
    ///            `.cross` (the anchor row + the anchor column).
    ///   - size:  kernel `(width, height)` in pixels.
    ///   - anchor: optional anchor. OpenCV's default is `(-1, -1)` which means
    ///             "centre", i.e. `(width / 2, height / 2)` using integer
    ///             division. A negative component is treated as "centre" too.
    /// - Returns: a `StructuringElement` whose `mask` is row-major (`true` where
    ///            the kernel is active).
    static func getStructuringElement(
        _ shape: MorphShape,
        _ size: (width: Int, height: Int),
        anchor: (x: Int, y: Int)? = nil
    ) -> StructuringElement {
        let width = size.width
        let height = size.height

        // Resolve the anchor exactly like OpenCV: a missing or negative value
        // collapses to the centre via integer division.
        let anchorX: Int = {
            guard let ax = anchor?.x, ax >= 0 else { return width / 2 }
            return ax
        }()
        let anchorY: Int = {
            guard let ay = anchor?.y, ay >= 0 else { return height / 2 }
            return ay
        }()

        // The active mask, initialised to all-false; each branch fills it in.
        var mask = [Bool](repeating: false, count: max(0, width * height))

        switch shape {
        case .rect:
            // MORPH_RECT: every cell is active.
            for i in mask.indices { mask[i] = true }

        case .cross:
            // MORPH_CROSS: a cell is active when it lies on the anchor row OR
            // the anchor column (OpenCV: `i == anchor.y` fills the whole row,
            // otherwise only `j == anchor.x` is set).
            for i in 0..<height {
                let rowBase = i * width
                if i == anchorY {
                    // The anchor row is fully active.
                    for j in 0..<width { mask[rowBase + j] = true }
                } else if anchorX >= 0 && anchorX < width {
                    // Off the anchor row, only the anchor column is active.
                    mask[rowBase + anchorX] = true
                }
            }

        case .ellipse:
            // MORPH_ELLIPSE: OpenCV's exact rasterization. It uses the *centre*
            // of the kernel (NOT the anchor) as the ellipse centre:
            //   r = height / 2, c = width / 2   (integer division)
            //   inv_r2 = 1 / (r*r)              (0 when r == 0)
            // For each row `i`, `dy = i - r`; when |dy| <= r the active span is
            //   dx = cvRound( c * sqrt((r*r - dy*dy) * inv_r2) )
            //   j1 = max(c - dx, 0),  j2 = min(c + dx + 1, width)
            // and cells in [j1, j2) are set. (`cvRound` = round-half-to-even.)
            let r = height / 2
            let c = width / 2
            let invR2: Double = r != 0 ? 1.0 / (Double(r) * Double(r)) : 0.0

            for i in 0..<height {
                let dy = i - r
                guard abs(dy) <= r else { continue } // outside the vertical extent
                // Half-chord width at this row, rounded like OpenCV's saturate_cast<int>.
                let inner = (Double(r) * Double(r) - Double(dy) * Double(dy)) * invR2
                let dx = Int((Double(c) * (inner.squareRoot())).rounded(.toNearestOrEven))
                let j1 = max(c - dx, 0)
                let j2 = min(c + dx + 1, width)
                if j1 < j2 {
                    let rowBase = i * width
                    for j in j1..<j2 { mask[rowBase + j] = true }
                }
            }
        }

        return StructuringElement(
            width: width,
            height: height,
            mask: mask,
            anchorX: anchorX,
            anchorY: anchorY
        )
    }

    // MARK: erode / dilate

    /// Grayscale erosion = local **minimum** over the kernel's active footprint.
    /// Mirrors `cv2.erode` with the default `BORDER_CONSTANT`, whose border value
    /// for erosion is `+inf` (here `255`) so out-of-bounds samples never lower the
    /// running minimum. Applied `iterations` times in sequence.
    static func erode(
        _ image: GrayscaleImage,
        _ kernel: StructuringElement,
        iterations: Int = 1
    ) -> GrayscaleImage {
        // `false` selects the min/255-border (erosion) variant below.
        return applyRankFilter(image, kernel, iterations: iterations, isDilate: false)
    }

    /// Grayscale dilation = local **maximum** over the kernel's active footprint.
    /// Mirrors `cv2.dilate`; the `BORDER_CONSTANT` value for dilation is `-inf`
    /// (here `0`) so out-of-bounds samples never raise the running maximum.
    /// Applied `iterations` times in sequence.
    static func dilate(
        _ image: GrayscaleImage,
        _ kernel: StructuringElement,
        iterations: Int = 1
    ) -> GrayscaleImage {
        // `true` selects the max/0-border (dilation) variant below.
        return applyRankFilter(image, kernel, iterations: iterations, isDilate: true)
    }

    // MARK: morphologyEx

    /// Compound morphology, mirroring `cv2.morphologyEx`.
    ///
    /// - `.erode`  → `erode(iterations)`
    /// - `.dilate` → `dilate(iterations)`
    /// - `.open`   → `erode(iterations)` then `dilate(iterations)` (removes
    ///               small bright specks).
    /// - `.close`  → `dilate(iterations)` then `erode(iterations)` (fills small
    ///               dark holes).
    ///
    /// Note: like OpenCV, the `iterations` count is applied to *each* primitive
    /// step (e.g. open with 2 iterations erodes twice, then dilates twice).
    static func morphologyEx(
        _ image: GrayscaleImage,
        _ op: MorphOp,
        _ kernel: StructuringElement,
        iterations: Int = 1
    ) -> GrayscaleImage {
        switch op {
        case .erode:
            return erode(image, kernel, iterations: iterations)
        case .dilate:
            return dilate(image, kernel, iterations: iterations)
        case .open:
            let eroded = erode(image, kernel, iterations: iterations)
            return dilate(eroded, kernel, iterations: iterations)
        case .close:
            let dilated = dilate(image, kernel, iterations: iterations)
            return erode(dilated, kernel, iterations: iterations)
        }
    }

    // MARK: - Private helpers

    /// Shared engine for erosion (min) and dilation (max).
    ///
    /// OpenCV defines both operations with the same neighbour-offset formula
    /// (the kernels homr uses are all centro-symmetric, so dilation's kernel
    /// reflection is a no-op):
    ///
    ///   dst(x, y) = reduce over active cells (kx, ky):
    ///                 src(x + kx - anchorX, y + ky - anchorY)
    ///
    /// where `reduce` is `min` for erosion and `max` for dilation. Out-of-bounds
    /// neighbours use the identity border value (255 for erode, 0 for dilate).
    private static func applyRankFilter(
        _ image: GrayscaleImage,
        _ kernel: StructuringElement,
        iterations: Int,
        isDilate: Bool
    ) -> GrayscaleImage {
        // Non-positive iteration counts are a pass-through (matches cv2, which
        // leaves the source untouched when iterations < 1).
        guard iterations >= 1 else { return image }

        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return image }

        // Precompute the signed pixel offsets of every active kernel cell once,
        // so the inner loop is a flat list walk instead of a 2-D scan with a
        // mask test. Offsets are relative to the anchor, per the formula above.
        var offsets: [(dx: Int, dy: Int)] = []
        offsets.reserveCapacity(kernel.mask.count)
        for ky in 0..<kernel.height {
            for kx in 0..<kernel.width where kernel.mask[ky * kernel.width + kx] {
                offsets.append((dx: kx - kernel.anchorX, dy: ky - kernel.anchorY))
            }
        }

        // Degenerate kernel with no active cells: OpenCV would leave the image
        // unchanged, so mirror that rather than producing an all-border result.
        guard !offsets.isEmpty else { return image }

        // Border identity: erosion ignores 255 (max), dilation ignores 0 (min).
        let border: UInt8 = isDilate ? 0 : 255

        // Ping-pong between two buffers across iterations to avoid reallocating.
        var src = image.pixels
        var dst = [UInt8](repeating: 0, count: width * height)

        for _ in 0..<iterations {
            src.withUnsafeBufferPointer { srcPtr in
                dst.withUnsafeMutableBufferPointer { dstPtr in
                    for y in 0..<height {
                        for x in 0..<width {
                            // Seed with the identity so a fully out-of-bounds
                            // footprint yields the border value.
                            var acc: UInt8 = border
                            for off in offsets {
                                let sx = x + off.dx
                                let sy = y + off.dy
                                // Out-of-bounds → identity border (does not move acc).
                                let value: UInt8
                                if sx >= 0, sx < width, sy >= 0, sy < height {
                                    value = srcPtr[sy * width + sx]
                                } else {
                                    value = border
                                }
                                if isDilate {
                                    if value > acc { acc = value }
                                } else {
                                    if value < acc { acc = value }
                                }
                            }
                            dstPtr[y * width + x] = acc
                        }
                    }
                }
            }
            // The freshly computed buffer becomes the source for the next pass.
            swap(&src, &dst)
        }

        // After the final swap, `src` holds the most recent result.
        return GrayscaleImage(pixels: src, width: width, height: height)
    }
}
