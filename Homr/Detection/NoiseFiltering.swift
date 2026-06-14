import CoreGraphics
import Foundation

// MARK: - Noise filtering (port of homr/noise_filtering.py)
//
// The Python module estimates per-tile "noise" on the staff mask, builds a
// coarse grid of those estimates, and zeroes out (via a 0/255 mask) the image
// regions whose tile AND a neighbouring tile are both noisy. The masked image
// is produced with `cv2.bitwise_and(img, img, mask=mask)`.
//
// Adaptations from the Python source (documented per function):
//   * All `Debug`, `cv2.rectangle`, `cv2.putText` and the BGR debug image are
//     dropped — they only feed the optional `noise_crop` debug artifact.
//   * Python calls `create_noise_grid(255 * prediction.staff, ...)` because in
//     that pipeline the staff mask is 0/1. In OUR pipeline (see
//     `SegnetInference.SegmentationMaps`) every mask is ALREADY 0/255, so we
//     compute the noise estimate on the staff mask AS-IS (no `* 255`).
//   * `estimate_noise` returns a float that Python stores into a `uint8` grid
//     (`grid[i, j] = noise`). numpy casts float→uint8 by truncating toward zero
//     and wrapping modulo 256; `gridCast` reproduces that exactly.

// MARK: - estimate_noise

/// Estimates image noise via a 3×3 Laplacian-style filter, port of
/// `estimate_noise`.
///
/// Implements `cv2.filter2D(gray, CV_64F, M)` by hand with the kernel
/// `[[1,-2,1],[-2,4,-2],[1,-2,1]]`, the default centre anchor, and
/// `BORDER_REFLECT_101` (OpenCV's `BORDER_DEFAULT`). The result is
/// `sum(|response|) / (H * W)`. Because the kernel is symmetric, correlation
/// (what `filter2D` actually performs) equals convolution.
func estimateNoise(_ gray: GrayscaleImage) -> Double {
    let H = gray.height
    let W = gray.width
    if H == 0 || W == 0 {
        return 0.0
    }

    // Laplacian-style kernel M, row-major.
    let kernel: [[Double]] = [
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1],
    ]

    var total = 0.0
    gray.pixels.withUnsafeBufferPointer { src in
        for r in 0..<H {
            for c in 0..<W {
                var acc = 0.0
                for ky in 0..<3 {
                    let sr = borderReflect101(r + ky - 1, H)
                    for kx in 0..<3 {
                        let sc = borderReflect101(c + kx - 1, W)
                        acc += kernel[ky][kx] * Double(src[sr * W + sc])
                    }
                }
                total += abs(acc)
            }
        }
    }
    return total / Double(H * W)
}

/// `cv2.BORDER_REFLECT_101` index reflection (a.k.a. `BORDER_DEFAULT`),
/// faithfully reproducing OpenCV's `borderInterpolate`: `gfedcb|abcdefgh|gfedcba`
/// (the border pixel is NOT repeated). A length-1 axis maps everything to 0.
private func borderReflect101(_ pIn: Int, _ len: Int) -> Int {
    if pIn >= 0 && pIn < len {
        return pIn
    }
    if len == 1 {
        return 0
    }
    var p = pIn
    repeat {
        if p < 0 {
            p = -p // delta == 1: -p - 1 + 1
        } else {
            p = 2 * len - 2 - p // len-1 - (p-len) - 1
        }
    } while p < 0 || p >= len
    return p
}

/// numpy float→uint8 element assignment (`grid[i, j] = noise`): truncate toward
/// zero, then wrap modulo 256.
private func gridCast(_ value: Double) -> UInt8 {
    UInt8(truncatingIfNeeded: Int(value)) // Int(Double) truncates toward zero
}

// MARK: - create_grid

/// Builds the coarse `ceil(H/M) × ceil(W/N)` grid of per-tile noise estimates,
/// stored as `uint8` (wrapping) like Python. Port of `create_grid`.
///
/// Returned as a row-major `[UInt8]` together with its dimensions so callers can
/// index `grid[i * cols + j]`.
private func createGrid(_ gray: GrayscaleImage, _ M: Int, _ N: Int) -> (cells: [UInt8], rows: Int, cols: Int) {
    let imgHeight = gray.height
    let imgWidth = gray.width
    let rows = Int(ceil(Double(imgHeight) / Double(M)))
    let cols = Int(ceil(Double(imgWidth) / Double(N)))
    var cells = [UInt8](repeating: 0, count: rows * cols)

    var i = 0
    var y1 = 0
    while y1 < imgHeight {
        var j = 0
        var x1 = 0
        while x1 < imgWidth {
            let y2 = min(y1 + M, imgHeight)
            let x2 = min(x1 + N, imgWidth)
            let tile = gray.cropped(to: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
            let noise = estimateNoise(tile)
            cells[i * cols + j] = gridCast(noise)
            j += 1
            x1 += N
        }
        i += 1
        y1 += M
    }

    return (cells, rows, cols)
}

// MARK: - get_neighbors

/// Returns the up/left/down/right neighbour cell values, port of `get_neighbors`.
private func getNeighbors(_ grid: [UInt8], rows: Int, cols: Int, _ i: Int, _ j: Int) -> [UInt8] {
    var neighbors: [UInt8] = []
    if i > 0 { neighbors.append(grid[(i - 1) * cols + j]) }
    if j > 0 { neighbors.append(grid[i * cols + (j - 1)]) }
    if i < rows - 1 { neighbors.append(grid[(i + 1) * cols + j]) }
    if j < cols - 1 { neighbors.append(grid[i * cols + (j + 1)]) }
    return neighbors
}

// MARK: - apply_noise_filter

/// Walks the grid, marking (255) every tile region that is NOT filtered out.
/// A tile is filtered (left at 0 in the mask) only when both it and at least
/// one neighbour exceed `image_noise_limit`. Port of `apply_noise_filter` with
/// the debug-image drawing removed.
///
/// `mask` is mutated in place (a full-image 0/255 buffer). Returns
/// `(filtered_cells, total_cells)`.
private func applyNoiseFilter(
    _ grid: [UInt8],
    rows: Int,
    cols: Int,
    mask: inout [UInt8],
    imgWidth: Int,
    imgHeight: Int,
    _ M: Int,
    _ N: Int
) -> (filtered: Int, total: Int) {
    var filteredCells = 0
    var totalCells = 0

    var i = 0
    var y1 = 0
    while y1 < imgHeight {
        var j = 0
        var x1 = 0
        while x1 < imgWidth {
            let y2 = min(y1 + M, imgHeight)
            let x2 = min(x1 + N, imgWidth)
            let noise = Int(grid[i * cols + j])
            let neighbors = getNeighbors(grid, rows: rows, cols: cols, i, j)
            let anyNeighborAboveLimit = neighbors.contains { Int($0) > Constants.imageNoiseLimit }

            if noise > Constants.imageNoiseLimit && anyNeighborAboveLimit {
                // Noisy tile next to a noisy neighbour: filter it out (leave mask 0).
                filteredCells += 1
            } else {
                // Keep this region: set the mask to 255 over the tile.
                for row in y1..<y2 {
                    let base = row * imgWidth
                    for col in x1..<x2 {
                        mask[base + col] = 255
                    }
                }
            }
            totalCells += 1
            j += 1
            x1 += N
        }
        i += 1
        y1 += M
    }

    return (filteredCells, totalCells)
}

// MARK: - handle_filter_results

/// Decides whether the computed mask should be applied, port of
/// `handle_filter_results`:
///   * filters > 50% of cells → give up (return nil, no filtering),
///   * some cells filtered → return the mask,
///   * nothing filtered → return nil.
private func handleFilterResults(_ filteredCells: Int, _ totalCells: Int, _ mask: [UInt8], width: Int, height: Int) -> GrayscaleImage? {
    let half = 0.5
    if totalCells == 0 {
        return nil
    }
    if Double(filteredCells) / Double(totalCells) > half {
        // Would filter more than 50% of the image; skip noise filtering.
        return nil
    } else if filteredCells > 0 {
        return GrayscaleImage(pixels: mask, width: width, height: height)
    }
    return nil
}

// MARK: - create_noise_grid

/// Produces the 0/255 keep-mask for the image, port of `create_noise_grid`.
/// Returns nil when filtering should be skipped (matching Python).
///
/// Deviation: when `M` or `N` collapse to 0 (image smaller than 20px in a
/// dimension) Python's `range(0, len, 0)` would raise; we instead skip
/// filtering (return nil) so iOS never crashes on a tiny input.
private func createNoiseGrid(_ gray: GrayscaleImage) -> GrayscaleImage? {
    let imgHeight = gray.height
    let imgWidth = gray.width
    let M = imgHeight / 20
    let N = imgWidth / 20
    guard M >= 1, N >= 1 else {
        return nil
    }

    var mask = [UInt8](repeating: 0, count: imgWidth * imgHeight)
    let grid = createGrid(gray, M, N)
    let (filteredCells, totalCells) = applyNoiseFilter(
        grid.cells,
        rows: grid.rows,
        cols: grid.cols,
        mask: &mask,
        imgWidth: imgWidth,
        imgHeight: imgHeight,
        M, N
    )
    return handleFilterResults(filteredCells, totalCells, mask, width: imgWidth, height: imgHeight)
}

// MARK: - filter_predictions

/// Applies noise filtering to every prediction mask, port of
/// `filter_predictions`.
///
/// The keep-mask is derived from the staff mask. Python passes
/// `255 * prediction.staff` (its staff mask is 0/1); ours is already 0/255 so
/// it is used directly. When no mask is produced the predictions are returned
/// unchanged.
func filterPredictions(_ prediction: InputPredictions) -> InputPredictions {
    // 0/255 adaptation: staff mask is already 0/255 here (no `* 255`).
    guard let mask = createNoiseGrid(prediction.staff) else {
        return prediction
    }
    return InputPredictions(
        original: CV.bitwiseAnd(prediction.original, prediction.original, mask: mask),
        preprocessed: CV.bitwiseAnd(prediction.preprocessed, prediction.preprocessed, mask: mask),
        notehead: CV.bitwiseAnd(prediction.notehead, prediction.notehead, mask: mask),
        symbols: CV.bitwiseAnd(prediction.symbols, prediction.symbols, mask: mask),
        staff: CV.bitwiseAnd(prediction.staff, prediction.staff, mask: mask),
        clefsKeys: CV.bitwiseAnd(prediction.clefsKeys, prediction.clefsKeys, mask: mask),
        stemsRest: CV.bitwiseAnd(prediction.stemsRest, prediction.stemsRest, mask: mask)
    )
}
