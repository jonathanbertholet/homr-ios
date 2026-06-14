import Foundation

/// Bar line detection — port of `homr/bar_line_detection.py`.
///
/// Two small free functions: one morphological pre-pass that thickens the
/// (already binarized) symbol mask so thin vertical bar lines survive contour
/// fitting, and one size filter that keeps only the candidate rotated boxes that
/// are tall enough and narrow enough to be bar lines.

/// Thickens the bar-line mask before contour detection.
///
/// Port of `prepare_bar_line_image`. Python uses `np.ones((5, 3), np.uint8)` as
/// the dilation kernel. numpy `(5, 3)` is `(rows, cols)` = `(height, width)`,
/// so this is a 5-tall × 3-wide solid rectangle. `cv2.dilate` with a flat
/// all-ones kernel is exactly a `MORPH_RECT` structuring element, so we build
/// the equivalent rectangular element (width = 3, height = 5) and dilate once.
func prepareBarLineImage(_ image: GrayscaleImage) -> GrayscaleImage {
    // numpy (rows=5, cols=3) -> CV size (width=3, height=5).
    let kernel = CV.getStructuringElement(.rect, (width: 3, height: 5))
    return CV.dilate(image, kernel, iterations: 1)
}

/// Filters bar-line candidates by size relative to the staff's unit size.
///
/// Port of `detect_bar_lines`. A candidate is kept only when it is at least
/// `bar_line_min_height(unit_size)` tall and at most `bar_line_max_width(unit_size)`
/// wide. Note the Python index mapping on `RotatedBoundingBox.size`:
/// `size[1]` is the height and `size[0]` is the width.
func detectBarLines(barLines: [RotatedBoundingBox], unitSize: Double) -> [RotatedBoundingBox] {
    var result: [RotatedBoundingBox] = []
    for barLine in barLines {
        // size[1] -> height: discard anything shorter than the minimum.
        if barLine.size.height < Constants.barLineMinHeight(unitSize) {
            continue
        }
        // size[0] -> width: discard anything wider than the maximum.
        if barLine.size.width > Constants.barLineMaxWidth(unitSize) {
            continue
        }
        result.append(barLine)
    }
    return result
}
