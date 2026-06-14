import Foundation

/// Bounding-box geometry for symbol detection — port of `homr/bounding_boxes.py`.
///
/// The Python module builds axis-aligned (`BoundingBox`) and rotated
/// (`RotatedBoundingBox`, `BoundingEllipse`) hulls around connected components,
/// tests them for overlap, and merges touching boxes into groups. All OpenCV
/// calls are routed through the native `CV` namespace; the only image type is
/// `GrayscaleImage` (the 0/255 binary masks the Python code operates on).
///
/// Reference semantics: Python mutates and de-duplicates these objects in lists
/// and sets, so the angled hierarchy is implemented as reference types
/// (`class`). `AngledBoundingBox` (and `BoundingBox`) are `Hashable`/`Equatable`
/// purely by their `box` value, matching Python's `__eq__`/`__hash__`.

// MARK: - Debug drawing

/// Marker protocol mirroring Python's `DebugDrawable` ABC.
///
/// In Python every drawable implements `draw_onto_image`. On iOS all drawing is
/// debug-only and unused for inference, so the requirement has an empty default
/// implementation and concrete types simply inherit the no-op.
protocol DebugDrawable: AnyObject {
    /// Debug-only rendering hook. No-op on iOS (inference never draws).
    func drawOntoImage()
}

extension DebugDrawable {
    func drawOntoImage() {} // debug-only, not needed for iOS inference
}

// MARK: - Free helpers (module-level functions in Python)

/// Computes the four axis-aligned corners of a rotated rectangle from its
/// `center`/`size`, ignoring the angle — exactly like
/// `calculate_edges_of_rotated_rectangle` in Python.
private func calculateEdgesOfRotatedRectangle(
    _ box: CV.RotatedRect
) -> (topLeft: CV.PointF, bottomLeft: CV.PointF, topRight: CV.PointF, bottomRight: CV.PointF) {
    let halfW = box.size.width / 2
    let halfH = box.size.height / 2
    let cx = box.center.x
    let cy = box.center.y
    return (
        CV.PointF(cx - halfW, cy - halfH),
        CV.PointF(cx - halfW, cy + halfH),
        CV.PointF(cx + halfW, cy - halfH),
        CV.PointF(cx + halfW, cy + halfH)
    )
}

/// Returns `true` if either polygon has a vertex inside (or on) the other.
/// Port of `do_polygons_overlap` using `cv2.pointPolygonTest`.
private func doPolygonsOverlap(_ poly1: [CV.Point], _ poly2: [CV.Point]) -> Bool {
    for point in poly1 {
        if CV.pointPolygonTest(poly2, CV.PointF(Double(point.x), Double(point.y)), measureDist: false) >= 0 {
            return true
        }
    }
    for point in poly2 {
        if CV.pointPolygonTest(poly1, CV.PointF(Double(point.x), Double(point.y)), measureDist: false) >= 0 {
            return true
        }
    }
    return false
}

/// `True` when both rotated-rect dimensions are finite and strictly positive.
/// Port of `_has_box_valid_size`.
private func hasBoxValidSize(_ box: CV.RotatedRect) -> Bool {
    !box.size.width.isNaN && !box.size.height.isNaN && box.size.width > 0 && box.size.height > 0
}

// MARK: - AnyPolygon

/// Base class holding a polygon outline — port of `AnyPolygon`.
///
/// `polygon` is the ordered list of integer vertices used by the overlap test
/// (`cv2.boxPoints` for rectangles, `cv2.ellipse2Poly` for ellipses).
class AnyPolygon: DebugDrawable {
    /// Ordered integer polygon vertices.
    let polygon: [CV.Point]

    init(polygon: [CV.Point]) {
        self.polygon = polygon
    }
}

// MARK: - BoundingBox

/// Axis-aligned bounding box stored as `(x1, y1, x2, y2)` — port of `BoundingBox`.
///
/// Python's `BoundingBox` derives directly from `AnyPolygon` (NOT from
/// `AngledBoundingBox`). Equality/hash are by the `box` tuple.
final class BoundingBox: AnyPolygon, Hashable {
    /// Debug identifier carried through transformations.
    let debugId: Int
    /// Source contour points for this box.
    let contours: [CV.Point]
    /// The box as `(x1, y1, x2, y2)` integer corners.
    let box: (x1: Int, y1: Int, x2: Int, y2: Int)
    /// Centre point `((x1 + x2) / 2, (y1 + y2) / 2)`.
    let center: CV.PointF
    /// `(width, height)` of the box.
    let size: CV.Size
    /// The box expressed as a zero-angle `RotatedRect` (Python `rotated_box`).
    let rotatedBox: CV.RotatedRect

    init(box: (x1: Int, y1: Int, x2: Int, y2: Int), contours: [CV.Point], debugId: Int = 0) {
        self.debugId = debugId
        self.contours = contours
        self.box = box
        let cx = Double(box.x1 + box.x2) / 2
        let cy = Double(box.y1 + box.y2) / 2
        let center = CV.PointF(cx, cy)
        let size = CV.Size(width: Double(box.x2 - box.x1), height: Double(box.y2 - box.y1))
        self.center = center
        self.size = size
        let rotated = CV.RotatedRect(center: center, size: size, angle: 0)
        self.rotatedBox = rotated
        let poly = CV.boxPoints(rotated).map { CV.Point(Int($0.x), Int($0.y)) }
        super.init(polygon: poly)
    }

    /// Crops `img` to this box (inclusive of the lower/right edge, hence `+1`).
    /// Port of `BoundingBox.extract`.
    func extract(_ img: GrayscaleImage) -> GrayscaleImage {
        cropImage(img, Double(box.x1), Double(box.y1), Double(box.x2 + 1), Double(box.y2 + 1))
    }

    /// Returns a copy of `img` where everything outside this box is white (255).
    /// Port of `BoundingBox.blank_everything_outside_of_box`.
    func blankEverythingOutsideOfBox(_ img: GrayscaleImage) -> GrayscaleImage {
        let width = img.width
        let height = img.height
        var out = [UInt8](repeating: 255, count: width * height)
        let yStart = max(0, box.y1)
        let yEnd = min(height, box.y2)
        let xStart = max(0, box.x1)
        let xEnd = min(width, box.x2)
        if yStart < yEnd && xStart < xEnd {
            for row in yStart..<yEnd {
                let base = row * width
                for col in xStart..<xEnd {
                    out[base + col] = img.pixels[base + col]
                }
            }
        }
        return GrayscaleImage(pixels: out, width: width, height: height)
    }

    /// Grows the box by `increase` pixels in every direction, clamped to the
    /// image bounds. `imageSize` mirrors numpy's `image.shape` ordering
    /// `(height, width)`. Port of `increase_size_in_each_dimension`.
    func increaseSizeInEachDimension(_ increase: Int, imageSize: (height: Int, width: Int)) -> BoundingBox {
        BoundingBox(
            box: (
                x1: max(box.x1 - increase, 0),
                y1: max(box.y1 - increase, 0),
                x2: min(box.x2 + increase, imageSize.width),
                y2: min(box.y2 + increase, imageSize.height)
            ),
            contours: contours,
            debugId: debugId
        )
    }

    static func == (lhs: BoundingBox, rhs: BoundingBox) -> Bool {
        lhs.box == rhs.box
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(box.x1)
        hasher.combine(box.y1)
        hasher.combine(box.x2)
        hasher.combine(box.y2)
    }
}

// MARK: - AngledBoundingBox

/// Abstract base for rotated boxes/ellipses — port of `AngledBoundingBox`.
///
/// The initializer normalizes the incoming angle into `(-45, 45]`-ish range,
/// swapping width/height for the >45° / <-45° cases, mirroring the Python
/// `if angle > 135 ... elif ...` cascade. Equality/hash are by `box`, and two
/// instances of DIFFERENT subclasses compare equal if their `box` matches
/// (matching Python, which only compares `self.box`).
class AngledBoundingBox: AnyPolygon, Hashable {
    /// Debug identifier carried through transformations.
    let debugId: Int
    /// Source contour points for this box.
    let contours: [CV.Point]
    /// Normalized rotated rectangle (Python `self.box`).
    let box: CV.RotatedRect
    /// Centre point (`box.center`).
    let center: CV.PointF
    /// `(width, height)` of the normalized box (`box.size`).
    let size: CV.Size
    /// Rotation angle in degrees (`box.angle`).
    let angle: Double
    /// Axis-aligned corner extents derived from the normalized box.
    let topLeft: CV.PointF
    let bottomLeft: CV.PointF
    let topRight: CV.PointF
    let bottomRight: CV.PointF

    init(box rawBox: CV.RotatedRect, contours: [CV.Point], polygon: [CV.Point], debugId: Int = 0) {
        self.debugId = debugId
        self.contours = contours

        var angle = rawBox.angle
        let center = rawBox.center
        let s = rawBox.size
        let normalized: CV.RotatedRect
        if angle > 135 {
            angle -= 180
            normalized = CV.RotatedRect(center: center, size: CV.Size(width: s.width, height: s.height), angle: angle)
        } else if angle < -135 {
            angle += 180
            normalized = CV.RotatedRect(center: center, size: CV.Size(width: s.width, height: s.height), angle: angle)
        } else if angle > 45 {
            angle -= 90
            normalized = CV.RotatedRect(center: center, size: CV.Size(width: s.height, height: s.width), angle: angle)
        } else if angle < -45 {
            angle += 90
            normalized = CV.RotatedRect(center: center, size: CV.Size(width: s.height, height: s.width), angle: angle)
        } else {
            normalized = CV.RotatedRect(center: center, size: CV.Size(width: s.width, height: s.height), angle: angle)
        }

        self.box = normalized
        self.center = normalized.center
        self.size = normalized.size
        self.angle = normalized.angle
        let edges = calculateEdgesOfRotatedRectangle(normalized)
        self.topLeft = edges.topLeft
        self.bottomLeft = edges.bottomLeft
        self.topRight = edges.topRight
        self.bottomRight = edges.bottomRight
        super.init(polygon: polygon)
    }

    /// Full overlap test: a cheap distance pre-check, then a polygon overlap test.
    /// Port of `is_overlapping`.
    func isOverlapping(_ other: AnyPolygon) -> Bool {
        if !canShapesPossiblyTouch(other) {
            return false
        }
        return doPolygonsOverlap(polygon, other.polygon)
    }

    /// Port of `is_overlapping_with_any`.
    func isOverlappingWithAny(_ others: [AngledBoundingBox]) -> Bool {
        for other in others where isOverlapping(other) {
            return true
        }
        return false
    }

    /// Fast rejection test: if the centre distance exceeds the sum of the major
    /// axes the shapes cannot touch. Port of `_can_shapes_possibly_touch`.
    func canShapesPossiblyTouch(_ other: AnyPolygon) -> Bool {
        let center1 = box.center
        let axes1 = box.size
        let center2: CV.PointF
        let axes2: CV.Size
        if let bb = other as? BoundingBox {
            center2 = bb.rotatedBox.center
            axes2 = bb.rotatedBox.size
        } else if let ab = other as? AngledBoundingBox {
            center2 = ab.box.center
            axes2 = ab.box.size
        } else {
            fatalError("Unknown type \(type(of: other))")
        }
        let majorAxis1 = max(axes1.width, axes1.height)
        let majorAxis2 = max(axes2.width, axes2.height)
        let dx = center1.x - center2.x
        let dy = center1.y - center2.y
        let distance = (dx * dx + dy * dy).squareRoot()
        if distance > majorAxis1 + majorAxis2 {
            return false
        }
        return true
    }

    static func == (lhs: AngledBoundingBox, rhs: AngledBoundingBox) -> Bool {
        lhs.box == rhs.box
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(box.center.x)
        hasher.combine(box.center.y)
        hasher.combine(box.size.width)
        hasher.combine(box.size.height)
        hasher.combine(box.angle)
    }
}

// MARK: - RotatedBoundingBox

/// Rotated rectangle around a connected component — port of `RotatedBoundingBox`.
final class RotatedBoundingBox: AngledBoundingBox {
    init(box: CV.RotatedRect, contours: [CV.Point], debugId: Int = 0) {
        // Polygon comes from the RAW box (matching Python, which calls
        // `cv2.boxPoints(box)` before the base class normalizes the angle).
        let poly = CV.boxPoints(box).map { CV.Point(Int($0.x), Int($0.y)) }
        super.init(box: box, contours: contours, polygon: poly, debugId: debugId)
    }

    /// Rotated-rect intersection test (port of `is_intersecting`). Uses the same
    /// cheap pre-check as overlap, then OpenCV's rotated-rect intersection.
    func isIntersecting(_ other: RotatedBoundingBox) -> Bool {
        if !canShapesPossiblyTouch(other) {
            return false
        }
        return CV.rotatedRectanglesIntersect(box, other.box)
    }

    /// Returns a copy whose width/height are at least the given minimums.
    func ensureMinDimension(_ minWidth: Int, _ minHeight: Int) -> RotatedBoundingBox {
        RotatedBoundingBox(
            box: CV.RotatedRect(
                center: box.center,
                size: CV.Size(width: max(box.size.width, Double(minWidth)), height: max(box.size.height, Double(minHeight))),
                angle: box.angle
            ),
            contours: contours,
            debugId: debugId
        )
    }

    /// Inflates both dimensions by `thickness` (no-op for non-positive values).
    /// The centre is intentionally NOT moved (Python notes downstream depends on this).
    func makeBoxThicker(_ thickness: Int) -> RotatedBoundingBox {
        if thickness <= 0 {
            return self
        }
        return RotatedBoundingBox(
            box: CV.RotatedRect(
                center: box.center,
                size: CV.Size(width: box.size.width + Double(thickness), height: box.size.height + Double(thickness)),
                angle: box.angle
            ),
            contours: contours,
            debugId: debugId
        )
    }

    /// Shifts the box horizontally by `xDelta`. Port of `move_to_x_horizontal_by`.
    func moveToXHorizontalBy(_ xDelta: Int) -> RotatedBoundingBox {
        let newX = center.x + Double(xDelta)
        return RotatedBoundingBox(
            box: CV.RotatedRect(center: CV.PointF(newX, center.y), size: box.size, angle: box.angle),
            contours: contours,
            debugId: debugId
        )
    }

    /// Adds `thickness` to the height, keeping the top edge fixed.
    func makeBoxTaller(_ thickness: Int) -> RotatedBoundingBox {
        RotatedBoundingBox(
            box: CV.RotatedRect(
                center: box.center,
                size: CV.Size(width: box.size.width, height: box.size.height + Double(thickness)),
                angle: box.angle
            ),
            contours: contours,
            debugId: debugId
        )
    }

    /// Adds `thickness` to the height while keeping the centre roughly fixed
    /// (centre moves up by `thickness // 2`). Port of `make_box_taller_keep_center`.
    func makeBoxTallerKeepCenter(_ thickness: Int) -> RotatedBoundingBox {
        // Python uses floor division `thickness // 2`.
        let half = Int((Double(thickness) / 2).rounded(.down))
        return RotatedBoundingBox(
            box: CV.RotatedRect(
                center: CV.PointF(box.center.x, box.center.y - Double(half)),
                size: CV.Size(width: box.size.width, height: box.size.height + Double(thickness)),
                angle: box.angle
            ),
            contours: contours,
            debugId: debugId
        )
    }

    /// Extrapolates the box's centre-line y-value at the given x. Port of
    /// `get_center_extrapolated`.
    func getCenterExtrapolated(_ x: Double) -> Double {
        (x - box.center.x) * tan(box.angle / 180 * .pi) + box.center.y
    }

    /// Whether two near-horizontal line boxes line up (within tolerance) when
    /// their centre-lines are extrapolated to a shared midpoint. Port of
    /// `is_overlapping_extrapolated`.
    func isOverlappingExtrapolated(_ other: RotatedBoundingBox, unitSize: Double) -> Bool {
        let left: RotatedBoundingBox
        let right: RotatedBoundingBox
        if box.center.x > other.box.center.x {
            left = other
            right = self
        } else {
            left = self
            right = other
        }

        let centerX = (left.center.x + right.center.x) * 0.5

        let tolerance = Constants.toleranceForStaffLineDetection(unitSize)
        let maxGap = Constants.maxLineGapSize(unitSize)

        // `size[0] // 2` in Python is a float floor division.
        let leftHalfWidth = (left.size.width / 2).rounded(.down)
        let rightHalfWidth = (right.size.width / 2).rounded(.down)
        if centerX - left.center.x - leftHalfWidth > maxGap
            || right.center.x - centerX - rightHalfWidth > maxGap {
            return false
        }

        let leftAngle = tan(left.box.angle * .pi / 180.0)
        let rightAngle = tan(right.box.angle * .pi / 180.0)

        let leftY = (centerX - left.box.center.x) * leftAngle + left.box.center.y
        let rightY = (centerX - right.box.center.x) * rightAngle + right.box.center.y

        return abs(leftY - rightY) <= tolerance
    }

    /// Converts this rotated box to an axis-aligned `BoundingBox` using its
    /// top-left/bottom-right extents. Port of `to_bounding_box`.
    func toBoundingBox() -> BoundingBox {
        BoundingBox(
            box: (
                x1: Int(topLeft.x),
                y1: Int(topLeft.y),
                x2: Int(bottomRight.x),
                y2: Int(bottomRight.y)
            ),
            contours: contours,
            debugId: debugId
        )
    }
}

// MARK: - BoundingEllipse

/// Fitted ellipse hull around a connected component — port of `BoundingEllipse`.
final class BoundingEllipse: AngledBoundingBox {
    init(box: CV.RotatedRect, contours: [CV.Point], debugId: Int = 0) {
        // Polygon comes from the RAW box via `ellipse2Poly` (matching Python).
        let poly = CV.ellipse2Poly(
            center: CV.Point(Int(box.center.x), Int(box.center.y)),
            axes: CV.Point(Int(box.size.width / 2), Int(box.size.height / 2)),
            angle: Int(box.angle),
            arcStart: 0,
            arcEnd: 360,
            delta: 1
        )
        super.init(box: box, contours: contours, polygon: poly, debugId: debugId)
    }

    /// Inflates both dimensions by `thickness`, returning a new ellipse.
    func makeBoxThicker(_ thickness: Int) -> BoundingEllipse {
        BoundingEllipse(
            box: CV.RotatedRect(
                center: box.center,
                size: CV.Size(width: box.size.width + Double(thickness), height: box.size.height + Double(thickness)),
                angle: box.angle
            ),
            contours: contours,
            debugId: debugId
        )
    }

    /// Adds `thickness` to the height, returning a `RotatedBoundingBox`
    /// (matching Python's return type for `BoundingEllipse.make_box_taller`).
    func makeBoxTaller(_ thickness: Int) -> RotatedBoundingBox {
        RotatedBoundingBox(
            box: CV.RotatedRect(
                center: box.center,
                size: CV.Size(width: box.size.width, height: box.size.height + Double(thickness)),
                angle: box.angle
            ),
            contours: contours,
            debugId: debugId
        )
    }
}

// MARK: - Detection entry points

/// Detects rotated bounding boxes for every contour in a binary mask.
/// Port of `create_rotated_bounding_boxes`.
func createRotatedBoundingBoxes(
    _ image: GrayscaleImage,
    skipMerging: Bool = false,
    minSize: (Int, Int)? = nil,
    maxSize: (Int, Int)? = nil,
    thickenBoxes: Int? = nil
) -> [RotatedBoundingBox] {
    let contours = CV.findContours(image, mode: .tree).contours
    var boxes: [RotatedBoundingBox] = []
    for (i, contour) in contours.enumerated() {
        let fitBox = CV.minAreaRect(contour)
        if !hasBoxValidSize(fitBox) {
            continue
        }
        let box = RotatedBoundingBox(box: fitBox, contours: contour, debugId: i)
        if let minSize, box.size.width < Double(minSize.0) || box.size.height < Double(minSize.1) {
            continue
        }
        if let maxSize {
            if maxSize.0 > 0 && box.size.width > Double(maxSize.0) {
                continue
            }
            if maxSize.1 > 0 && box.size.height > Double(maxSize.1) {
                continue
            }
        }
        boxes.append(box)
    }
    if skipMerging {
        return boxes
    }
    if let thickenBoxes {
        boxes = boxes.map { $0.makeBoxThicker(thickenBoxes) }
    }
    return getBoxForWholeGroup(mergeOverlayingBoundingBoxes(boxes))
}

/// Builds a single rotated box from one contour. Port of `create_rotated_bounding_box`.
func createRotatedBoundingBox(_ contour: [CV.Point], debugId: Int) -> RotatedBoundingBox {
    let box = CV.minAreaRect(contour)
    return RotatedBoundingBox(box: box, contours: contour, debugId: debugId)
}

/// Detects (predominantly horizontal) line segments via Hough transform.
/// Port of `create_lines`.
func createLines(
    _ image: GrayscaleImage,
    threshold: Int = 100,
    minLineLength: Int = 100,
    maxLineGap: Int = 10,
    skipMerging: Bool = false
) -> [RotatedBoundingBox] {
    let lines = CV.houghLinesP(
        image,
        rho: 1,
        theta: .pi / 180,
        threshold: threshold,
        minLineLength: Double(minLineLength),
        maxLineGap: Double(maxLineGap)
    )
    var boxes: [RotatedBoundingBox] = []
    for (i, line) in lines.enumerated() {
        let contour = [line.0, line.1]
        let box = CV.minAreaRect(contour)
        if box.size.width > box.size.height {
            boxes.append(RotatedBoundingBox(box: box, contours: contour, debugId: i))
        }
    }
    if skipMerging {
        return boxes
    }
    return getBoxForWholeGroup(mergeOverlayingBoundingBoxes(boxes))
}

/// Detects fitted ellipses for every sufficiently large contour in a binary mask.
/// Port of `create_bounding_ellipses`.
func createBoundingEllipses(
    _ image: GrayscaleImage,
    skipMerging: Bool = false,
    minSize: (Int, Int)? = nil,
    maxSize: (Int, Int)? = nil
) -> [BoundingEllipse] {
    let contours = CV.findContours(image, mode: .tree).contours
    var boxes: [BoundingEllipse] = []
    for (i, contour) in contours.enumerated() {
        // OpenCV's fitEllipse requires at least 5 points.
        let minLengthToFitEllipse = 5
        if contour.count < minLengthToFitEllipse {
            continue
        }
        let fitBox = CV.fitEllipse(contour)
        if !hasBoxValidSize(fitBox) {
            continue
        }
        let box = BoundingEllipse(box: fitBox, contours: contour, debugId: i)
        if let minSize, box.size.width < Double(minSize.0) || box.size.height < Double(minSize.1) {
            continue
        }
        if let maxSize, box.size.width > Double(maxSize.0) || box.size.height > Double(maxSize.1) {
            continue
        }
        boxes.append(box)
    }
    if skipMerging {
        return boxes
    }
    return getEllipseForWholeGroup(mergeOverlayingBoundingBoxes(boxes))
}

// MARK: - Grouping / merging

/// `True` if any box in `group1` overlaps any box in `group2`. Port of `_do_groups_overlap`.
private func doGroupsOverlap(_ group1: [AngledBoundingBox], _ group2: [AngledBoundingBox]) -> Bool {
    for box1 in group1 {
        for box2 in group2 where box1.isOverlapping(box2) {
            return true
        }
    }
    return false
}

/// Recursive pairwise group merger. Port of `_merge_groups_recursive`.
///
/// NOTE: This is faithfully ported but unused — `mergeOverlayingBoundingBoxes`
/// uses the union-find path below, exactly as Python does.
private func mergeGroupsRecursive(_ groups: [[AngledBoundingBox]], step: Int) -> [[AngledBoundingBox]] {
    let stepLimit = 10
    if step > stepLimit {
        print("Too many steps in mergeGroupsRecursive, giving back current results")
        return groups
    }
    var numberOfChanges = 0
    var merged: [[AngledBoundingBox]] = []
    var usedGroups = Set<Int>()
    for (i, group) in groups.enumerated() {
        var matchFound = false
        if usedGroups.contains(i) {
            continue
        }
        var j = i + 1
        while j < groups.count {
            if usedGroups.contains(j) {
                j += 1
                continue
            }
            let otherGroup = groups[j]
            if doGroupsOverlap(group, otherGroup) {
                merged.append(group + otherGroup)
                numberOfChanges += 1
                usedGroups.insert(j)
                matchFound = true
                break
            }
            j += 1
        }
        if !matchFound {
            merged.append(group)
        }
    }

    if numberOfChanges == 0 {
        return merged
    } else {
        return mergeGroupsRecursive(merged, step: step + 1)
    }
}

/// Fits one ellipse per merged group from the group's combined contours.
/// Port of `_get_ellipse_for_whole_group`.
private func getEllipseForWholeGroup(_ groups: [[AngledBoundingBox]]) -> [BoundingEllipse] {
    var result: [BoundingEllipse] = []
    for group in groups {
        let completeContour = group.flatMap { $0.contours }
        let box = CV.minAreaRect(completeContour)
        result.append(BoundingEllipse(box: box, contours: completeContour))
    }
    return result
}

/// Fits one rotated box per merged group from the group's combined contours.
/// Port of `_get_box_for_whole_group`.
private func getBoxForWholeGroup(_ groups: [[AngledBoundingBox]]) -> [RotatedBoundingBox] {
    var result: [RotatedBoundingBox] = []
    for group in groups {
        let completeContour = group.flatMap { $0.contours }
        let box = CV.minAreaRect(completeContour)
        result.append(RotatedBoundingBox(box: box, contours: completeContour))
    }
    return result
}

/// Disjoint-set (union-find) with path compression and union by rank.
/// Port of the Python `UnionFind` class.
final class UnionFind {
    private var parent: [Int]
    private var rank: [Int]

    init(_ n: Int) {
        parent = Array(0..<n)
        rank = [Int](repeating: 0, count: n)
    }

    /// Recursive find with path compression (faithful to Python).
    func find(_ x: Int) -> Int {
        if parent[x] != x {
            parent[x] = find(parent[x])
        }
        return parent[x]
    }

    func union(_ x: Int, _ y: Int) {
        let rootX = find(x)
        let rootY = find(y)
        if rootX != rootY {
            if rank[rootX] > rank[rootY] {
                parent[rootY] = rootX
            } else if rank[rootX] < rank[rootY] {
                parent[rootX] = rootY
            } else {
                parent[rootY] = rootX
                rank[rootX] += 1
            }
        }
    }
}

/// Union-find based grouping of overlapping groups. Port of `_merge_groups_optimized`.
private func mergeGroupsOptimized(_ groups: [[AngledBoundingBox]]) -> [[AngledBoundingBox]] {
    let n = groups.count
    let uf = UnionFind(n)

    for i in 0..<n {
        var j = i + 1
        while j < n {
            if doGroupsOverlap(groups[i], groups[j]) {
                uf.union(i, j)
            }
            j += 1
        }
    }

    // Preserve first-seen root order so output ordering matches Python's dict
    // (insertion-ordered) iteration.
    var mergedGroups: [Int: [AngledBoundingBox]] = [:]
    var rootOrder: [Int] = []
    for i in 0..<n {
        let root = uf.find(i)
        if mergedGroups[root] == nil {
            mergedGroups[root] = []
            rootOrder.append(root)
        }
        mergedGroups[root]?.append(contentsOf: groups[i])
    }

    return rootOrder.map { mergedGroups[$0]! }
}

/// Wraps each box in its own singleton group and merges overlapping ones.
/// Port of `merge_overlaying_bounding_boxes`.
func mergeOverlayingBoundingBoxes(_ boxes: [AngledBoundingBox]) -> [[AngledBoundingBox]] {
    var initialGroups: [[AngledBoundingBox]] = []
    for box in boxes {
        initialGroups.append([box])
    }
    return mergeGroupsOptimized(initialGroups)
}

// MARK: - Image cropping helper

/// Slices a sub-image with the same clamping/ordering as `homr/image_utils.crop_image`
/// (`crop_image_and_return_new_top`). Coordinates are clamped into `[0, dim - 1]`
/// and the end indices are exclusive (numpy slicing semantics).
private func cropImage(_ image: GrayscaleImage, _ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double) -> GrayscaleImage {
    func limitX(_ v: Double) -> Int { max(0, min(image.width - 1, Int(v.rounded(.toNearestOrEven)))) }
    func limitY(_ v: Double) -> Int { max(0, min(image.height - 1, Int(v.rounded(.toNearestOrEven)))) }

    let xMin = min(x1, x2)
    let xMax = max(x1, x2)
    let yMin = min(y1, y2)
    let yMax = max(y1, y2)
    let x1l = limitX(xMin)
    let y1l = limitY(yMin)
    let x2l = limitX(xMax)
    let y2l = limitY(yMax)

    let w = max(0, x2l - x1l)
    let h = max(0, y2l - y1l)
    var out = [UInt8](repeating: 0, count: w * h)
    if w > 0 && h > 0 {
        for row in 0..<h {
            let srcStart = (y1l + row) * image.width + x1l
            let dstStart = row * w
            for col in 0..<w {
                out[dstStart + col] = image.pixels[srcStart + col]
            }
        }
    }
    return GrayscaleImage(pixels: out, width: w, height: h)
}
