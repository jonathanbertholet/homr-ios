import Foundation

// MARK: - StaffDewarping
//
// Port of `homr/staff_dewarping.py`.
//
// The Python module corrects gentle vertical waviness in a cropped staff image
// before inference. It samples the detected staff line (the 3rd line, `y[2]`) at
// a grid of x positions, builds a set of "span" points (where the line actually
// is) and "optimal" points (where the line should be — a flat average), then
// estimates a piecewise-affine transform (Delaunay triangulation + per-triangle
// `cv2.getAffineTransform` + `cv2.warpAffine`) that straightens the image.
//
// ───────────────────────────────────────────────────────────────────────────
// TODO / SIMPLIFICATION (identity dewarp):
// `PiecewiseAffineTransform` below is an IDENTITY fallback. Reproducing OpenCV's
// `Subdiv2D` Delaunay triangulation plus per-triangle `warpAffine` in pure Swift
// is a large undertaking, and homr ITSELF already falls back to an identity
// transform whenever dewarping fails: `dewarp_staff_image` returns
// `StaffDewarping(None)` on any exception, and both `StaffDewarping.dewarp` and
// `StaffDewarping.dewarp_point` return their input UNCHANGED when `tform is None`.
// We therefore implement `estimate` (it just records the point sets) and make
// `transformPoint` / `warpImage` return the input unchanged. The pipeline stays
// correct — it simply skips the (usually minor) warp correction. A full
// piecewise-affine implementation can be added later without changing any of the
// public signatures here. We deliberately do NOT ship a half-broken Delaunay.
// ───────────────────────────────────────────────────────────────────────────
//
// The span/optimal/transform-building helpers (`calculateSpanAndOptimalPoints`,
// `isPointOnImage`, `calculateDewarpTransformation`, `dewarpStaffImage`) ARE
// ported faithfully, so the point geometry matches Python; only the final warp
// is identity.
//
// NOTE: the training-only `warp_image_randomly` / `warp_image_array_randomly*`
// functions are intentionally NOT ported — they are dataset augmentation used
// during model training, not part of the inference pipeline.

// MARK: - PiecewiseAffineTransform (identity fallback)

/// Piecewise affine transform — IDENTITY fallback (see file header).
///
/// `estimate` records the source / destination point sets (so the public
/// interface mirrors Python's `skimage`-style API), but `transformPoint` and
/// `warpImage` return their input unchanged. This matches homr's own behaviour
/// whenever its real transform construction fails.
final class PiecewiseAffineTransform {
    /// Recorded source control points (kept for parity / future use).
    private(set) var srcPoints: [CV.PointF] = []
    /// Recorded destination control points (kept for parity / future use).
    private(set) var dstPoints: [CV.PointF] = []

    init() {}

    /// Port of `estimate`: store the control-point correspondence.
    ///
    /// The real implementation would also triangulate `src` and precompute a
    /// per-triangle affine matrix here; the identity fallback only records them.
    func estimate(src: [CV.PointF], dst: [CV.PointF]) {
        self.srcPoints = src
        self.dstPoints = dst
    }

    /// Port of `transform_point` — IDENTITY: returns the point unchanged.
    func transformPoint(_ point: CV.PointF) -> CV.PointF {
        point
    }

    /// Port of `warp_image` — IDENTITY: returns the image unchanged.
    func warpImage(_ image: GrayscaleImage, fillColor: UInt8 = 1, order: Int = 1) -> GrayscaleImage {
        _ = fillColor
        _ = order
        return image
    }
}

// MARK: - StaffDewarping

/// Port of `StaffDewarping`.
///
/// Wraps an optional `PiecewiseAffineTransform`. With `nil` (homr's
/// `StaffDewarping(None)`) both operations are pass-through; with the identity
/// transform above they are likewise pass-through (see file header).
final class StaffDewarping {
    private let tform: PiecewiseAffineTransform?

    init(_ tform: PiecewiseAffineTransform?) {
        self.tform = tform
    }

    /// Port of `dewarp`: warp the image, or return it unchanged when there is no
    /// transform.
    func dewarp(_ image: GrayscaleImage, fillColor: UInt8 = 1, order: Int = 1) -> GrayscaleImage {
        guard let tform = tform else { return image }
        return tform.warpImage(image, fillColor: fillColor, order: order)
    }

    /// Port of `dewarp_point`: transform a point, or return it unchanged when
    /// there is no transform.
    func dewarpPoint(_ point: CV.PointF) -> CV.PointF {
        guard let tform = tform else { return point }
        return tform.transformPoint(point)
    }
}

// MARK: - Geometry helpers (ported faithfully)

/// Port of `is_point_on_image`: a point counts as "on the image" only if it sits
/// at least `margin` (10) pixels inside every edge.
func isPointOnImage(_ pts: CV.Point, _ image: GrayscaleImage) -> Bool {
    let margin = 10
    let width = image.width
    let height = image.height
    if pts.x < margin || pts.x > width - margin || pts.y < margin || pts.y > height - margin {
        return false
    }
    return true
}

/// Port of `calculate_span_and_optimal_points`.
///
/// Walks a coarse grid of rows; for each row samples the staff's 3rd line
/// (`get_at(x).y[2]`) every 80 px, offset so the sampled rows track the staff's
/// vertical drift relative to the first sampled offset. Rows with more than two
/// on-image points contribute a "span" line (actual points) and an "optimal"
/// line (same x, flattened to the row's average y).
///
/// Bug-for-bug: Python's `if not first_y_offset:` is truthy for BOTH `None` and
/// `0.0`, so a first offset of exactly 0 keeps re-seeding; replicated here.
func calculateSpanAndOptimalPoints(
    staff: Staff, image: GrayscaleImage
) -> (span: [[CV.Point]], optimal: [[CV.Point]]) {
    var spanPoints: [[CV.Point]] = []
    var optimalPoints: [[CV.Point]] = []
    var firstYOffset: Double?
    let numberOfYIntervals = 6

    // `int(image.shape[0] / number_of_y_intervals)` — integer step; bail if 0.
    let step = image.height / numberOfYIntervals
    if step == 0 {
        return (spanPoints, optimalPoints)
    }

    var y = 2
    while y < image.height - 2 {
        var linePoints: [CV.Point] = []
        var x = 2
        while x < image.width {
            if let yValues = staff.getAt(Double(x)) {
                let yOffset = yValues.y[2]
                let yDelta: Int
                // Truthy on None OR 0.0 (replicated quirk).
                if firstYOffset == nil || firstYOffset == 0.0 {
                    firstYOffset = yOffset
                    yDelta = 0
                } else {
                    yDelta = homrInt(yOffset - firstYOffset!)
                }
                let point = CV.Point(x, y + yDelta)
                if isPointOnImage(point, image) {
                    linePoints.append(point)
                }
            }
            x += 80
        }

        let minimumNumberOfPoints = 2
        if linePoints.count > minimumNumberOfPoints {
            let averageY = Double(linePoints.reduce(0) { $0 + $1.y }) / Double(linePoints.count)
            spanPoints.append(linePoints)
            optimalPoints.append(linePoints.map { CV.Point($0.x, homrInt(averageY)) })
        }
        y += step
    }

    return (spanPoints, optimalPoints)
}

/// Port of `calculate_dewarp_transformation`.
///
/// Extends every line to the full image width, prepends/appends the top/bottom
/// image edges, concatenates the lines into flat control-point lists, and
/// estimates the (identity) piecewise-affine transform from source → destination.
func calculateDewarpTransformation(
    image: GrayscaleImage,
    source: [[CV.Point]],
    destination: [[CV.Point]]
) -> StaffDewarping {
    // Mirrors the nested `add_first_and_last_point_to_every_line`: clamp every
    // line to x = 0 ... width using that line's first / last y.
    func addFirstAndLastPointToEveryLine(_ lines: inout [[CV.Point]]) {
        for i in lines.indices {
            let firstY = lines[i][0].y
            let lastY = lines[i][lines[i].count - 1].y
            lines[i].insert(CV.Point(0, firstY), at: 0)
            lines[i].append(CV.Point(image.width, lastY))
        }
    }

    // Mirrors the nested `add_image_edges_to_lines`: add the top and bottom
    // image borders as their own lines so the warp is anchored at the frame.
    func addImageEdgesToLines(_ lines: inout [[CV.Point]]) {
        lines.insert([CV.Point(0, 0), CV.Point(image.width, 0)], at: 0)
        lines.append([CV.Point(0, image.height), CV.Point(image.width, image.height)])
    }

    var src = source
    var dst = destination

    addFirstAndLastPointToEveryLine(&src)
    addFirstAndLastPointToEveryLine(&dst)

    addImageEdgesToLines(&src)
    addImageEdgesToLines(&dst)

    // np.concatenate → flatten into single point lists, as CV.PointF for estimate.
    let srcConc = src.flatMap { $0 }.map { CV.PointF(Double($0.x), Double($0.y)) }
    let dstConc = dst.flatMap { $0 }.map { CV.PointF(Double($0.x), Double($0.y)) }

    let tform = PiecewiseAffineTransform()
    tform.estimate(src: srcConc, dst: dstConc)
    return StaffDewarping(tform)
}

/// Port of `dewarp_staff_image` (debug drawing dropped).
///
/// Builds the span / optimal control points for the staff and returns the
/// dewarping transform. Python wraps this in a try/except that falls back to
/// `StaffDewarping(None)`; the pure-Swift port cannot throw here, and the
/// transform it returns is the identity fallback regardless (see file header).
func dewarpStaffImage(image: GrayscaleImage, staff: Staff, index: Int) -> StaffDewarping {
    _ = index
    let (spanPoints, optimalPoints) = calculateSpanAndOptimalPoints(staff: staff, image: image)
    return calculateDewarpTransformation(image: image, source: spanPoints, destination: optimalPoints)
}
