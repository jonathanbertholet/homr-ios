import Foundation

/// Native port of OpenCV's `cv2.ellipse2Poly`.
///
/// `bounding_boxes.py` calls `cv2.ellipse2Poly` inside `BoundingEllipse.__init__`
/// to turn a fitted ellipse into a closed integer polygon that is later used by
/// the point-in-polygon overlap test (`do_polygons_overlap`). This implementation
/// matches OpenCV 4.x semantics: it samples the ellipse from `arcStart` to
/// `arcEnd` degrees in `delta`-degree steps, rotates the samples by `angle`,
/// translates them by `center`, and rounds to integer points (with consecutive
/// duplicate points collapsed, exactly like OpenCV).
extension CV {

    /// Approximates an elliptic arc with a polyline of integer points.
    ///
    /// - Parameters:
    ///   - center: Ellipse centre (integer pixel coordinate).
    ///   - axes: Half-axis lengths `(x, y)` — i.e. semi-axes, NOT full sizes.
    ///   - angle: Rotation of the ellipse, in whole degrees.
    ///   - arcStart: Starting angle of the arc, in whole degrees.
    ///   - arcEnd: Ending angle of the arc, in whole degrees.
    ///   - delta: Angular step between sampled vertices, in whole degrees.
    /// - Returns: Ordered polygon vertices (consecutive duplicates removed).
    static func ellipse2Poly(
        center: Point,
        axes: Point,
        angle: Int,
        arcStart: Int,
        arcEnd: Int,
        delta: Int
    ) -> [Point] {
        // Rotation of the ellipse's major axis, matching OpenCV's `alpha`.
        let alpha = Double(angle) * .pi / 180.0
        let cosAlpha = cos(alpha)
        let sinAlpha = sin(alpha)

        // Normalise the arc range exactly like OpenCV does before sampling.
        var start = arcStart
        var end = arcEnd
        if start > end { swap(&start, &end) }
        while start < 0 {
            start += 360
            end += 360
        }
        while end > 360 {
            start -= 360
            end -= 360
        }
        if end - start > 360 {
            start = 0
            end = 360
        }

        let step = delta == 0 ? 1 : delta
        var pts: [Point] = []
        // Sentinel that no real rounded point can equal, so the first sample is
        // always kept (mirrors OpenCV's `prevPt(INT_MIN, INT_MIN)`).
        var prev = Point(Int.min, Int.min)

        var i = start
        while i < end + step {
            // OpenCV clamps the final sample to `arcEnd` so the arc closes exactly.
            let angleI = min(i, end)
            let t = Double(angleI) * .pi / 180.0
            let x = Double(axes.x) * cos(t)
            let y = Double(axes.y) * sin(t)
            // Rotate by `alpha`, then translate to `center`.
            let px = Double(center.x) + x * cosAlpha - y * sinAlpha
            let py = Double(center.y) + x * sinAlpha + y * cosAlpha
            let pt = Point(
                Int(px.rounded(.toNearestOrEven)),
                Int(py.rounded(.toNearestOrEven))
            )
            if pt != prev {
                pts.append(pt)
                prev = pt
            }
            i += step
        }

        // Degenerate polygon: OpenCV returns the centre twice so callers always
        // receive at least a 2-point "line".
        if pts.count == 1 {
            pts = [center, center]
        }
        return pts
    }
}
