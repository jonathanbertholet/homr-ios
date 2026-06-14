import Foundation

// MARK: - Geometry primitives (native OpenCV 4.x port)
//
// This file re-implements the handful of `cv2` geometry helpers that the Python
// `homr` pipeline relies on, without any OpenCV dependency. Each function is a
// faithful port of the corresponding routine in opencv-python 4.13 (the version
// `homr` pins), so the golden-fixture harness can validate bit-for-bit parity.
//
// IMPORTANT parity notes:
//   * OpenCV performs most of this geometry in 32-bit `float`. We therefore use
//     Swift `Float` (== Float32) in exactly the same places OpenCV does, and only
//     widen to `Double` where OpenCV widens to `double`. This is what makes the
//     sub-pixel results match the reference values (e.g. 34.70587921142578).
//   * `minAreaRect` angle convention in this OpenCV build is the half-open range
//     [-90, 0) degrees (verified empirically and by the `CV_DbgCheck` in the
//     OpenCV source). See `minAreaRect` below for details.
//
// Shared value types (`CV.Point`, `CV.PointF`, ...) live in `CVTypes.swift`.

extension CV {

    // =========================================================================
    // MARK: boundingRect
    // =========================================================================

    /// Axis-aligned bounding box of an integer point set (`cv2.boundingRect`).
    ///
    /// OpenCV's right/bottom edges are *exclusive*, hence the `+1` on width and
    /// height: a single point yields a 1x1 rect. Mirrors
    /// `pointSetBoundingRect` in `imgproc/src/geometry.cpp`.
    static func boundingRect(_ points: [Point]) -> Rect {
        // OpenCV returns an empty rect for an empty input.
        guard let first = points.first else { return Rect(x: 0, y: 0, width: 0, height: 0) }

        var xmin = first.x, xmax = first.x
        var ymin = first.y, ymax = first.y
        for p in points {
            if p.x < xmin { xmin = p.x }
            if p.x > xmax { xmax = p.x }
            if p.y < ymin { ymin = p.y }
            if p.y > ymax { ymax = p.y }
        }
        // Width/height are inclusive of both extreme columns/rows → maxX-minX+1.
        return Rect(x: xmin, y: ymin, width: xmax - xmin + 1, height: ymax - ymin + 1)
    }

    // =========================================================================
    // MARK: contourArea
    // =========================================================================

    /// Absolute polygon area via the shoelace formula (`cv2.contourArea`).
    ///
    /// OpenCV casts every vertex to `float`, accumulates the cross products in
    /// `double`, multiplies by 0.5, then takes the absolute value (the default
    /// `oriented == false`). Mirrors `cv::contourArea` in
    /// `imgproc/src/shapedescr.cpp`.
    static func contourArea(_ contour: [Point]) -> Double {
        let n = contour.count
        if n == 0 { return 0.0 }

        // `prev` starts at the LAST vertex so the polygon is treated as closed.
        var prev = contour[n - 1]
        var a00 = 0.0
        for i in 0..<n {
            let p = contour[i]
            // Cross product of consecutive vertices (the shoelace term).
            // Vertices are small integers, so Double exactly represents them
            // (matching OpenCV's float→double promotion).
            a00 += Double(prev.x) * Double(p.y) - Double(p.x) * Double(prev.y)
            prev = p
        }
        a00 *= 0.5
        return abs(a00)
    }

    // =========================================================================
    // MARK: convexHull
    // =========================================================================

    /// Convex hull of an integer point set (`cv2.convexHull`, default flags:
    /// `clockwise == false`, `returnPoints == true`).
    ///
    /// This is a faithful port of OpenCV's Sklansky-based `cv::convexHull`
    /// (`imgproc/src/convhull.cpp`), including the final "cyclic shift" pass that
    /// canonicalises the starting vertex. Reproducing the exact vertex order
    /// matters because `minAreaRect` runs its rotating calipers on this output.
    static func convexHull(_ points: [Point]) -> [Point] {
        convexHullIndexed(points, clockwise: false).map { points[$0] }
    }

    /// Shared hull implementation returning ORIGINAL-input indices (in hull
    /// order). `clockwise` selects the winding, exactly like OpenCV's flag.
    /// `minAreaRect` reuses this with `clockwise == false`.
    private static func convexHullIndexed(_ points: [Point], clockwise: Bool) -> [Int] {
        let total = points.count
        if total == 0 { return [] }

        // --- sort pointers by (x asc, y asc, original-index asc) -------------
        // OpenCV sorts an array of pointers with a comparator whose final tie
        // break is the pointer address (== original order). We emulate that with
        // the original index, giving a deterministic total order.
        var sorted = Array(0..<total)
        sorted.sort { a, b in
            let pa = points[a], pb = points[b]
            if pa.x != pb.x { return pa.x < pb.x }
            if pa.y != pb.y { return pa.y < pb.y }
            return a < b
        }

        // Coordinate accessors keyed by *sorted position* (mirrors `array[pos]`).
        @inline(__always) func X(_ pos: Int) -> Int { points[sorted[pos]].x }
        @inline(__always) func Y(_ pos: Int) -> Int { points[sorted[pos]].y }

        // --- find min-y / max-y sorted positions ----------------------------
        var minyInd = 0, maxyInd = 0
        for i in 1..<total {
            let y = Y(i)
            if Y(minyInd) > y { minyInd = i }
            if Y(maxyInd) < y { maxyInd = i }
        }

        var hullbuf: [Int] = []   // holds sorted positions, later original indices

        // Degenerate case: every point is identical.
        if X(0) == X(total - 1) && Y(0) == Y(total - 1) {
            hullbuf = [0]
        } else {
            // Sklansky scan of one monotone chain. `nsign` filters edges by the
            // sign of the y-step; `sign2` is the required convexity sign.
            // Returns the list of sorted positions forming that chain.
            func sklansky(_ start: Int, _ end: Int, _ nsign: Int, _ sign2: Int) -> [Int] {
                let incr = end > start ? 1 : -1
                var pprev = start
                var pcur = pprev + incr
                var pnext = pcur + incr

                // Single point / coincident endpoints → trivial chain.
                if start == end || (X(start) == X(end) && Y(start) == Y(end)) {
                    return [start]
                }

                // Fixed-size scratch stack (over-allocated like OpenCV's buffer).
                var st = [Int](repeating: 0, count: total + 2)
                st[0] = pprev; st[1] = pcur; st[2] = pnext
                var stacksize = 3
                let endA = end + incr   // one-past-the-end sentinel

                while pnext != endA {
                    let by = Y(pnext) - Y(pcur)
                    if sign(by) != nsign {
                        // Vectors a = cur-prev, b = next-cur.
                        let ax = X(pcur) - X(pprev)
                        let ay = Y(pcur) - Y(pprev)
                        let bx = X(pnext) - X(pcur)
                        // convexity = a.y*b.x - a.x*b.y (Int64 to avoid overflow).
                        let convexity = Int64(ay) * Int64(bx) - Int64(ax) * Int64(by)
                        if sign64(convexity) == sign2 && (ax != 0 || ay != 0) {
                            // Keep `next`: advance the window and push.
                            pprev = pcur
                            pcur = pnext
                            pnext += incr
                            st[stacksize] = pnext
                            stacksize += 1
                        } else if pprev == start {
                            // Cannot pop past the start: slide `cur` forward.
                            pcur = pnext
                            st[1] = pcur
                            pnext += incr
                            st[2] = pnext
                        } else {
                            // Pop the offending vertex and re-test.
                            st[stacksize - 2] = pnext
                            pcur = pprev
                            pprev = st[stacksize - 4]
                            stacksize -= 1
                        }
                    } else {
                        // Wrong y-direction: just skip this vertex.
                        pnext += incr
                        st[stacksize - 1] = pnext
                    }
                }
                // OpenCV returns `--stacksize`, i.e. the count is stacksize-1.
                return Array(st[0..<(stacksize - 1)])
            }

            // --- upper half ------------------------------------------------
            var tl = sklansky(0, maxyInd, -1, 1)
            var tr = sklansky(total - 1, maxyInd, -1, -1)
            if !clockwise { swap(&tl, &tr) }

            for i in 0..<(tl.count - 1) { hullbuf.append(tl[i]) }
            if tr.count >= 2 {
                for i in stride(from: tr.count - 1, through: 1, by: -1) { hullbuf.append(tr[i]) }
            }
            let stopIdx = tr.count > 2 ? tr[1] : (tl.count > 2 ? tl[tl.count - 2] : -1)

            // --- lower half ------------------------------------------------
            var bl = sklansky(0, minyInd, 1, -1)
            var br = sklansky(total - 1, minyInd, 1, 1)
            if clockwise { swap(&bl, &br) }

            // Collinear-set guard: if the lower chain merely mirrors the upper
            // one, clamp it to the two extreme points (OpenCV's special case).
            if stopIdx >= 0 {
                let checkIdx: Int
                if bl.count > 2 {
                    checkIdx = bl[1]
                } else if bl.count + br.count > 2 {
                    let bi = 2 - bl.count
                    checkIdx = (bi >= 0 && bi < br.count) ? br[bi] : -1
                } else {
                    checkIdx = -1
                }
                let sameAsStop = checkIdx == stopIdx ||
                    (checkIdx >= 0 &&
                     X(checkIdx) == X(stopIdx) && Y(checkIdx) == Y(stopIdx))
                if sameAsStop {
                    if bl.count > 2 { bl = Array(bl.prefix(2)) }
                    if br.count > 2 { br = Array(br.prefix(2)) }
                }
            }

            for i in 0..<(bl.count - 1) { hullbuf.append(bl[i]) }
            if br.count >= 2 {
                for i in stride(from: br.count - 1, through: 1, by: -1) { hullbuf.append(br[i]) }
            }
        }

        // Convert sorted positions → original input indices.
        for i in 0..<hullbuf.count { hullbuf[i] = sorted[hullbuf[i]] }

        // --- canonical cyclic shift ----------------------------------------
        // OpenCV rotates the index sequence so that it forms an ascending (or
        // descending) run of original indices, fixing the hull's starting point.
        let nout = hullbuf.count
        if nout >= 3 {
            var minIdx = 0, maxIdx = 0, lt = 0
            var broke = false
            for i in 1..<nout {
                let idx = hullbuf[i]
                lt += (hullbuf[i - 1] < idx) ? 1 : 0
                if lt > 1 && lt <= i - 2 { broke = true; break }
                if idx < hullbuf[minIdx] { minIdx = i }
                if idx > hullbuf[maxIdx] { maxIdx = i }
            }
            _ = broke
            let mmdist = abs(maxIdx - minIdx)
            if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt >= nout - 2) {
                let ascending = (maxIdx + 1) % nout == minIdx
                let i0 = ascending ? minIdx : maxIdx
                if i0 > 0 {
                    var scratch = [Int](repeating: 0, count: nout)
                    var j = i0
                    var i = 0
                    while i < nout {
                        let currIdx = hullbuf[j]
                        scratch[i] = currIdx
                        let nextJ = j + 1 < nout ? j + 1 : 0
                        let nextIdx = hullbuf[nextJ]
                        if i < nout - 1 && (ascending != (currIdx < nextIdx)) { break }
                        j = nextJ
                        i += 1
                    }
                    if i == nout { hullbuf = scratch }
                }
            }
        }

        return hullbuf
    }

    // =========================================================================
    // MARK: minAreaRect
    // =========================================================================

    /// Minimum-area enclosing rotated rectangle (`cv2.minAreaRect`).
    ///
    /// Algorithm (mirrors `cv::minAreaRect` + `rotatingCalipers` in
    /// `imgproc/src/rotcalipers.cpp`):
    ///   1. Convex hull (clockwise == false, same as OpenCV).
    ///   2. Rotating calipers over the hull to minimise area.
    ///   3. Derive center / size / angle.
    ///
    /// Angle convention for this OpenCV build is **[-90, 0) degrees** — the
    /// reference `RotatedRect` stores `width = |out2|`, `height = |out1|`, and
    /// `angle = -atan2(out1.x, out1.y)`. (Note: this differs from the optimistic
    /// "[0, 90)" wording in `CVTypes.swift`; the real library returns [-90, 0).)
    static func minAreaRect(_ points: [Point]) -> RotatedRect {
        // Hull as float points (OpenCV converts the integer hull to CV_32F).
        let hullIdx = convexHullIndexed(points, clockwise: false)
        let hull: [(x: Float, y: Float)] = hullIdx.map { (Float(points[$0].x), Float(points[$0].y)) }
        let n = hull.count

        // Default for a degenerate / single-point box.
        var angle = -Double.pi / 2   // → -90 degrees after conversion
        var cx: Float = 0, cy: Float = 0
        var width: Float = 0, height: Float = 0

        if n > 2 {
            let (o0, o1, o2) = rotatingCalipersMinArea(hull)
            cx = o0.x + (o1.x + o2.x) * 0.5
            cy = o0.y + (o1.y + o2.y) * 0.5
            width = Float((Double(o2.x) * Double(o2.x) + Double(o2.y) * Double(o2.y)).squareRoot())
            height = Float((Double(o1.x) * Double(o1.x) + Double(o1.y) * Double(o1.y)).squareRoot())
            if o1.x == 0.0 && o1.y > 0.0 {
                swap(&width, &height)   // angle stays at the -90 default
            } else {
                angle = -atan2(Double(o1.x), Double(o1.y))
            }
        } else if n == 2 {
            // A segment: center is the midpoint, one side has zero length.
            cx = (hull[0].x + hull[1].x) * 0.5
            cy = (hull[0].y + hull[1].y) * 0.5
            let dx = Double(hull[0].x) - Double(hull[1].x)
            let dy = Double(hull[0].y) - Double(hull[1].y)
            width = 0
            height = Float((dx * dx + dy * dy).squareRoot())
            if dx == 0 {
                swap(&width, &height)
            } else if dy < 0 {
                angle = atan2(dy, dx)
                swap(&width, &height)
            } else if dy > 0 {
                angle = -atan2(dx, dy)
            }
        } else if n == 1 {
            cx = hull[0].x
            cy = hull[0].y
        }

        let angleDeg = Float(angle * 180.0 / Double.pi)
        return RotatedRect(center: PointF(Double(cx), Double(cy)),
                           size: Size(width: Double(width), height: Double(height)),
                           angle: Double(angleDeg))
    }

    /// Rotating-calipers core for the minimum-area rectangle.
    ///
    /// Returns `(corner, edgeVector1, edgeVector2)` exactly as OpenCV's
    /// `rotatingCalipers(..., CALIPERS_MINAREARECT, out)` does. The hull must be
    /// supplied in order; orientation is fixed to `+1` (CCW assumption) to match
    /// `minAreaRect`'s `clockwise == false` call.
    private static func rotatingCalipersMinArea(
        _ pts: [(x: Float, y: Float)]
    ) -> (corner: (x: Float, y: Float), v1: (x: Float, y: Float), v2: (x: Float, y: Float)) {
        let n = pts.count

        var vect = [(x: Float, y: Float)](repeating: (0, 0), count: n)
        var invLen = [Float](repeating: 0, count: n)
        var left = 0, bottom = 0, right = 0, top = 0

        // Edge vectors + extremal vertices. At step i, `pt0` is pts[i].
        var leftX = pts[0].x, rightX = pts[0].x, topY = pts[0].y, bottomY = pts[0].y
        for i in 0..<n {
            let pt0 = pts[i]
            if pt0.x < leftX { leftX = pt0.x; left = i }
            if pt0.x > rightX { rightX = pt0.x; right = i }
            if pt0.y > topY { topY = pt0.y; top = i }
            if pt0.y < bottomY { bottomY = pt0.y; bottom = i }

            let pt = pts[(i + 1) < n ? i + 1 : 0]
            let dx = Double(pt.x) - Double(pt0.x)
            let dy = Double(pt.y) - Double(pt0.y)
            vect[i] = (Float(dx), Float(dy))
            invLen[i] = Float(1.0 / (dx * dx + dy * dy).squareRoot())
        }

        // orientation is +1 for this call → first base vector is (1, 0).
        var baseA: Float = 1
        var baseB: Float = 0

        var seq = [bottom, right, top, left]

        // Best-so-far rectangle parameters (OpenCV's `buffer`).
        var minarea = Float.greatestFiniteMagnitude
        var bufLeftIdx = seq[3]
        var bufA = baseA, bufWidth: Float = 0, bufB = baseB, bufHeight: Float = 0
        var bufBottomIdx = seq[0]

        // True iff vec1 is clockwise-of (to the right of) vec2.
        @inline(__always) func firstVecIsRight(_ a: (x: Float, y: Float), _ b: (x: Float, y: Float)) -> Bool {
            // rotate90CW(a) = (a.y, -a.x); dot with b:
            return a.y * b.x - a.x * b.y < 0
        }

        for _ in 0..<n {
            // Rotate each caliper edge into a common frame, then pick the one
            // making the smallest angle with its supporting polygon edge.
            let r0 = vect[seq[0]]
            let r1 = (x: vect[seq[1]].y, y: -vect[seq[1]].x)          // rotate90CW
            let r2 = (x: -vect[seq[2]].x, y: -vect[seq[2]].y)         // rotate180
            let r3 = (x: -vect[seq[3]].y, y: vect[seq[3]].x)          // rotate90CCW
            let rot = [r0, r1, r2, r3]

            var mainElement = 0
            for i in 1..<4 where firstVecIsRight(rot[i], rot[mainElement]) {
                mainElement = i
            }

            // Advance the base direction to lie along the chosen edge.
            let pindex = seq[mainElement]
            let leadX = vect[pindex].x * invLen[pindex]
            let leadY = vect[pindex].y * invLen[pindex]
            switch mainElement {
            case 0: baseA = leadX;  baseB = leadY
            case 1: baseA = leadY;  baseB = -leadX
            case 2: baseA = -leadX; baseB = -leadY
            default: baseA = -leadY; baseB = leadX
            }

            // Step that caliper onto the next hull vertex.
            seq[mainElement] += 1
            if seq[mainElement] == n { seq[mainElement] = 0 }

            // Evaluate the rectangle spanned by the four current support points.
            var dx = pts[seq[1]].x - pts[seq[3]].x
            var dy = pts[seq[1]].y - pts[seq[3]].y
            let width = dx * baseA + dy * baseB
            dx = pts[seq[2]].x - pts[seq[0]].x
            dy = pts[seq[2]].y - pts[seq[0]].y
            let height = -dx * baseB + dy * baseA
            let area = width * height
            if area <= minarea {
                minarea = area
                bufLeftIdx = seq[3]
                bufA = baseA
                bufWidth = width
                bufB = baseB
                bufHeight = height
                bufBottomIdx = seq[0]
            }
        }

        // Reconstruct corner + edge vectors from the best caliper state.
        let a1 = bufA, b1 = bufB
        let a2 = -bufB, b2 = bufA
        let c1 = a1 * pts[bufLeftIdx].x + pts[bufLeftIdx].y * b1
        let c2 = a2 * pts[bufBottomIdx].x + pts[bufBottomIdx].y * b2
        let idet = 1.0 / (a1 * b2 - a2 * b1)
        let px = (c1 * b2 - c2 * b1) * idet
        let py = (a1 * c2 - a2 * c1) * idet
        return ((px, py), (a1 * bufWidth, b1 * bufWidth), (a2 * bufHeight, b2 * bufHeight))
    }

    // =========================================================================
    // MARK: boxPoints
    // =========================================================================

    /// Four corners of a rotated rectangle (`cv2.boxPoints`).
    ///
    /// Mirrors `cv::RotatedRect::points` (`core/src/types.cpp`). The corner order
    /// is OpenCV's: index 0 bottom-left, 1 top-left, 2 top-right, 3 bottom-right
    /// (for an un-flipped, near-axis-aligned box). Computed in `float`.
    static func boxPoints(_ rect: RotatedRect) -> [PointF] {
        let corners = rotatedRectCorners(cx: Float(rect.center.x), cy: Float(rect.center.y),
                                         w: Float(rect.size.width), h: Float(rect.size.height),
                                         angleDeg: rect.angle)
        return corners.map { PointF(Double($0.x), Double($0.y)) }
    }

    /// Shared `RotatedRect::points` implementation (used by `boxPoints` and the
    /// rotated-rectangle intersection test), kept in `Float` for parity.
    private static func rotatedRectCorners(cx: Float, cy: Float, w: Float, h: Float,
                                           angleDeg: Double) -> [(x: Float, y: Float)] {
        let rad = angleDeg * Double.pi / 180.0      // OpenCV widens angle to double
        let b = Float(cos(rad)) * 0.5
        let a = Float(sin(rad)) * 0.5
        var pt = [(x: Float, y: Float)](repeating: (0, 0), count: 4)
        pt[0] = (cx - a * h - b * w, cy + b * h - a * w)
        pt[1] = (cx + a * h - b * w, cy - b * h - a * w)
        pt[2] = (2 * cx - pt[0].x, 2 * cy - pt[0].y)
        pt[3] = (2 * cx - pt[1].x, 2 * cy - pt[1].y)
        return pt
    }

    // =========================================================================
    // MARK: pointPolygonTest
    // =========================================================================

    /// Point-in-polygon test (`cv2.pointPolygonTest`).
    ///
    /// `measureDist == false`: returns +1 (inside), 0 (on an edge/vertex),
    /// or -1 (outside). `measureDist == true`: returns the signed Euclidean
    /// distance (positive inside, negative outside). Mirrors
    /// `cv::pointPolygonTest` in `imgproc/src/geometry.cpp`.
    static func pointPolygonTest(_ contour: [Point], _ point: PointF, measureDist: Bool) -> Double {
        let total = contour.count
        if total == 0 { return measureDist ? -Double.greatestFiniteMagnitude : -1 }

        // Rounded test point (used by the fast integer branch).
        let ipx = Int(point.x.rounded(.toNearestOrEven))
        let ipy = Int(point.y.rounded(.toNearestOrEven))
        var counter = 0

        // --- fast integer branch: integer contour + integer query, no dist ---
        if !measureDist && Double(ipx) == point.x && Double(ipy) == point.y {
            var v = contour[total - 1]
            for i in 0..<total {
                let v0 = v
                v = contour[i]

                if (v0.y <= ipy && v.y <= ipy) ||
                   (v0.y > ipy && v.y > ipy) ||
                   (v0.x < ipx && v.x < ipx) {
                    // Horizontal edge / vertex exactly under the query → on edge.
                    if ipy == v.y && (ipx == v.x || (ipy == v0.y &&
                        ((v0.x <= ipx && ipx <= v.x) || (v.x <= ipx && ipx <= v0.x)))) {
                        return 0
                    }
                    continue
                }

                var dist = Int64(ipy - v0.y) * Int64(v.x - v0.x)
                         - Int64(ipx - v0.x) * Int64(v.y - v0.y)
                if dist == 0 { return 0 }            // exactly on the edge
                if v.y < v0.y { dist = -dist }
                counter += dist > 0 ? 1 : 0          // crossing-number parity
            }
            return counter % 2 == 0 ? -1 : 1
        }

        // --- general branches (floating-point query point) -------------------
        let px = point.x, py = point.y

        if !measureDist {
            var v = (x: Double(contour[total - 1].x), y: Double(contour[total - 1].y))
            for i in 0..<total {
                let v0 = v
                v = (x: Double(contour[i].x), y: Double(contour[i].y))

                if (v0.y <= py && v.y <= py) ||
                   (v0.y > py && v.y > py) ||
                   (v0.x < px && v.x < px) {
                    if py == v.y && (px == v.x || (py == v0.y &&
                        ((v0.x <= px && px <= v.x) || (v.x <= px && px <= v0.x)))) {
                        return 0
                    }
                    continue
                }

                var dist = (py - v0.y) * (v.x - v0.x) - (px - v0.x) * (v.y - v0.y)
                if dist == 0 { return 0 }
                if v.y < v0.y { dist = -dist }
                counter += dist > 0 ? 1 : 0
            }
            return counter % 2 == 0 ? -1 : 1
        }

        // measureDist == true: track the minimum distance to any edge.
        var minNum = Double(Float.greatestFiniteMagnitude)   // OpenCV seeds with FLT_MAX
        var minDen = 1.0
        var v = (x: Double(contour[total - 1].x), y: Double(contour[total - 1].y))
        for i in 0..<total {
            let v0 = v
            v = (x: Double(contour[i].x), y: Double(contour[i].y))

            let dx = v.x - v0.x, dy = v.y - v0.y
            let dx1 = px - v0.x, dy1 = py - v0.y
            let dx2 = px - v.x, dy2 = py - v.y

            var num: Double
            var den = 1.0
            if dx1 * dx + dy1 * dy <= 0 {
                num = dx1 * dx1 + dy1 * dy1            // closest to start vertex
            } else if dx2 * dx + dy2 * dy >= 0 {
                num = dx2 * dx2 + dy2 * dy2            // closest to end vertex
            } else {
                num = dy1 * dx - dx1 * dy              // perpendicular distance²·den
                num *= num
                den = dx * dx + dy * dy
            }

            if num * minDen < minNum * den {
                minNum = num
                minDen = den
                if minNum == 0 { break }
            }

            // Crossing-number parity for the sign.
            if (v0.y <= py && v.y <= py) ||
               (v0.y > py && v.y > py) ||
               (v0.x < px && v.x < px) {
                continue
            }
            var distNum = dy1 * dx - dx1 * dy
            if dy < 0 { distNum = -distNum }
            counter += distNum > 0 ? 1 : 0
        }

        var result = (minNum / minDen).squareRoot()
        if counter % 2 == 0 { result = -result }
        return result
    }

    // =========================================================================
    // MARK: getAffineTransform
    // =========================================================================

    /// 2x3 affine transform mapping `src[0..2]` onto `dst[0..2]`
    /// (`cv2.getAffineTransform`). Returns the 6 coefficients row-major:
    /// `[m00, m01, m02, m10, m11, m12]` such that
    /// `dst.x = m00*x + m01*y + m02`, `dst.y = m10*x + m11*y + m12`.
    ///
    /// OpenCV solves a 6x6 system; because the x- and y-rows are independent and
    /// share the same coefficient matrix, we solve a single 3x3 system with two
    /// right-hand sides via Cramer's rule (a "hand 3x3 solve", which the spec
    /// explicitly permits). Returns all-zeros if the source points are collinear
    /// (zero determinant), mirroring an unsolvable system.
    static func getAffineTransform(src: [PointF], dst: [PointF]) -> [Double] {
        precondition(src.count >= 3 && dst.count >= 3, "getAffineTransform needs 3 point pairs")

        // Coefficient matrix rows: [sx, sy, 1].
        let x0 = src[0].x, y0 = src[0].y
        let x1 = src[1].x, y1 = src[1].y
        let x2 = src[2].x, y2 = src[2].y

        // det of [[x0,y0,1],[x1,y1,1],[x2,y2,1]] via cofactor expansion.
        let det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1)
        if det == 0 { return [0, 0, 0, 0, 0, 0] }
        let invDet = 1.0 / det

        // Inverse of the 3x3 (adjugate / det). Rows of inv applied to the rhs.
        // inv = (1/det) * adjugate. Compute each entry explicitly.
        let i00 = (y1 - y2) * invDet
        let i01 = (y2 - y0) * invDet
        let i02 = (y0 - y1) * invDet
        let i10 = (x2 - x1) * invDet
        let i11 = (x0 - x2) * invDet
        let i12 = (x1 - x0) * invDet
        let i20 = (x1 * y2 - x2 * y1) * invDet
        let i21 = (x2 * y0 - x0 * y2) * invDet
        let i22 = (x0 * y1 - x1 * y0) * invDet

        // Solve for the x-coefficients (rhs = dst.x) and y-coefficients (dst.y).
        func solve(_ b0: Double, _ b1: Double, _ b2: Double) -> (Double, Double, Double) {
            (i00 * b0 + i01 * b1 + i02 * b2,
             i10 * b0 + i11 * b1 + i12 * b2,
             i20 * b0 + i21 * b1 + i22 * b2)
        }
        let (m00, m01, m02) = solve(dst[0].x, dst[1].x, dst[2].x)
        let (m10, m11, m12) = solve(dst[0].y, dst[1].y, dst[2].y)
        return [m00, m01, m02, m10, m11, m12]
    }

    // =========================================================================
    // MARK: rotatedRectanglesIntersect
    // =========================================================================

    /// True iff two rotated rectangles intersect at all (the OpenCV result is not
    /// `INTERSECT_NONE`). Faithful port of `cv::rotatedRectangleIntersection`
    /// (`imgproc/src/intersection.cpp`); we only need the NONE/!NONE
    /// classification so the actual clipped polygon is discarded.
    static func rotatedRectanglesIntersect(_ a: RotatedRect, _ b: RotatedRect) -> Bool {
        // Empty (degenerate) rectangles never intersect.
        if a.size.width <= 0 || a.size.height <= 0 { return false }
        if b.size.width <= 0 || b.size.height <= 0 { return false }

        // OpenCV shifts both rects so their average center sits at the origin,
        // improving float conditioning. The classification is translation
        // invariant, but we replicate the shift for numerical parity.
        let acx = (Float(a.center.x) + Float(b.center.x)) / 2
        let acy = (Float(a.center.y) + Float(b.center.y)) / 2

        let pts1 = rotatedRectCorners(cx: Float(a.center.x) - acx, cy: Float(a.center.y) - acy,
                                      w: Float(a.size.width), h: Float(a.size.height), angleDeg: a.angle)
        let pts2 = rotatedRectCorners(cx: Float(b.center.x) - acx, cy: Float(b.center.y) - acy,
                                      w: Float(b.size.width), h: Float(b.size.height), angleDeg: b.angle)

        // samePointEps: scaled to the larger rect area, then shrunk to the
        // smallest edge length, finally floored at 1e-16.
        let area1 = Float(a.size.width) * Float(a.size.height)
        let area2 = Float(b.size.width) * Float(b.size.height)
        var samePointEps = Float(1e-6) * max(area1, area2)

        // Special case: identical rectangles → FULL overlap.
        var same = true
        for i in 0..<4 where abs(pts1[i].x - pts2[i].x) > samePointEps || abs(pts1[i].y - pts2[i].y) > samePointEps {
            same = false
            break
        }
        if same { return true }

        // Edge vectors of both rectangles.
        var vec1 = [(x: Float, y: Float)](repeating: (0, 0), count: 4)
        var vec2 = [(x: Float, y: Float)](repeating: (0, 0), count: 4)
        for i in 0..<4 {
            vec1[i] = (pts1[(i + 1) % 4].x - pts1[i].x, pts1[(i + 1) % 4].y - pts1[i].y)
            vec2[i] = (pts2[(i + 1) % 4].x - pts2[i].x, pts2[(i + 1) % 4].y - pts2[i].y)
        }
        for i in 0..<4 {
            samePointEps = min(samePointEps, (vec1[i].x * vec1[i].x + vec1[i].y * vec1[i].y).squareRoot())
            samePointEps = min(samePointEps, (vec2[i].x * vec2[i].x + vec2[i].y * vec2[i].y).squareRoot())
        }
        samePointEps = max(Float(1e-16), samePointEps)

        var found = false

        // (1) Edge-edge intersections: solve each 2x2 line system.
        for i in 0..<4 {
            for j in 0..<4 {
                let x21 = pts2[j].x - pts1[i].x
                let y21 = pts2[j].y - pts1[i].y
                let vx1 = vec1[i].x, vy1 = vec1[i].y
                let vx2 = vec2[j].x, vy2 = vec2[j].y
                let det = vx2 * vy1 - vx1 * vy2
                if abs(det) < 1e-12 { continue }   // (near-)parallel edges
                let invDet = Float(1) / det
                let t1 = (vx2 * y21 - vy2 * x21) * invDet
                let t2 = (vx1 * y21 - vy1 * x21) * invDet
                if t1.isInfinite || t2.isInfinite || t1.isNaN || t2.isNaN { continue }
                if t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1 {
                    found = true   // an actual segment crossing exists
                }
            }
        }

        // Sign of `pt` relative to the directed line (line_vec through line_pt).
        @inline(__always) func isOnPositiveSide(_ lineVec: (x: Float, y: Float),
                                                _ linePt: (x: Float, y: Float),
                                                _ pt: (x: Float, y: Float)) -> Bool {
            lineVec.y * (linePt.x - pt.x) >= lineVec.x * (linePt.y - pt.y)
        }

        // (2) Vertices of rect1 inside rect2 (consistent side of all 4 edges).
        for i in 0..<4 {
            var pos = 0, neg = 0
            for j in 0..<4 {
                if isOnPositiveSide(vec2[j], pts2[j], pts1[i]) { pos += 1 } else { neg += 1 }
            }
            if pos == 4 || neg == 4 { found = true }
        }
        // (3) Vertices of rect2 inside rect1.
        for i in 0..<4 {
            var pos = 0, neg = 0
            for j in 0..<4 {
                if isOnPositiveSide(vec1[j], pts1[j], pts2[i]) { pos += 1 } else { neg += 1 }
            }
            if pos == 4 || neg == 4 { found = true }
        }

        // OpenCV returns NONE iff no intersection point was collected at all.
        return found
    }

    // =========================================================================
    // MARK: fillConvexPoly
    // =========================================================================

    /// Scanline fill of a convex polygon (`cv2.fillConvexPoly`, default
    /// `lineType == LINE_8`, `shift == 0`). Faithful port of OpenCV's
    /// `FillConvexPoly` (+ the `Line` LINE_8 boundary draw it performs first) in
    /// `imgproc/src/drawing.cpp`. Every covered pixel is set to `value`.
    static func fillConvexPoly(_ image: inout GrayscaleImage, _ points: [Point], value: UInt8) {
        let npts = points.count
        if npts == 0 { return }

        let width = image.width
        let height = image.height
        var px = image.pixels

        // Fixed-point constants from OpenCV (sub-pixel math at 1/65536 units).
        let XY_SHIFT = 16
        let XY_ONE: Int64 = 1 << 16
        let delta1: Int64 = XY_ONE >> 1   // rounding bias for the left edge (+0.5)
        let delta2: Int64 = XY_ONE >> 1   // rounding bias for the right edge

        // --- 1. draw the polygon boundary with the LINE_8 rasteriser --------
        // OpenCV draws every edge first; the spans below then fill the interior.
        var p0 = points[npts - 1]
        for i in 0..<npts {
            let p = points[i]
            drawLine8(&px, width, height, p0.x, p0.y, p.x, p.y, value)
            p0 = p
        }

        // --- 2. bounding box + top vertex -----------------------------------
        var xmin = points[0].x, xmax = points[0].x
        var ymin = points[0].y, ymax = points[0].y
        var imin = 0
        for i in 0..<npts {
            let p = points[i]
            if p.y < ymin { ymin = p.y; imin = i }
            if p.y > ymax { ymax = p.y }
            if p.x > xmax { xmax = p.x }
            if p.x < xmin { xmin = p.x }
        }

        // Early-out matching OpenCV (boundary lines are already drawn above).
        if npts < 3 || xmax < 0 || ymax < 0 || xmin >= width || ymin >= height {
            image.pixels = px
            return
        }
        ymax = min(ymax, height - 1)

        // --- 3. two active edges sweeping down from the top vertex -----------
        // edge[0] walks vertices forward (di = +1), edge[1] backward (di = -1).
        struct Edge { var idx: Int; var di: Int; var x: Int64; var dx: Int64; var ye: Int }
        var edge = [Edge(idx: imin, di: 1, x: -XY_ONE, dx: 0, ye: ymin),
                    Edge(idx: imin, di: npts - 1, x: -XY_ONE, dx: 0, ye: ymin)]

        var y = ymin
        var edges = npts        // global budget shared by both edge walks
        let yL0 = Double(0)
        _ = yL0

        repeat {
            // Refresh whichever edge(s) the sweep line has passed.
            for i in 0..<2 {
                if y >= edge[i].ye {
                    var idx0 = edge[i].idx
                    let di = edge[i].di
                    var idx = idx0 + di
                    if idx >= npts { idx -= npts }

                    // Walk vertices until one lies strictly below the sweep line.
                    while true {
                        let cont = edges > 0
                        edges -= 1
                        if !cont { break }

                        let ty = points[idx].y          // (delta == 0, shift == 0)
                        if ty > y {
                            // Set up linear x-interpolation along this edge.
                            let xs = Int64(points[idx0].x) << (XY_SHIFT)
                            let xe = Int64(points[idx].x) << (XY_SHIFT)
                            let span = Int64(ty - y)
                            edge[i].ye = ty
                            edge[i].dx = ((xe - xs) * 2 + span) / (2 * span)
                            edge[i].x = xs
                            edge[i].idx = idx
                            break
                        }
                        idx0 = idx
                        idx += di
                        if idx >= npts { idx -= npts }
                    }
                }
            }

            if edges < 0 { break }   // both edge chains exhausted

            // Fill the horizontal span between the two edges on this row.
            if y >= 0 {
                var left = 0, right = 1
                if edge[0].x > edge[1].x { left = 1; right = 0 }

                var xx1 = Int((edge[left].x + delta1) >> Int64(XY_SHIFT))
                var xx2 = Int((edge[right].x + delta2) >> Int64(XY_SHIFT))

                if xx2 >= 0 && xx1 < width {
                    if xx1 < 0 { xx1 = 0 }
                    if xx2 >= width { xx2 = width - 1 }
                    if xx1 <= xx2 {
                        let rowBase = y * width
                        for x in xx1...xx2 { px[rowBase + x] = value }
                    }
                }
            }

            edge[0].x += edge[0].dx
            edge[1].x += edge[1].dx
            y += 1
        } while y <= ymax

        image.pixels = px
    }

    // MARK: - Internal helpers

    /// LINE_8 Bresenham line rasteriser, a port of OpenCV's `LineIterator`
    /// (connectivity 8, `leftToRight == true`) + `Line`. Clips to the image and
    /// writes `value` to every covered pixel.
    private static func drawLine8(_ px: inout [UInt8], _ width: Int, _ height: Int,
                                  _ x1in: Int, _ y1in: Int, _ x2in: Int, _ y2in: Int,
                                  _ value: UInt8) {
        var x1 = x1in, y1 = y1in, x2 = x2in, y2 = y2in

        // Clip the segment to the image rectangle (Cohen–Sutherland, OpenCV's
        // `clipLine`). If fully outside, nothing is drawn.
        if !clipLine(width, height, &x1, &y1, &x2, &y2) { return }

        var dx = x2 - x1
        var dy = y2 - y1
        var deltaX = 1
        var deltaY = 1

        // Normalise direction so we always iterate left-to-right.
        if dx < 0 {
            dx = -dx; dy = -dy
            x1 = x2; y1 = y2
        }
        if dy < 0 { dy = -dy; deltaY = -1 }

        let vert = dy > dx
        if vert { swap(&dx, &dy); swap(&deltaX, &deltaY) }

        // Bresenham state (connectivity 8).
        let err0 = dx - (dy + dy)
        let plusDelta = dx + dx
        let minusDelta = -(dy + dy)
        var minusShift = deltaX
        var plusShift = 0
        var minusStep = 0
        var plusStep = deltaY
        let count = dx + 1
        if vert { swap(&plusStep, &plusShift); swap(&minusStep, &minusShift) }

        var err = err0
        var p = (x: x1, y: y1)
        for _ in 0..<count {
            if p.x >= 0 && p.x < width && p.y >= 0 && p.y < height {
                px[p.y * width + p.x] = value
            }
            // `mask` is all-ones when err < 0, selecting the diagonal step.
            let mask = err < 0 ? -1 : 0
            err += minusDelta + (plusDelta & mask)
            p.x += minusShift + (plusShift & mask)
            p.y += minusStep + (plusStep & mask)
        }
    }

    /// Cohen–Sutherland line clipping to `[0, width) x [0, height)`, a port of
    /// OpenCV's integer `clipLine`. Returns false if the segment is fully
    /// outside; otherwise updates the endpoints in place.
    private static func clipLine(_ width: Int, _ height: Int,
                                 _ x1: inout Int, _ y1: inout Int,
                                 _ x2: inout Int, _ y2: inout Int) -> Bool {
        if width <= 0 || height <= 0 { return false }
        let right = width - 1
        let bottom = height - 1

        // Region codes: bit0 left, bit1 right, bit2 top, bit3 bottom.
        var c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0) + (y1 < 0 ? 4 : 0) + (y1 > bottom ? 8 : 0)
        var c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0) + (y2 < 0 ? 4 : 0) + (y2 > bottom ? 8 : 0)

        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            // Clip against the horizontal edges first.
            if c1 & 12 != 0 {
                let a = c1 < 8 ? 0 : bottom
                x1 += Int(Double(a - y1) * Double(x2 - x1) / Double(y2 - y1))
                y1 = a
                c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0)
            }
            if c2 & 12 != 0 {
                let a = c2 < 8 ? 0 : bottom
                x2 += Int(Double(a - y2) * Double(x2 - x1) / Double(y2 - y1))
                y2 = a
                c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0)
            }
            // Then against the vertical edges.
            if (c1 & c2) == 0 && (c1 | c2) != 0 {
                if c1 != 0 {
                    let a = c1 == 1 ? 0 : right
                    y1 += Int(Double(a - x1) * Double(y2 - y1) / Double(x2 - x1))
                    x1 = a
                    c1 = 0
                }
                if c2 != 0 {
                    let a = c2 == 1 ? 0 : right
                    y2 += Int(Double(a - x2) * Double(y2 - y1) / Double(x2 - x1))
                    x2 = a
                    c2 = 0
                }
            }
        }
        return (c1 | c2) == 0
    }

    /// Three-way sign of an `Int` (`CV_SIGN`): -1, 0, or +1.
    @inline(__always) private static func sign(_ v: Int) -> Int {
        (v > 0 ? 1 : 0) - (v < 0 ? 1 : 0)
    }

    /// Three-way sign of an `Int64`.
    @inline(__always) private static func sign64(_ v: Int64) -> Int {
        (v > 0 ? 1 : 0) - (v < 0 ? 1 : 0)
    }
}
