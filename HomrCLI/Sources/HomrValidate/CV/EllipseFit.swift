import Foundation

/// Native port of OpenCV's `fitEllipse` and `HoughLinesP`.
///
/// `fitEllipse` is a faithful transcription of OpenCV 4.x `fitEllipseNoDirect`
/// (modules/imgproc/src/shapedescr.cpp) — the path `cv2.fitEllipse` takes for
/// contours with more than 5 points, which is exactly how homr uses it
/// (modelling noteheads as ellipses in `bounding_boxes.py`). The least-squares
/// systems OpenCV solves via SVD are solved here with normal equations +
/// Gaussian elimination; for these well-scaled, well-conditioned systems the
/// result matches OpenCV to well within the fixture tolerance, and it avoids
/// depending on Accelerate's (shifting) LAPACK interface.
extension CV {

    // MARK: fitEllipse

    /// Fits an ellipse to a set of points (OpenCV `fitEllipseNoDirect`).
    /// Requires at least 5 points; fewer returns a degenerate rect at the centroid.
    static func fitEllipse(_ pointsInt: [Point]) -> RotatedRect {
        let n = pointsInt.count

        // Centroid of the input points (used to center the data for stability).
        var cx = 0.0, cy = 0.0
        var pts = pointsInt.map { (x: Double($0.x), y: Double($0.y)) }
        for p in pts { cx += p.x; cy += p.y }
        let center = (x: cx / Double(max(1, n)), y: cy / Double(max(1, n)))

        guard n >= 5 else {
            // OpenCV raises here; we degrade gracefully instead.
            return RotatedRect(center: PointF(center.x, center.y),
                               size: Size(width: 0, height: 0), angle: 0)
        }

        let fltEps = Double(Float.ulpOfOne)       // FLT_EPSILON
        let minEps = 1e-8

        // Isotropic scale so coordinates sit around ±100 (matches OpenCV).
        var s = 0.0
        for p in pts { s += abs(p.x - center.x) + abs(p.y - center.y) }
        let scale = 100.0 / (s > fltEps ? s : fltEps)

        // --- First fit: solve for general conic params A..E (5 unknowns). ---
        // Row layout matches OpenCV: [-px², -py², -px·py, px, py] · g = 10000.
        func buildConicRows() -> [[Double]] {
            pts.map { p -> [Double] in
                let px = (p.x - center.x) * scale
                let py = (p.y - center.y) * scale
                return [-px * px, -py * py, -px * py, px, py]
            }
        }

        var gfp = [Double](repeating: 0, count: 5)
        if let sol = solveLeastSquares(rows: buildConicRows(),
                                       rhs: [Double](repeating: 10000.0, count: n)) {
            gfp = sol.solution
            // OpenCV perturbs the points and refits when the system is near
            // singular (ratio of largest to smallest pivot too large). This is
            // rare for real contours; we use a deterministic offset instead of
            // OpenCV's RNG (so the rare path stays reproducible).
            if sol.nearSingular {
                let eps = s / (Double(n) * 2) * 1e-3
                for i in 0..<n {
                    pts[i].x += deterministicOffset(eps, i, 0)
                    pts[i].y += deterministicOffset(eps, i, 1)
                }
                if let retry = solveLeastSquares(rows: buildConicRows(),
                                                 rhs: [Double](repeating: 10000.0, count: n)) {
                    gfp = retry.solution
                }
            }
        }

        // --- Solve the 2×2 system for the ellipse center offset (rp[0], rp[1]). ---
        // [2A  B ] [cx]   [D]
        // [ B 2C ] [cy] = [E]
        var rp = [Double](repeating: 0, count: 5)
        if let cs = solveLeastSquares(
            rows: [[2 * gfp[0], gfp[2]], [gfp[2], 2 * gfp[1]]],
            rhs: [gfp[3], gfp[4]]
        ) {
            rp[0] = cs.solution[0]
            rp[1] = cs.solution[1]
        }

        // --- Re-fit A,B,C about the recovered center (3 unknowns, rhs = 1). ---
        let refitRows: [[Double]] = pts.map { p -> [Double] in
            let px = (p.x - center.x) * scale
            let py = (p.y - center.y) * scale
            return [(px - rp[0]) * (px - rp[0]),
                    (py - rp[1]) * (py - rp[1]),
                    (px - rp[0]) * (py - rp[1])]
        }
        if let rf = solveLeastSquares(rows: refitRows, rhs: [Double](repeating: 1.0, count: n)) {
            gfp[0] = rf.solution[0]
            gfp[1] = rf.solution[1]
            gfp[2] = rf.solution[2]
        }

        // --- Recover angle and radii (verbatim from OpenCV). ---
        rp[4] = -0.5 * atan2(gfp[2], gfp[1] - gfp[0])
        let t: Double
        if abs(gfp[2]) > minEps {
            t = gfp[2] / sin(-2.0 * rp[4])
        } else {
            t = gfp[1] - gfp[0]
        }
        rp[2] = abs(gfp[0] + gfp[1] - t)
        if rp[2] > minEps { rp[2] = (2.0 / rp[2]).squareRoot() }
        rp[3] = abs(gfp[0] + gfp[1] + t)
        if rp[3] > minEps { rp[3] = (2.0 / rp[3]).squareRoot() }

        let boxCenterX = rp[0] / scale + center.x
        let boxCenterY = rp[1] / scale + center.y
        var width = rp[2] * 2 / scale
        var height = rp[3] * 2 / scale
        var angle = 0.0

        // NOTE: OpenCV only assigns the angle when it swaps width/height. When
        // no swap happens the angle is left at 0 — this is correct because the
        // recovered orientation already aligns the major axis with `height`.
        if width > height {
            swap(&width, &height)
            angle = 90 + rp[4] * 180 / Double.pi
        }
        if angle < -180 { angle += 360 }
        if angle > 360 { angle -= 360 }

        return RotatedRect(center: PointF(boxCenterX, boxCenterY),
                           size: Size(width: width, height: height),
                           angle: angle)
    }

    // MARK: HoughLinesP

    /// Progressive Probabilistic Hough Transform (OpenCV `HoughLinesP`).
    ///
    /// Returns detected line segments as endpoint pairs. NOTE: OpenCV's
    /// implementation visits edge pixels in a *random* order, so exact
    /// bit-for-bit parity is impossible. This is a faithful implementation of
    /// the same algorithm with a fixed RNG seed for reproducibility. homr only
    /// uses this once (splitting wide staff fragments), where the general
    /// behaviour — not exact parity — is what matters.
    static func houghLinesP(
        _ image: GrayscaleImage,
        rho: Double,
        theta: Double,
        threshold: Int,
        minLineLength: Double,
        maxLineGap: Double
    ) -> [(Point, Point)] {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0, rho > 0, theta > 0 else { return [] }

        let numAngle = Int((Double.pi / theta).rounded())
        let numRho = Int((((Double(width) + Double(height)) * 2 + 1) / rho).rounded())
        guard numAngle > 0, numRho > 0 else { return [] }

        let irho = 1.0 / rho
        var tabSin = [Double](repeating: 0, count: numAngle)
        var tabCos = [Double](repeating: 0, count: numAngle)
        for n in 0..<numAngle {
            let ang = Double(n) * theta
            tabSin[n] = sin(ang) * irho
            tabCos[n] = cos(ang) * irho
        }

        // Collect all foreground pixels.
        var mask = [Bool](repeating: false, count: width * height)
        var points: [(x: Int, y: Int)] = []
        for y in 0..<height {
            let row = y * width
            for x in 0..<width where image.pixels[row + x] > 0 {
                mask[row + x] = true
                points.append((x, y))
            }
        }

        var accum = [Int](repeating: 0, count: numAngle * numRho)
        var rng = SeededRNG(seed: 0x2d2816fe)   // fixed seed → reproducible
        var lines: [(Point, Point)] = []
        let rhoOffset = (numRho - 1) / 2

        var count = points.count
        while count > 0 {
            // Pick a random unprocessed point and remove it from the pool.
            let idx = Int(rng.next() % UInt64(count))
            let pt = points[idx]
            points[idx] = points[count - 1]
            count -= 1

            if !mask[pt.y * width + pt.x] { continue }

            // Vote for every line through this point; track the strongest bin.
            var maxVal = threshold - 1
            var maxAngle = 0
            for n in 0..<numAngle {
                var r = Int((Double(pt.x) * tabCos[n] + Double(pt.y) * tabSin[n]).rounded())
                r += rhoOffset
                if r < 0 || r >= numRho { continue }
                let bin = n * numRho + r
                accum[bin] += 1
                if accum[bin] > maxVal {
                    maxVal = accum[bin]
                    maxAngle = n
                }
            }
            if maxVal < threshold { continue }

            // Walk the line in both directions to find the connected segment.
            let a = -tabSin[maxAngle]
            let b = tabCos[maxAngle]
            let (endA, endB, gapPoints) = traceSegment(
                from: pt, a: a, b: b, mask: mask, width: width, height: height,
                maxLineGap: maxLineGap)

            let dx = Double(endA.x - endB.x)
            let dy = Double(endA.y - endB.y)
            let lineLength = (dx * dx + dy * dy).squareRoot()

            // Remove the segment's pixels and un-vote them from the accumulator.
            for gp in gapPoints where mask[gp.y * width + gp.x] {
                mask[gp.y * width + gp.x] = false
                for n in 0..<numAngle {
                    var r = Int((Double(gp.x) * tabCos[n] + Double(gp.y) * tabSin[n]).rounded())
                    r += rhoOffset
                    if r < 0 || r >= numRho { continue }
                    accum[n * numRho + r] -= 1
                }
            }

            if lineLength >= minLineLength {
                lines.append((Point(endA.x, endA.y), Point(endB.x, endB.y)))
            }
        }
        return lines
    }

    // MARK: - Private helpers

    /// Solves `min ||A x - b||` via normal equations (AᵀA x = Aᵀb) with
    /// Gaussian elimination. Returns the solution plus a near-singular flag
    /// derived from the pivot magnitude ratio.
    private static func solveLeastSquares(
        rows: [[Double]], rhs: [Double]
    ) -> (solution: [Double], nearSingular: Bool)? {
        guard let firstRow = rows.first else { return nil }
        let cols = firstRow.count
        let m = rows.count
        guard m >= cols, rhs.count == m else { return nil }

        // Build the symmetric normal matrix AᵀA (cols × cols) and Aᵀb.
        var ata = [Double](repeating: 0, count: cols * cols)
        var atb = [Double](repeating: 0, count: cols)
        for i in 0..<m {
            let row = rows[i]
            let bi = rhs[i]
            for j in 0..<cols {
                atb[j] += row[j] * bi
                for k in j..<cols {
                    ata[j * cols + k] += row[j] * row[k]
                }
            }
        }
        // Mirror the upper triangle into the lower triangle.
        for j in 0..<cols {
            for k in 0..<j {
                ata[j * cols + k] = ata[k * cols + j]
            }
        }

        // Gaussian elimination with partial pivoting on [AtA | Atb].
        var maxPivot = 0.0
        var minPivot = Double.greatestFiniteMagnitude
        var a = ata
        var b = atb
        for col in 0..<cols {
            // Find the pivot row.
            var pivotRow = col
            var pivotVal = abs(a[col * cols + col])
            for r in (col + 1)..<cols {
                let v = abs(a[r * cols + col])
                if v > pivotVal { pivotVal = v; pivotRow = r }
            }
            if pivotVal < 1e-300 { return ([Double](repeating: 0, count: cols), true) }
            if pivotRow != col {
                for c in 0..<cols { a.swapAt(pivotRow * cols + c, col * cols + c) }
                b.swapAt(pivotRow, col)
            }
            let pivot = a[col * cols + col]
            maxPivot = Swift.max(maxPivot, abs(pivot))
            minPivot = Swift.min(minPivot, abs(pivot))
            for r in 0..<cols where r != col {
                let factor = a[r * cols + col] / pivot
                if factor == 0 { continue }
                for c in col..<cols {
                    a[r * cols + c] -= factor * a[col * cols + c]
                }
                b[r] -= factor * b[col]
            }
        }

        var x = [Double](repeating: 0, count: cols)
        for i in 0..<cols { x[i] = b[i] / a[i * cols + i] }

        let nearSingular = minPivot <= maxPivot * Double(Float.ulpOfOne)
        return (x, nearSingular)
    }

    /// Deterministic replacement for OpenCV's RNG-based point jitter.
    private static func deterministicOffset(_ eps: Double, _ index: Int, _ axis: Int) -> Double {
        // Hash the index/axis into a value in [-eps, eps].
        let h = (UInt64(bitPattern: Int64(index &* 73856093 ^ axis &* 19349663)) &* 0x9E3779B97F4A7C15)
        let unit = Double(h >> 11) * (1.0 / 9007199254740992.0)   // [0,1)
        return (unit * 2 - 1) * eps
    }

    /// Traces the connected run of foreground pixels through `start` along the
    /// line direction `(a, b)`, tolerating gaps up to `maxLineGap`. Returns the
    /// two endpoints and every pixel visited along the run.
    private static func traceSegment(
        from start: (x: Int, y: Int),
        a: Double, b: Double,
        mask: [Bool], width: Int, height: Int,
        maxLineGap: Double
    ) -> (Point, Point, [(x: Int, y: Int)]) {
        // Step one pixel at a time along the dominant axis of the line
        // direction (a, b), letting the other axis advance by the slope.
        let stepX: Double
        let stepY: Double
        if abs(a) > abs(b) {
            stepX = a > 0 ? 1 : -1
            stepY = (b / abs(a)) * (a > 0 ? 1 : -1)
        } else {
            stepY = b > 0 ? 1 : -1
            stepX = (a / abs(b)) * (b > 0 ? 1 : -1)
        }

        // Walk in one direction from the seed, collecting foreground pixels and
        // stopping once an empty run exceeds `maxLineGap`.
        func walk(stepX: Double, stepY: Double) -> ((x: Int, y: Int), [(x: Int, y: Int)]) {
            var fx = Double(start.x)
            var fy = Double(start.y)
            var last = start
            var visited: [(x: Int, y: Int)] = []
            var gap = 0.0
            while true {
                let ix = Int(fx.rounded())
                let iy = Int(fy.rounded())
                if ix < 0 || ix >= width || iy < 0 || iy >= height { break }
                if mask[iy * width + ix] {
                    visited.append((ix, iy))
                    last = (ix, iy)
                    gap = 0
                } else {
                    gap += 1
                    if gap > maxLineGap { break }
                }
                fx += stepX
                fy += stepY
            }
            return (last, visited)
        }

        let (endForward, visitedF) = walk(stepX: stepX, stepY: stepY)
        let (endBackward, visitedB) = walk(stepX: -stepX, stepY: -stepY)

        var all = visitedF
        all.append(contentsOf: visitedB)
        return (Point(endForward.x, endForward.y), Point(endBackward.x, endBackward.y), all)
    }
}

/// Small deterministic PRNG (SplitMix64) for the Hough point ordering.
private struct SeededRNG {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
