import Foundation

// MARK: - findContours (Suzuki–Abe 1985 border following)
//
// This file is a native Swift re-implementation of OpenCV's `cv2.findContours`
// with `CHAIN_APPROX_SIMPLE`. It contains NO OpenCV dependency.
//
// The algorithm implemented is the classic *border following* procedure from:
//
//     S. Suzuki and K. Abe, "Topological Structural Analysis of Digitized
//     Binary Images by Border Following", CVGIP 30 (1985), 32–46.
//
// This is exactly the algorithm OpenCV uses internally (`icvFetchContour` plus
// the NBD/LNBD raster-scan bookkeeping). We faithfully reproduce:
//
//   * the 8-neighbour clockwise / counter-clockwise search order,
//   * the NBD (border counter) and LNBD (last-met border) bookkeeping that
//     drives the parent/child hierarchy (Table 1 of the paper),
//   * the pixel marking convention (positive = "left" border pixel, negative =
//     "right" border pixel that has been examined),
//   * OpenCV's point-writing / direction-change logic that yields the
//     `CHAIN_APPROX_SIMPLE` polygon (only direction-change vertices are kept).
//
// Because the traversal is reproduced step-for-step, the emitted point
// sequences and contour ordering match OpenCV's output. The hierarchy is
// derived from the canonical Suzuki–Abe NBD table rather than OpenCV's
// rectangle-containment optimisation; the two agree for well-formed binary
// images.
//
// ---------------------------------------------------------------------------
// Coordinate / winding convention (documented precisely so callers know what
// they get):
//
//   * Coordinates are in the ORIGINAL image space (origin top-left, x → right,
//     y → down). We internally work on a 1-pixel zero-padded copy and subtract
//     the padding offset (1,1) from every emitted point.
//
//   * Border following starts at the first foreground pixel of a border met by
//     the raster scan (top-to-bottom, left-to-right). For a small solid square
//     with corners A=(x,y), B=(x,y+1), C=(x+1,y+1), D=(x+1,y), the outer
//     contour is emitted as A → B → C → D, i.e. down the left edge, along the
//     bottom, up the right edge, across the top. Using the shoelace formula in
//     image coordinates (y pointing down) this has NEGATIVE signed area, i.e.
//     it is counter-clockwise *as drawn on screen*.
//
//   * Hole borders wind the opposite way (positive shoelace area in image
//     coordinates). This is identical to OpenCV's behaviour.
//
//   * `contours` are ordered by the raster-scan order in which each border's
//     starting pixel is first encountered (matching OpenCV).
// ---------------------------------------------------------------------------

extension CV {

    /// Find contours of a binary image, mirroring `cv2.findContours(img, mode,
    /// cv2.CHAIN_APPROX_SIMPLE)`.
    ///
    /// - Parameters:
    ///   - image: Grayscale buffer; any pixel `> 0` is treated as foreground
    ///     (exactly like OpenCV which treats every non-zero pixel as set).
    ///   - mode: `.tree` reproduces `RETR_TREE` (full parent/child hierarchy);
    ///     `.externalOnly` reproduces `RETR_EXTERNAL` (only the outermost
    ///     contours, chained as siblings with no parents or children).
    /// - Returns: Parallel arrays of contours and hierarchy nodes. Hierarchy
    ///   indices reference `contours`; `-1` means "none".
    static func findContours(_ image: GrayscaleImage, mode: RetrievalMode) -> ContourResult {
        let width = image.width
        let height = image.height

        // Degenerate image: nothing to trace.
        guard width > 0, height > 0 else {
            return ContourResult(contours: [], hierarchy: [])
        }

        // ---------------------------------------------------------------
        // 1. Build the padded "border image" F (Int-valued).
        //
        // We surround the image with a 1-pixel ring of zeros. This is the
        // standard trick that lets the 8-neighbour search never go out of
        // bounds and guarantees the image frame is background. F holds the
        // Suzuki–Abe labels as the algorithm runs:
        //   0      → background
        //   1      → unvisited foreground
        //   +NBD   → foreground pixel marked as a "left" border pixel
        //   -NBD   → foreground pixel marked as a "right" border pixel
        //            (its right neighbour was examined and found to be 0)
        // All non-zero values count as foreground for the neighbour search.
        // ---------------------------------------------------------------
        let pw = width + 2          // padded width
        let ph = height + 2         // padded height
        var f = [Int](repeating: 0, count: pw * ph)
        for y in 0..<height {
            let srcRow = y * width
            let dstRow = (y + 1) * pw + 1   // +1 row, +1 col padding offset
            for x in 0..<width {
                // Foreground iff the source pixel is strictly positive.
                f[dstRow + x] = image.pixels[srcRow + x] > 0 ? 1 : 0
            }
        }

        // ---------------------------------------------------------------
        // 2. Neighbour direction tables.
        //
        // Directions are numbered 0...7 in the SAME order OpenCV uses
        // (`CV_INIT_3X3_DELTAS` / `icvCodeDeltas`):
        //
        //     index:  0   1   2   3   4   5   6   7
        //     dir:    E   NE  N   NW  W   SW  S   SE
        //
        // In image coordinates (y down), INCREASING the index rotates
        // counter-clockwise and DECREASING it rotates clockwise. The two
        // tables must stay in lock-step:
        //   * `delta`     : linear index offset into `f` for each direction.
        //   * `codeDelta` : the matching (dx, dy) used to advance the running
        //                   output point.
        // ---------------------------------------------------------------
        let delta: [Int] = [
            1,          // 0  E   ( 1,  0)
            -pw + 1,    // 1  NE  ( 1, -1)
            -pw,        // 2  N   ( 0, -1)
            -pw - 1,    // 3  NW  (-1, -1)
            -1,         // 4  W   (-1,  0)
            pw - 1,     // 5  SW  (-1,  1)
            pw,         // 6  S   ( 0,  1)
            pw + 1      // 7  SE  ( 1,  1)
        ]
        let codeDelta: [(dx: Int, dy: Int)] = [
            (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)
        ]

        // ---------------------------------------------------------------
        // 3. Hierarchy bookkeeping (the heart of Suzuki–Abe).
        //
        // Every border is given a sequential number NBD starting at 2. NBD 1
        // is the "frame" (the outer image border) — a virtual hole border that
        // is the implicit parent of all top-level outer contours but is itself
        // never returned as a contour.
        //
        // For each border we remember:
        //   * borderIsHole[k]  : true if border k is a hole border.
        //   * borderParent[k]  : the border number of border k's parent
        //                        (0 / 1 ultimately map to "no contour parent").
        //
        // Index 0 is an unused dummy so the arrays can be addressed directly by
        // border number. Index 1 is the frame: a hole with no parent.
        // ---------------------------------------------------------------
        var borderIsHole: [Bool] = [false, true]   // [dummy, frame=hole]
        var borderParent: [Int] = [0, 0]            // [dummy, frame parent=none]
        var nbd = 1                                 // current border counter

        // Contours collected in discovery (raster-scan) order. Contour at index
        // `c` always corresponds to border number `c + 2`.
        var contours: [[CV.Point]] = []

        // ---------------------------------------------------------------
        // 4. Raster scan with LNBD (last-met border) tracking.
        //
        // LNBD = the sequential number of the border most recently encountered
        // on the current scan line before the current pixel. It is reset to the
        // frame (1) at the start of every row.
        // ---------------------------------------------------------------
        for i in 1...height {        // padded interior rows
            var lnbd = 1             // reset to the frame at the start of a row
            for j in 1...width {     // padded interior columns
                let idx = i * pw + j
                let fval = f[idx]

                // Step (1)(c): background pixels are skipped entirely.
                if fval == 0 { continue }

                var isBorderStart = false
                var isHole = false

                // Step (1)(a): outer-border starting point.
                // f(i,j) == 1 (unvisited foreground) and the pixel to the LEFT
                // is background. The examined background neighbour (i2,j2) is to
                // the West, so border following will start searching there.
                if fval == 1 && f[idx - 1] == 0 {
                    isBorderStart = true
                    isHole = false
                }
                // Step (1)(b): hole-border starting point.
                // f(i,j) >= 1 (any foreground, possibly already on a border) and
                // the pixel to the RIGHT is background. The examined background
                // neighbour is to the East. Per the paper, if the current pixel
                // already carries a border label (> 1) we update LNBD to it
                // BEFORE deciding the parent below.
                else if fval >= 1 && f[idx + 1] == 0 {
                    isBorderStart = true
                    isHole = true
                    if fval > 1 { lnbd = fval }
                }

                if isBorderStart {
                    nbd += 1

                    // -------------------------------------------------------
                    // Step (2): decide the parent of the new border B using the
                    // border B' whose number is LNBD, per Table 1 of the paper:
                    //
                    //   B \ B'    outer            hole
                    //   outer     parent(B')       B'
                    //   hole      B'               parent(B')
                    //
                    // i.e. if B and B' have the SAME type the new border shares
                    // B''s parent; if they DIFFER, B''s itself is the parent.
                    // -------------------------------------------------------
                    let bPrimeIsHole = borderIsHole[lnbd]
                    let parentNum: Int = (isHole == bPrimeIsHole)
                        ? borderParent[lnbd]
                        : lnbd
                    borderIsHole.append(isHole)
                    borderParent.append(parentNum)

                    // -------------------------------------------------------
                    // Step (3): follow the border, marking pixels in `f` and
                    // collecting the CHAIN_APPROX_SIMPLE point list.
                    // -------------------------------------------------------
                    let pts = followBorder(
                        &f,
                        start: idx,
                        startX: j,
                        startY: i,
                        isHole: isHole,
                        nbd: nbd,
                        delta: delta,
                        codeDelta: codeDelta
                    )
                    contours.append(pts)
                }

                // Step (4): update LNBD for the pixel we are leaving. After
                // border following the start pixel carries ±NBD, so |f| recovers
                // the border number. A bare "1" means an interior foreground
                // pixel that is on no border, which leaves LNBD unchanged.
                let after = f[idx]
                if after != 1 { lnbd = abs(after) }
            }
        }

        // ---------------------------------------------------------------
        // 5. Translate the border-number bookkeeping into contour-index
        //    parents, then build the requested hierarchy representation.
        // ---------------------------------------------------------------
        let count = contours.count
        var parent = [Int](repeating: -1, count: count)
        for c in 0..<count {
            // Parent border number; the frame (<= 1) maps to "no parent" (-1).
            let pn = borderParent[c + 2]
            parent[c] = (pn <= 1) ? -1 : (pn - 2)
        }

        // OpenCV does not return contours in raster-discovery order: each new
        // contour is prepended to its parent's child list, and the result is a
        // pre-order tree walk. The net effect is that siblings (and the
        // top-level roots) come out in REVERSE discovery order, with every
        // parent emitted before its children. Reorder to match exactly.
        let (orderedContours, orderedParent) =
            reorderLikeOpenCV(contours: contours, parent: parent, count: count)

        switch mode {
        case .tree:
            return ContourResult(
                contours: orderedContours,
                hierarchy: buildTreeHierarchy(parent: orderedParent, count: count)
            )
        case .externalOnly:
            return buildExternalResult(contours: orderedContours, parent: orderedParent, count: count)
        }
    }

    /// Reorders raster-discovered contours into OpenCV's output order: a
    /// pre-order DFS where each node's children — and the root set — are visited
    /// in reverse discovery order. Returns the reordered contours together with
    /// a parent array remapped to the new indices.
    private static func reorderLikeOpenCV(
        contours: [[CV.Point]],
        parent: [Int],
        count: Int
    ) -> ([[CV.Point]], [Int]) {
        guard count > 1 else { return (contours, parent) }

        // Children of each contour (and of the virtual root, key -1), in
        // discovery order.
        var children: [Int: [Int]] = [:]
        for c in 0..<count { children[parent[c], default: []].append(c) }

        // Pre-order walk, visiting children in reverse discovery order.
        var order: [Int] = []
        order.reserveCapacity(count)
        func emit(_ node: Int) {
            order.append(node)
            if let kids = children[node] {
                for child in kids.reversed() { emit(child) }
            }
        }
        if let roots = children[-1] {
            for root in roots.reversed() { emit(root) }
        }

        // Build old→new index map and remap contours + parents.
        var oldToNew = [Int](repeating: -1, count: count)
        for (newIdx, old) in order.enumerated() { oldToNew[old] = newIdx }

        let newContours = order.map { contours[$0] }
        var newParent = [Int](repeating: -1, count: count)
        for newIdx in 0..<count {
            let p = parent[order[newIdx]]
            newParent[newIdx] = (p == -1) ? -1 : oldToNew[p]
        }
        return (newContours, newParent)
    }

    // MARK: - Border following (Suzuki–Abe step 3 / OpenCV icvFetchContour)

    /// Traces a single border starting at `start`, marking visited pixels in
    /// `f` and returning the `CHAIN_APPROX_SIMPLE` polygon (direction-change
    /// vertices only), in ORIGINAL image coordinates.
    ///
    /// This reproduces OpenCV's `icvFetchContour` exactly, including the
    /// clockwise start search, the counter-clockwise step search, the ±NBD
    /// marking rule, and the "emit a point only when the chain direction
    /// changes" rule that produces the simplified polygon.
    ///
    /// - Parameters:
    ///   - f: The padded border image (mutated in place with ±NBD marks).
    ///   - start: Linear index in `f` of the border's starting pixel (i0).
    ///   - startX/startY: Padded coordinates of `start`.
    ///   - isHole: Whether this is a hole border (changes the initial search
    ///     direction).
    ///   - nbd: The border number used for the ±NBD marks.
    ///   - delta/codeDelta: The direction tables (see `findContours`).
    private static func followBorder(
        _ f: inout [Int],
        start i0: Int,
        startX: Int,
        startY: Int,
        isHole: Bool,
        nbd: Int,
        delta: [Int],
        codeDelta: [(dx: Int, dy: Int)]
    ) -> [CV.Point] {
        var contour: [CV.Point] = []

        // Running output point, tracking the current border pixel (i3) in
        // padded coordinates. Emitted points subtract the (1,1) padding offset.
        var px = startX
        var py = startY

        // ---------------------------------------------------------------
        // Step (3.1): from the examined background neighbour (i2,j2) search the
        // 8-neighbourhood of i0 CLOCKWISE for the first non-zero pixel (i1).
        //
        // The starting direction is the direction of the examined background
        // neighbour: West (4) for an outer border, East (0) for a hole border.
        // Clockwise means decreasing the direction index (mod 8).
        // ---------------------------------------------------------------
        let startDir = isHole ? 0 : 4   // East for holes, West for outer borders
        var s = startDir
        var i1 = -1
        repeat {
            s = (s - 1) & 7
            let n = i0 + delta[s]
            if f[n] != 0 {
                i1 = n
                break
            }
            // The `startDir` neighbour is the known background pixel, so the
            // loop terminates harmlessly once it wraps back to it.
        } while s != startDir

        // Isolated single-pixel border: mark -NBD and emit just the one point.
        if i1 == -1 {
            f[i0] = -nbd
            contour.append(CV.Point(startX - 1, startY - 1))
            return contour
        }

        // ---------------------------------------------------------------
        // Step (3.2): initialise the walk. i3 is the "current" border pixel;
        // `prevS` seeds the direction-change detector (OpenCV uses `s ^ 4`,
        // i.e. the direction opposite to the one that found i1).
        // ---------------------------------------------------------------
        var i3 = i0
        var prevS = s ^ 4

        while true {
            // -----------------------------------------------------------
            // Step (3.3): from the direction just past where we arrived, search
            // the neighbourhood of i3 COUNTER-CLOCKWISE (increasing index) for
            // the first non-zero pixel (i4). Track whether the East neighbour
            // (direction 0) was examined while still zero — this is what decides
            // the "right border pixel" mark below (OpenCV's
            // `(unsigned)(s-1) < (unsigned)s_end` test, expressed directly).
            // -----------------------------------------------------------
            let sEnd = s
            var crossedEast = false
            var i4 = -1
            var k = sEnd
            while true {
                k = (k + 1) & 7
                let n = i3 + delta[k]
                if f[n] != 0 {
                    i4 = n
                    s = k
                    break
                }
                if k == 0 { crossedEast = true }   // East examined and was zero
            }

            // -----------------------------------------------------------
            // Step (3.4): set the label of the current border pixel i3.
            //   (a) If the East neighbour was examined as a 0-pixel, i3 is a
            //       "right" border pixel → mark -NBD.
            //   (b) Else, if i3 is still unvisited foreground (== 1), mark NBD.
            //   (c) Otherwise leave its existing label untouched.
            // -----------------------------------------------------------
            if crossedEast {
                f[i3] = -nbd
            } else if f[i3] == 1 {
                f[i3] = nbd
            }

            // -----------------------------------------------------------
            // CHAIN_APPROX_SIMPLE point emission: keep i3 only when the chain
            // direction changed since the last kept vertex. This collapses
            // straight horizontal / vertical / diagonal runs to their
            // endpoints, exactly like OpenCV.
            // -----------------------------------------------------------
            if s != prevS {
                contour.append(CV.Point(px - 1, py - 1))
                prevS = s
            }
            // Advance the running point along the chosen direction so that it
            // tracks i4 (the next i3).
            px += codeDelta[s].dx
            py += codeDelta[s].dy

            // -----------------------------------------------------------
            // Step (3.5): we have closed the border when we step back onto the
            // start pixel (i4 == i0) from the second border pixel (i3 == i1).
            // -----------------------------------------------------------
            if i4 == i0 && i3 == i1 {
                break
            }

            // Otherwise continue: i4 becomes the new current pixel, and the next
            // search restarts from the opposite direction (s + 4).
            i3 = i4
            s = (s + 4) & 7
        }

        return contour
    }

    // MARK: - Hierarchy construction

    /// Builds the full `RETR_TREE` hierarchy from per-contour parent indices.
    ///
    /// Siblings (contours sharing a parent, including the virtual root for
    /// top-level contours) are linked via `next`/`previous` in discovery order,
    /// and each parent's `firstChild` points at its earliest-discovered child —
    /// matching OpenCV's `[next, previous, firstChild, parent]` quadruples.
    private static func buildTreeHierarchy(parent: [Int], count: Int) -> [HierarchyNode] {
        var hierarchy = [HierarchyNode](repeating: HierarchyNode(), count: count)

        // For each parent (keyed by contour index, with -1 = virtual root),
        // remember the previously seen child so we can chain siblings.
        var lastSibling: [Int: Int] = [:]

        for c in 0..<count {
            let p = parent[c]
            hierarchy[c].parent = p

            if let prev = lastSibling[p] {
                // Link this contour after its previous sibling.
                hierarchy[c].previous = prev
                hierarchy[prev].next = c
            } else if p != -1 {
                // First child encountered for parent `p`.
                hierarchy[p].firstChild = c
            }
            lastSibling[p] = c
        }

        return hierarchy
    }

    /// Builds the `RETR_EXTERNAL` result: only the outermost contours (those
    /// whose parent is the frame) are kept, re-indexed, and chained together as
    /// siblings with `parent = -1` and `firstChild = -1`.
    private static func buildExternalResult(
        contours: [[CV.Point]],
        parent: [Int],
        count: Int
    ) -> ContourResult {
        var extContours: [[CV.Point]] = []
        var extHierarchy: [HierarchyNode] = []
        var prev = -1

        for c in 0..<count where parent[c] == -1 {
            let newIndex = extContours.count
            extContours.append(contours[c])
            // Chain to the previous external contour; next is filled in when the
            // following sibling is appended.
            extHierarchy.append(
                HierarchyNode(next: -1, previous: prev, firstChild: -1, parent: -1)
            )
            if prev != -1 {
                extHierarchy[prev].next = newIndex
            }
            prev = newIndex
        }

        return ContourResult(contours: extContours, hierarchy: extHierarchy)
    }
}
