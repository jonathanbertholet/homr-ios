import Foundation

// MARK: - Staff detection (port of homr/staff_detection.py)
//
// Detects musical staffs from the segmentation masks. The pipeline:
//   1. find "anchors" — five parallel staff-line fragments that pass over a
//      symbol which is always ON a staff (clefs, then bar-lines/rests),
//   2. connect line fragments through those anchors into raw staffs,
//   3. resample each raw staff onto a regular x-grid so every x has the y of
//      all five lines (`Staff`/`StaffPoint`),
//   4. filter staffs at the edge of the page and sort top-to-bottom.
//
// Adaptations from the Python source (documented per declaration):
//   * All `Debug` / `cv2.putText` / `cv2.line` / `draw_onto_image` debug code is
//     dropped; `detectStaff` takes no `Debug` argument.
//   * Images are `GrayscaleImage` (0/255 masks) instead of numpy arrays.
//   * Python `range` objects are modelled by `StaffRange` (a half-open `[start,
//     stop)` integer interval) so the `.stop`-after-iteration semantics survive.
//   * `RawStaff` cannot subclass the `final` `RotatedBoundingBox`, so it COMPOSES
//     one (`box`) and re-exposes the members the algorithm needs.
//   * `connect_staff_lines` relies on Python list aliasing (a group appended to
//     both `result` and `active` is mutated in place); reproduced with the
//     reference type `LineGroup`.
//   * `int()` → toward zero, `round()`/`np.round` → banker's (`.toNearestOrEven`),
//     `//` → floor. `np.std` is population std (ddof=0).

// MARK: - Small helpers

/// Half-open integer interval mirroring Python's `range(start, stop)` (step 1).
/// Iteration yields `start ..< stop`; `.stop` is preserved verbatim so callers
/// can read it after the fact (homr does `x = to_right.stop`).
struct StaffRange {
    var start: Int
    var stop: Int
    var width: Int { max(0, stop - start) }
    func contains(_ value: Int) -> Bool { value >= start && value < stop }
}

/// Population mean of a `[Double]` (`np.mean`/`np.average`), 0 for an empty input.
private func sdMean(_ values: [Double]) -> Double {
    values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
}

/// Population standard deviation (`np.std`, ddof=0), 0 for an empty input.
private func sdStd(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let m = sdMean(values)
    let variance = values.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(values.count)
    return variance.squareRoot()
}

/// Consecutive differences (`np.diff`).
private func sdDiff(_ values: [Double]) -> [Double] {
    guard values.count > 1 else { return [] }
    return (1..<values.count).map { values[$0] - values[$0 - 1] }
}

/// Python `range(start, stop, step)` materialised as a list (positive step only,
/// matching every call site here).
private func pyRange(_ start: Int, _ stop: Int, _ step: Int) -> [Int] {
    guard step > 0 else { return [] }
    var result: [Int] = []
    var v = start
    while v < stop { result.append(v); v += step }
    return result
}

/// `int(round(x / density)) * density` — homr's `round_to_density` (banker's round).
private func roundToDensity(_ x: Double, _ density: Int) -> Int {
    Int((x / Double(density)).rounded(.toNearestOrEven)) * density
}

/// Extracts the columns `[range.start, range.stop)` of an image as a new image
/// (`image[:, zone]`). Out-of-range columns are clamped away.
private func columnSlice(_ image: GrayscaleImage, _ range: StaffRange) -> GrayscaleImage {
    let x0 = max(0, range.start)
    let x1 = min(image.width, range.stop)
    let w = max(0, x1 - x0)
    let h = image.height
    guard w > 0, h > 0 else { return GrayscaleImage(pixels: [], width: 0, height: 0) }
    var out = [UInt8](repeating: 0, count: w * h)
    image.pixels.withUnsafeBufferPointer { src in
        out.withUnsafeMutableBufferPointer { dst in
            for row in 0..<h {
                let srcRow = row * image.width + x0
                let dstRow = row * w
                for col in 0..<w { dst[dstRow + col] = src[srcRow + col] }
            }
        }
    }
    return GrayscaleImage(pixels: out, width: w, height: h)
}

// MARK: - prepare_staff_image / make_lines_stronger

/// Removes small details from a staff mask (`prepare_staff_image`): erode with a
/// 5×3 ellipse then dilate with a 9×3 ellipse. Not on the main detection path
/// but ported for completeness.
func prepareStaffImageForDetection(_ image: GrayscaleImage) -> GrayscaleImage {
    let erodeKernel = CV.getStructuringElement(.ellipse, (width: 5, height: 3))
    let eroded = CV.erode(image, erodeKernel)
    let dilateKernel = CV.getStructuringElement(.ellipse, (width: 9, height: 3))
    return CV.dilate(eroded, dilateKernel)
}

/// Thickens/binarises staff lines (`make_lines_stronger`): dilate with an ellipse
/// of `kernelSize` then threshold any non-zero pixel to 255.
///
/// Python uses `cv2.threshold(img, 0.1, 1, THRESH_BINARY)` because its masks are
/// 0/1; ours are already 0/255, so we binarise with `thresh: 0` → 255.
func makeLinesStronger(_ image: GrayscaleImage, kernelSize: (width: Int, height: Int)) -> GrayscaleImage {
    let kernel = CV.getStructuringElement(.ellipse, kernelSize)
    let dilated = CV.dilate(image, kernel)
    return CV.threshold(dilated, thresh: 0, maxValue: 255, type: .binary)
}

// MARK: - StaffLineSegment

/// A run of staff-line fragments that connect into one (possibly broken) line.
/// Port of `StaffLineSegment`. Drawing dropped.
final class StaffLineSegment: Hashable {
    let debugId: Int
    /// Fragments sorted left-to-right by their rotated-rect centre x.
    let staffFragments: [RotatedBoundingBox]
    let minX: Double
    let maxX: Double
    let minY: Double
    let maxY: Double

    init(debugId: Int, staffFragments fragments: [RotatedBoundingBox]) {
        self.debugId = debugId
        let sorted = fragments.sorted { $0.box.center.x < $1.box.center.x }
        self.staffFragments = sorted
        self.minX = fragments.map { $0.center.x - $0.size.width / 2 }.min() ?? 0
        self.maxX = fragments.map { $0.center.x + $0.size.width / 2 }.max() ?? 0
        self.minY = fragments.map { $0.center.y - $0.size.height / 2 }.min() ?? 0
        self.maxY = fragments.map { $0.center.y + $0.size.height / 2 }.max() ?? 0
    }

    /// Union of fragments from both segments (dedup by fragment equality).
    func merge(_ other: StaffLineSegment) -> StaffLineSegment {
        var lines = staffFragments
        for fragment in other.staffFragments where !lines.contains(fragment) {
            lines.append(fragment)
        }
        return StaffLineSegment(debugId: debugId, staffFragments: lines)
    }

    /// Fragment covering `x` (within `staff_line_segment_x_tolerance`), or nil.
    func getAt(_ x: Double) -> RotatedBoundingBox? {
        let tolerance = Double(Constants.staffLineSegmentXTolerance)
        for fragment in staffFragments {
            if x >= fragment.center.x - fragment.size.width / 2 - tolerance
                && x <= fragment.center.x + fragment.size.width / 2 + tolerance {
                return fragment
            }
        }
        return nil
    }

    /// True if any fragment of `self` overlaps any fragment of `other`.
    func isOverlapping(_ other: StaffLineSegment) -> Bool {
        for line in staffFragments {
            for otherLine in other.staffFragments where line.isOverlapping(otherLine) {
                return true
            }
        }
        return false
    }

    // __eq__/__hash__ over the (unordered) set of fragments (Python frozenset).
    static func == (lhs: StaffLineSegment, rhs: StaffLineSegment) -> Bool {
        Set(lhs.staffFragments) == Set(rhs.staffFragments)
    }

    func hash(into hasher: inout Hasher) {
        // Order-independent hash to match the frozenset semantics.
        var combined = 0
        for fragment in staffFragments { combined ^= fragment.hashValue }
        hasher.combine(combined)
    }
}

// MARK: - StaffAnchor

/// A reliable staff location: five parallel lines passing through an anchor
/// symbol. Port of `StaffAnchor`. Drawing dropped.
final class StaffAnchor {
    let staffLines: [StaffLineSegment]
    let unitSizes: [Double]
    let averageUnitSize: Double
    let symbol: RotatedBoundingBox
    let maxY: Double
    let minY: Double
    let yRange: StaffRange
    let zone: StaffRange

    init(staffLines: [StaffLineSegment], symbol: RotatedBoundingBox) {
        self.staffLines = staffLines
        let yPositions = staffLines
            .map { $0.staffFragments[0].getCenterExtrapolated(symbol.center.x) }
            .sorted()
        let yDeltas = yPositions.count > 1
            ? (1..<yPositions.count).map { abs(yPositions[$0] - yPositions[$0 - 1]) }
            : []
        self.unitSizes = yDeltas
        self.averageUnitSize = yDeltas.isEmpty ? 0.0 : sdMean(yDeltas)
        self.symbol = symbol
        self.maxY = staffLines.map { $0.maxY }.max() ?? 0
        self.minY = staffLines.map { $0.minY }.min() ?? 0
        // Local constant in homr (5), distinct from Constants.maxNumberOfLedgerLines.
        let maxNumberOfLedgerLines = 5.0
        let lo = yPositions.min() ?? 0
        let hi = yPositions.max() ?? 0
        self.yRange = StaffRange(start: Int(lo), stop: Int(hi))
        self.zone = StaffRange(
            start: Int(minY - maxNumberOfLedgerLines * averageUnitSize),
            stop: Int(maxY + maxNumberOfLedgerLines * averageUnitSize)
        )
    }
}

// MARK: - RawStaff

/// A staff assembled from found parts; has gaps and uneven line ends. Port of
/// `RawStaff`. Python subclasses `RotatedBoundingBox`; since that type is `final`
/// here, `RawStaff` COMPOSES one (`box`) and re-exposes the needed members.
final class RawStaff {
    let staffId: Int
    let lines: [StaffLineSegment]
    let anchors: [StaffAnchor]
    let box: RotatedBoundingBox
    let minX: Double
    let maxX: Double
    let minY: Double
    let maxY: Double

    init(staffId: Int, lines: [StaffLineSegment], anchors: [StaffAnchor]) {
        self.staffId = staffId
        self.lines = lines
        self.anchors = anchors
        // contours = every contour point of every fragment of every line.
        var allContours: [CV.Point] = []
        for line in lines {
            for fragment in line.staffFragments { allContours.append(contentsOf: fragment.contours) }
        }
        let rect = CV.minAreaRect(allContours)
        self.box = RotatedBoundingBox(box: rect, contours: allContours, debugId: staffId)
        self.minX = box.center.x - box.size.width / 2
        self.maxX = box.center.x + box.size.width / 2
        self.minY = box.center.y - box.size.height / 2
        self.maxY = box.center.y + box.size.height / 2
    }

    /// Merge line-by-line (Python pairs `other.lines[i].merge(self.lines[i])`).
    func merge(_ other: RawStaff) -> RawStaff {
        var mergedLines: [StaffLineSegment] = []
        for (i, line) in lines.enumerated() {
            mergedLines.append(other.lines[i].merge(line))
        }
        return RawStaff(staffId: staffId, lines: mergedLines, anchors: anchors + other.anchors)
    }

    /// Polygon overlap of the bounding boxes (inherited behaviour in Python).
    func isOverlapping(_ other: RawStaff) -> Bool {
        box.isOverlapping(other.box)
    }
}

private func getAllContours(_ lines: [StaffLineSegment]) -> [CV.Point] {
    var result: [CV.Point] = []
    for line in lines {
        for fragment in line.staffFragments { result.append(contentsOf: fragment.contours) }
    }
    return result
}

// MARK: - Raw-staff assembly

/// Returns the staff already containing one of the anchor's lines (`get_staff_for_anchor`).
private func getStaffForAnchor(_ anchor: StaffAnchor, _ staffs: [RawStaff]) -> RawStaff? {
    for staff in staffs {
        for (i, anchorLine) in anchor.staffLines.enumerated() {
            let requirement = Set(anchorLine.staffFragments)
            if i < staff.lines.count, requirement.isSubset(of: Set(staff.lines[i].staffFragments)) {
                return staff
            }
        }
    }
    return nil
}

/// Builds raw staffs by connecting line fragments through each anchor
/// (`find_raw_staffs_by_connecting_line_fragments`).
func findRawStaffsByConnectingLineFragments(
    anchors: [StaffAnchor],
    staffFragments: [RotatedBoundingBox]
) -> [RawStaff] {
    var staffs: [RawStaff] = []
    var staffId = 0
    for anchor in anchors {
        let existingStaff = getStaffForAnchor(anchor, staffs)
        let fragments = staffFragments.filter {
            Double($0.center.y) >= Double(anchor.zone.start)
                && Double($0.center.y) <= Double(anchor.zone.stop)
        }
        let connected = connectStaffLines(fragments, unitSize: anchor.averageUnitSize)
        var staffLines: [StaffLineSegment] = []
        for anchorLine in anchor.staffLines {
            let requirement = Set(anchorLine.staffFragments)
            let matchingAnchor = connected.filter { requirement.isSubset(of: Set($0.staffFragments)) }
            if matchingAnchor.count == 1 {
                staffLines.append(contentsOf: matchingAnchor)
            } else {
                staffLines.append(anchorLine)
            }
        }
        if let existingStaff = existingStaff {
            staffs.removeAll { $0 === existingStaff }
            staffs.append(existingStaff.merge(RawStaff(staffId: staffId, lines: staffLines, anchors: [anchor])))
        } else {
            staffs.append(RawStaff(staffId: staffId, lines: staffLines, anchors: [anchor]))
        }
        staffId += 1
    }
    return staffs
}

/// Removes staffs found more than once (`remove_duplicate_staffs`).
func removeDuplicateStaffs(_ staffs: [RawStaff]) -> [RawStaff] {
    var result: [RawStaff] = []
    for staff in staffs {
        let overlapping = result.filter { staff.isOverlapping($0) }
        if overlapping.isEmpty {
            result.append(staff)
            continue
        }
        let staffDuplicates = 2
        if overlapping.count >= staffDuplicates {
            // Ambiguous: keep the existing ones (homr's documented choice).
            continue
        }
        if overlapping[0].anchors.count < staff.anchors.count {
            // The staff with the most anchors is the most reliable one.
            result = result.filter { $0 !== overlapping[0] }
            result.append(staff)
        }
    }
    return result
}

/// Reference wrapper so a group appended to both `result` and `active` is shared
/// and mutated in place, reproducing Python list aliasing in `connect_staff_lines`.
private final class LineGroup {
    var items: [RotatedBoundingBox]
    init(_ first: RotatedBoundingBox) { self.items = [first] }
}

/// Connects fragments left-to-right into `StaffLineSegment`s, extrapolating over
/// gaps (`connect_staff_lines`).
func connectStaffLines(_ staffLines: [RotatedBoundingBox], unitSize: Double) -> [StaffLineSegment] {
    // Sorted right-to-left so popping the last element walks left-to-right.
    var sortedRightToLeft = staffLines.sorted { $0.bottomLeft.x > $1.bottomLeft.x }
    var result: [LineGroup] = []
    var activeLinesToCheck: [LineGroup] = []
    var lastCleanupAtX: Double = 0

    while !sortedRightToLeft.isEmpty {
        let current = sortedRightToLeft.removeLast()
        let x = current.bottomLeft.x

        if x - lastCleanupAtX > Constants.maxLineGapSize(unitSize) {
            // Drop line ends too far left to ever connect again.
            activeLinesToCheck = activeLinesToCheck.filter {
                x - $0.items[$0.items.count - 1].bottomRight.x < Constants.maxLineGapSize(unitSize)
            }
            lastCleanupAtX = x
        }

        let isShortLine = current.box.size.width < Constants.isShortLine(unitSize)
        if isShortLine { continue }

        var connected = false
        for activeLine in activeLinesToCheck {
            if activeLine.items[activeLine.items.count - 1].isOverlappingExtrapolated(current, unitSize: unitSize) {
                activeLine.items.append(current)
                connected = true
            }
        }
        if !connected {
            let newGroup = LineGroup(current)
            result.append(newGroup)
            activeLinesToCheck.append(newGroup)
        }
    }

    let resultTopToBottom = result.sorted { $0.items[0].box.center.y < $1.items[0].box.center.y }
    return resultTopToBottom.enumerated().map { StaffLineSegment(debugId: $0.offset, staffFragments: $0.element.items) }
}

// MARK: - Geometric predicates

/// True if any two segments overlap (`are_lines_crossing`).
private func areLinesCrossing(_ lines: [StaffLineSegment]) -> Bool {
    for i in 0..<lines.count {
        for j in (i + 1)..<lines.count where lines[i].isOverlapping(lines[j]) {
            return true
        }
    }
    return false
}

/// True if the lines are (approximately) parallel (`are_lines_parallel`).
private func areLinesParallel(_ lines: [StaffLineSegment], unitSize: Double) -> Bool {
    var allAngles: [Double] = []
    var allFragments: [RotatedBoundingBox] = []
    for line in lines {
        for fragment in line.staffFragments {
            allAngles.append(fragment.angle)
            allFragments.append(fragment)
        }
    }
    if allAngles.isEmpty { return false }
    let averageAngle = sdMean(allAngles)
    for fragment in allFragments {
        if abs(fragment.angle - averageAngle) > Double(Constants.maxAngleForLinesToBeParallel)
            && fragment.size.width > Constants.isShortConnectedLine(unitSize) {
            return false
        }
    }
    return true
}

/// True if a candidate begins/ends on a staff line (`begins_or_ends_on_one_staff_line`).
private func beginsOrEndsOnOneStaffLine(
    _ line: RotatedBoundingBox,
    staffLines: [StaffLineSegment],
    unitSize: Double
) -> Bool {
    for staffLine in staffLines {
        guard let fragment = staffLine.getAt(line.center.x) else { continue }
        let staffY = fragment.getCenterExtrapolated(line.center.x)
        if abs(staffY - line.center.y) < unitSize { return true }
    }
    return false
}

// MARK: - find_staff_anchors

/// Finds staff anchors around the given symbols (`find_staff_anchors`).
func findStaffAnchors(
    staffLines: [RotatedBoundingBox],
    anchorSymbols: [RotatedBoundingBox],
    areClefs: Bool = false
) -> [StaffAnchor] {
    var result: [StaffAnchor] = []
    for centerSymbol in anchorSymbols {
        // The symbol disconnects the lines, so we also probe to its left/right.
        let adjacent: [RotatedBoundingBox]
        if areClefs {
            adjacent = [
                centerSymbol.moveToXHorizontalBy(-10),
                centerSymbol,
                centerSymbol.moveToXHorizontalBy(10),
                centerSymbol.moveToXHorizontalBy(30),
                centerSymbol.moveToXHorizontalBy(60),
                centerSymbol.moveToXHorizontalBy(80),
            ]
        } else {
            adjacent = [
                centerSymbol.moveToXHorizontalBy(-10),
                centerSymbol.moveToXHorizontalBy(-5),
                centerSymbol,
                centerSymbol.moveToXHorizontalBy(5),
                centerSymbol.moveToXHorizontalBy(10),
            ]
        }
        for symbol in adjacent {
            let estimatedUnitSize = Int(
                (symbol.size.height / Double(Constants.numberOfLinesOnAStaff - 1)).rounded(.toNearestOrEven)
            )
            let thickenedBarLine = symbol.makeBoxTaller(estimatedUnitSize)
            var connectedLines = connectStaffLines(
                staffLines.filter { $0.isIntersecting(thickenedBarLine) },
                unitSize: Double(estimatedUnitSize)
            )
            if connectedLines.count > Constants.numberOfLinesOnAStaff {
                connectedLines = connectedLines.filter {
                    ($0.maxX - $0.minX) > Constants.isShortConnectedLine(Double(estimatedUnitSize))
                }
            }
            if connectedLines.count != Constants.numberOfLinesOnAStaff { continue }
            if !areLinesParallel(connectedLines, unitSize: Double(estimatedUnitSize)) { continue }
            if areLinesCrossing(connectedLines) { continue }
            if !areClefs && !beginsOrEndsOnOneStaffLine(symbol, staffLines: connectedLines, unitSize: Double(estimatedUnitSize)) {
                continue
            }
            result.append(StaffAnchor(staffLines: connectedLines, symbol: symbol))
        }
    }
    return result
}

// MARK: - Resampling

/// Resamples one staff segment along `axisRange`, yielding `StaffPoint`s
/// (`resample_staff_segment`, a Python generator → array here).
///
/// Replicates homr's compacted-index quirk: `deltas` is computed over the
/// non-nil centres while the invalidation indexes the full (nil-bearing)
/// `axisCenter` list by the compacted index.
private func resampleStaffSegment(
    anchor: StaffAnchor,
    staff: RawStaff,
    axisRange: [Int]
) -> [StaffPoint] {
    var output: [StaffPoint] = []
    let x0 = anchor.symbol.center.x
    let lineFragments = anchor.staffLines.map { $0.staffFragments[0] }
    let centers = lineFragments.map { $0.getCenterExtrapolated(x0) }
    var previousPoint = StaffPoint(
        x: x0,
        y: centers,
        angle: sdMean(lineFragments.map { $0.angle })
    )

    for x in axisRange {
        let xd = Double(x)
        let lines = staff.lines.map { $0.getAt(xd) }
        var axisCenter: [Double?] = lines.map { $0?.getCenterExtrapolated(xd) }
        let centerValues = axisCenter.compactMap { $0 }
        let incompleteAll = axisCenter.allSatisfy { $0 == nil }
        if incompleteAll { continue }

        let deltas = sdDiff(centerValues)
        let nonParallel = deltas.map { $0 < 0.5 * anchor.averageUnitSize }
        for (i, invalid) in nonParallel.enumerated() where invalid {
            if i < axisCenter.count { axisCenter[i] = nil }
            if i + 1 < axisCenter.count { axisCenter[i + 1] = nil }
        }

        for (i, previousY) in previousPoint.y.enumerated() where i < axisCenter.count {
            if let centerValue = axisCenter[i],
               abs(centerValue - previousY) > 0.5 * anchor.averageUnitSize {
                axisCenter[i] = nil
            }
        }

        // Forward then backward fill of gaps using the average unit size.
        var prevCenter = -1
        let order = Array(0..<axisCenter.count) + Array((0..<axisCenter.count).reversed())
        for i in order {
            if axisCenter[i] != nil {
                prevCenter = i
            } else if prevCenter >= 0, let centerValue = axisCenter[prevCenter] {
                axisCenter[i] = centerValue + anchor.averageUnitSize * Double(i - prevCenter)
            }
        }

        let incompleteAny = axisCenter.contains { $0 == nil }
        if incompleteAny { continue }
        let angle = sdMean(zip(lines, axisCenter).compactMap { line, _ in line?.angle })
        previousPoint = StaffPoint(x: xd, y: axisCenter.compactMap { $0 }, angle: angle)
        output.append(previousPoint)
    }
    return output
}

/// Resamples a whole raw staff onto a regular grid (`resample_staff`).
private func resampleStaff(_ staff: RawStaff) -> Staff {
    let anchorsLeftToRight = staff.anchors.sorted { $0.symbol.center.x < $1.symbol.center.x }
    let staffDensity = 10

    let start = (staff.minX / Double(staffDensity)).rounded(.down) * Double(staffDensity)
    let stop = ((staff.maxX / Double(staffDensity)).rounded(.down) + 1) * Double(staffDensity)

    var grid: [StaffPoint] = []
    var x = start
    for (i, anchor) in anchorsLeftToRight.enumerated() {
        let toLeft = StaffRange(
            start: roundToDensity(x, staffDensity),
            stop: roundToDensity(anchor.symbol.center.x, staffDensity)
        )
        let toRight: StaffRange
        if i < anchorsLeftToRight.count - 1 {
            toRight = StaffRange(
                start: Int(anchor.symbol.center.x),
                stop: Int((anchor.symbol.center.x + anchorsLeftToRight[i + 1].symbol.center.x) / 2)
            )
        } else {
            toRight = StaffRange(
                start: roundToDensity(anchor.symbol.center.x, staffDensity),
                stop: roundToDensity(stop, staffDensity)
            )
        }
        x = Double(toRight.stop)

        let toLeftValues = pyRange(toLeft.start, toLeft.stop, staffDensity)
        let leftSegment = resampleStaffSegment(anchor: anchor, staff: staff, axisRange: Array(toLeftValues.reversed()))
        grid.append(contentsOf: leftSegment.reversed())

        let toRightValues = pyRange(toRight.start, toRight.stop, staffDensity)
        grid.append(contentsOf: resampleStaffSegment(anchor: anchor, staff: staff, axisRange: toRightValues))
    }

    return Staff(grid: grid)
}

/// Resamples all raw staffs (`resample_staffs`).
private func resampleStaffs(_ staffs: [RawStaff]) -> [Staff] {
    staffs.map { resampleStaff($0) }
}

/// Removes staffs outside the page (`filter_edge_of_vision`).
private func filterEdgeOfVision(_ staffs: [Staff], imageHeight: Int, imageWidth: Int) -> [Staff] {
    let staffWidths = staffs.map { $0.maxX - $0.minX }
    let usualWidth = sdMean(staffWidths)
    var result: [Staff] = []
    for staff in staffs {
        let beyondBottom = staff.maxY >= Double(imageHeight)
        let beyondTop = staff.minY < 0
        if beyondBottom || beyondTop { continue }

        let staffWidth = staff.maxX - staff.minX
        let shorterThanUsual = staffWidth < usualWidth / 2
        let beyondLeft = staff.minX < 0.01 * Double(imageWidth)
        let beyondRight = staff.maxX > 0.99 * Double(imageWidth)
        if (beyondLeft || beyondRight) && shorterThanUsual { continue }
        result.append(staff)
    }
    return result
}

/// Sorts staffs top-to-bottom (`sort_staffs_top_to_bottom`).
private func sortStaffsTopToBottom(_ staffs: [Staff]) -> [Staff] {
    staffs.sorted { $0.minY < $1.minY }
}

/// Drops anchors with an unusual unit size (`filter_unusual_anchors`).
private func filterUnusualAnchors(_ anchors: [StaffAnchor]) -> [StaffAnchor] {
    if anchors.isEmpty { return anchors }
    let unitSizes = anchors.map { $0.averageUnitSize }
    let averageUnitSize = sdMean(unitSizes)
    let deviation = sdStd(unitSizes)
    return anchors.filter { abs($0.averageUnitSize - averageUnitSize) <= 3 * deviation }
}

// MARK: - Horizontal-line detection from clefs

/// Builds the (clef-zone) x ranges to search for additional staff lines
/// (`init_zone`).
private func initZone(_ clefAnchors: [StaffAnchor], imageWidth: Int) -> [StaffRange] {
    func makeRange(_ start: Double, _ stop: Double) -> StaffRange {
        StaffRange(start: max(Int(start), 0), stop: min(Int(stop), imageWidth))
    }
    let marginRight = 10.0
    let ranges = clefAnchors
        .map { makeRange($0.symbol.bottomLeft.x, $0.symbol.topRight.x + marginRight) }
        .sorted { $0.start < $1.start }
    var result: [StaffRange] = []
    for (i, r) in ranges.enumerated() {
        if i == 0 {
            result.append(r)
        } else if r.start < result[result.count - 1].stop {
            result[result.count - 1] = StaffRange(start: result[result.count - 1].start, stop: r.stop)
        } else {
            result.append(r)
        }
    }
    return result
}

/// Filters detected line peaks into complete groups (`filter_line_peaks`).
/// Returns `(validPeaks, groups)`; `validPeaks` is computed for fidelity but the
/// caller (`findHorizontalLines`) only uses `groups`.
private func filterLinePeaks(_ peaks: [Int], _ norm: [Double], maxGapRatio: Double = 1.5) -> (valid: [Bool], groups: [Int]) {
    var validPeaks = [Bool](repeating: true, count: peaks.count)
    let maxPeakHeight = 15.0
    for (idx, p) in peaks.enumerated() where p >= 0 && p < norm.count {
        if norm[p] > maxPeakHeight { validPeaks[idx] = false }
    }

    guard peaks.count > 1 else { return (validPeaks, peaks.isEmpty ? [] : [0]) }

    let gaps = (1..<peaks.count).map { Double(peaks[$0] - peaks[$0 - 1]) }
    let count0 = max(5, Int((Double(peaks.count) * 0.2).rounded(.toNearestOrEven)))
    let sortedGaps = gaps.sorted()
    let approxUnit = sdMean(Array(sortedGaps.prefix(count0)))
    let maxGap = approxUnit * maxGapRatio

    // Prepend an invalid peak to handle the leading edge.
    var extPeaks: [Double] = [Double(peaks[0]) - maxGap - 1]
    extPeaks.append(contentsOf: peaks.map { Double($0) })

    var groups: [Int] = []
    var group = -1
    for i in 1..<extPeaks.count {
        if extPeaks[i] - extPeaks[i - 1] > maxGap { group += 1 }
        groups.append(group)
    }
    groups.append((groups.last ?? -1) + 1) // trailing invalid group

    var curG = groups[0]
    var count = 1
    let lines = Constants.numberOfLinesOnAStaff
    for idx in 1..<groups.count {
        let g = groups[idx]
        if g == curG { count += 1; continue }
        if count < lines {
            for k in max(0, idx - count)..<min(idx, validPeaks.count) { validPeaks[k] = false }
        } else if count > lines {
            let lo = idx - count
            let candPeaks = Array(peaks[max(0, lo)..<min(idx, peaks.count)])
            let headPart = Array(candPeaks.prefix(lines))
            let tailPart = Array(candPeaks.suffix(lines))
            let headSum = headPart.reduce(0.0) { $0 + (($1 >= 0 && $1 < norm.count) ? norm[$1] : 0) }
            let tailSum = tailPart.reduce(0.0) { $0 + (($1 >= 0 && $1 < norm.count) ? norm[$1] : 0) }
            if headSum > tailSum {
                for k in max(0, lo + lines)..<min(idx, validPeaks.count) { validPeaks[k] = false }
            } else {
                for k in max(0, lo)..<min(idx - lines, validPeaks.count) { validPeaks[k] = false }
            }
        }
        curG = g
        count = 1
    }

    return (validPeaks, Array(groups.dropLast()))
}

/// Detects horizontal staff-line groups in an image slice (`find_horizontal_lines`).
private func findHorizontalLines(_ image: GrayscaleImage, unitSize: Double, lineThreshold: Double = 0.0) -> [[Int]] {
    let height = image.height
    guard height > 0, image.width > 0 else { return [] }

    // Per-row foreground-pixel count.
    var count = [Double](repeating: 0, count: height)
    image.pixels.withUnsafeBufferPointer { src in
        for y in 0..<height {
            let base = y * image.width
            var rowCount = 0.0
            for xCol in 0..<image.width where src[base + xCol] > 0 { rowCount += 1 }
            count[y] = rowCount
        }
    }
    // np.insert(count, [0, len], [0, 0]) → prepend and append a 0.
    var padded = [0.0]
    padded.append(contentsOf: count)
    padded.append(0.0)

    let meanC = sdMean(padded)
    let stdC = sdStd(padded)
    guard stdC != 0 else { return [] } // uniform slice → no lines (avoids NaN)
    let norm = padded.map { ($0 - meanC) / stdC }

    let (peaks, _) = findPeaks(norm, height: lineThreshold, distance: unitSize, prominence: 1)
    let centers = peaks.map { $0 - 1 }
    let trimmedNorm = Array(norm[1..<(norm.count - 1)])
    let (_, groups) = filterLinePeaks(centers, trimmedNorm)

    var groupedCenters: [Int: [Int]] = [:]
    for (i, center) in centers.enumerated() where i < groups.count {
        groupedCenters[groups[i], default: []].append(center)
    }
    var completeGroups: [[Int]] = []
    // Iterate in insertion order of groups for deterministic output.
    for groupNumber in groupedCenters.keys.sorted() {
        if let g = groupedCenters[groupNumber], g.count == Constants.numberOfLinesOnAStaff {
            completeGroups.append(g.sorted())
        }
    }
    return completeGroups
}

/// Predicts additional staff anchors from detected clefs (`predict_other_anchors_from_clefs`).
private func predictOtherAnchorsFromClefs(_ clefAnchors: [StaffAnchor], image: GrayscaleImage) -> [RotatedBoundingBox] {
    if clefAnchors.isEmpty { return [] }
    let averageUnitSize = sdMean(clefAnchors.map { $0.averageUnitSize })
    let anchorSymbols = clefAnchors.map { $0.symbol }
    let clefZones = initZone(clefAnchors, imageWidth: image.width)
    var result: [RotatedBoundingBox] = []
    for zone in clefZones {
        let verticalSlice = columnSlice(image, zone)
        let linesGroups = findHorizontalLines(verticalSlice, unitSize: averageUnitSize)
        for group in linesGroups {
            guard let minY = group.min(), let maxY = group.max() else { continue }
            let centerY = Double(minY + maxY) / 2
            let centerX = Double(zone.start) + Double(zone.stop - zone.start) / 2
            let rect = CV.RotatedRect(
                center: CV.PointF(centerX, centerY),
                size: CV.Size(width: Double(zone.stop - zone.start), height: Double(maxY - minY)),
                angle: 0
            )
            result.append(RotatedBoundingBox(box: rect, contours: [], debugId: 0))
        }
    }
    return result.filter { !$0.isOverlappingWithAny(anchorSymbols) }
}

// MARK: - break_wide_fragments

/// Splits wide (curved) fragments into smaller parts (`break_wide_fragments`).
func breakWideFragments(_ fragments: [RotatedBoundingBox], limit: Int = 100) -> [RotatedBoundingBox] {
    var result: [RotatedBoundingBox] = []
    for fragment in fragments {
        var remainingFragment = fragment
        while remainingFragment.size.width > Double(limit) {
            guard let minX = remainingFragment.contours.map({ $0.x }).min() else { break }
            var contoursLeft = remainingFragment.contours.filter { $0.x < minX + limit }
            var contoursRight = remainingFragment.contours.filter { $0.x >= minX + limit }
            contoursLeft.sort { $0.x < $1.x }
            contoursRight.sort { $0.x < $1.x }
            if contoursLeft.isEmpty || contoursRight.isEmpty { break }
            // Keep the parts connected by sharing a boundary point each way.
            contoursLeft.append(contoursRight[0])
            contoursRight.append(contoursLeft[contoursLeft.count - 1])
            result.append(createRotatedBoundingBox(contoursLeft, debugId: remainingFragment.debugId))
            remainingFragment = createRotatedBoundingBox(contoursRight, debugId: remainingFragment.debugId)
        }
        result.append(remainingFragment)
    }
    return result
}

// MARK: - detect_staff (top level)

/// Detects staffs on the image (`detect_staff`). The `Debug` argument and all
/// debug drawing are dropped.
func detectStaff(
    image: GrayscaleImage,
    staffFragments: [RotatedBoundingBox],
    clefsKeys: [RotatedBoundingBox],
    likelyBarOrRestsLines: [RotatedBoundingBox]
) -> [Staff] {
    var staffAnchors = findStaffAnchors(staffLines: staffFragments, anchorSymbols: clefsKeys, areClefs: true)

    let possibleOtherClefs = predictOtherAnchorsFromClefs(staffAnchors, image: image)
    staffAnchors.append(contentsOf: findStaffAnchors(staffLines: staffFragments, anchorSymbols: possibleOtherClefs, areClefs: true))

    staffAnchors.append(contentsOf: findStaffAnchors(staffLines: staffFragments, anchorSymbols: likelyBarOrRestsLines, areClefs: false))

    staffAnchors = filterUnusualAnchors(staffAnchors)

    let rawStaffsWithPossibleDuplicates = findRawStaffsByConnectingLineFragments(
        anchors: staffAnchors,
        staffFragments: staffFragments
    )
    let rawStaffs = removeDuplicateStaffs(rawStaffsWithPossibleDuplicates)

    var staffs = resampleStaffs(rawStaffs)
    staffs = filterEdgeOfVision(staffs, imageHeight: image.height, imageWidth: image.width)
    staffs = sortStaffsTopToBottom(staffs)
    return staffs
}
