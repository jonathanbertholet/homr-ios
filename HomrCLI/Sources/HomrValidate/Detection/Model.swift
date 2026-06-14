import Foundation

/// Detected-symbol and staff model — port of `homr/model.py`.
///
/// These types are the structured output of symbol detection: notes, rests,
/// clefs, bar lines, accidentals, and the `Staff`/`MultiStaff` containers that
/// position them. Python mutates them in place (e.g. `note.beams.append(...)`,
/// `staff.symbols.append(...)`, `note.has_dot = True`) and relies on object
/// identity in lists, so they are implemented as reference types (`final class`).
///
/// Coordinate convention: a symbol/staff `center` is a `CV.PointF` (`.x` / `.y`),
/// matching Python's `tuple[float, float]` where callers use `center[0]`/`center[1]`.

// MARK: - Small numeric helpers (numpy stand-ins)

/// Arithmetic mean of a list (numpy `np.mean`). Returns NaN for an empty list,
/// matching numpy's behavior.
private func mean(_ values: [Double]) -> Double {
    if values.isEmpty {
        return Double.nan
    }
    return values.reduce(0, +) / Double(values.count)
}

/// Median of a list (numpy `np.median`): the average of the two middle values
/// for an even count, otherwise the middle value. Returns NaN for an empty list.
private func median(_ values: [Double]) -> Double {
    if values.isEmpty {
        return Double.nan
    }
    let sorted = values.sorted()
    let n = sorted.count
    if n % 2 == 1 {
        return sorted[n / 2]
    }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}

/// Consecutive differences (numpy `np.diff`): `[v[1]-v[0], v[2]-v[1], ...]`.
private func diff(_ values: [Double]) -> [Double] {
    guard values.count > 1 else { return [] }
    var result: [Double] = []
    result.reserveCapacity(values.count - 1)
    for i in 1..<values.count {
        result.append(values[i] - values[i - 1])
    }
    return result
}

/// Index of the minimum value (numpy `np.argmin`): first index on ties.
private func argmin(_ values: [Double]) -> Int {
    var bestIndex = 0
    var bestValue = values[0]
    for i in 1..<values.count where values[i] < bestValue {
        bestValue = values[i]
        bestIndex = i
    }
    return bestIndex
}

// MARK: - InputPredictions

/// Bundle of segmentation/preprocessing rasters produced by the neural net.
/// Port of `InputPredictions`. Each field is one binary/grayscale mask.
final class InputPredictions {
    let original: GrayscaleImage
    let preprocessed: GrayscaleImage
    let notehead: GrayscaleImage
    let symbols: GrayscaleImage
    let staff: GrayscaleImage
    let stemsRest: GrayscaleImage
    let clefsKeys: GrayscaleImage

    init(
        original: GrayscaleImage,
        preprocessed: GrayscaleImage,
        notehead: GrayscaleImage,
        symbols: GrayscaleImage,
        staff: GrayscaleImage,
        clefsKeys: GrayscaleImage,
        stemsRest: GrayscaleImage
    ) {
        self.original = original
        self.preprocessed = preprocessed
        self.notehead = notehead
        self.symbols = symbols
        self.staff = staff
        self.stemsRest = stemsRest
        self.clefsKeys = clefsKeys
    }
}

// MARK: - SymbolOnStaff

/// Abstract base for any symbol placed relative to a staff. Port of `SymbolOnStaff`.
class SymbolOnStaff: DebugDrawable {
    /// Symbol centre (mutated by `transformCoordinates`).
    var center: CV.PointF

    init(center: CV.PointF) {
        self.center = center
    }

    /// Abstract clone. Subclasses return their concrete type (covariant override).
    func copy() -> SymbolOnStaff {
        fatalError("copy() is abstract and must be overridden")
    }

    /// Returns a copy whose centre is mapped through `transformation`.
    /// Port of `transform_coordinates`.
    func transformCoordinates(_ transformation: (CV.PointF) -> CV.PointF) -> SymbolOnStaff {
        let result = copy()
        result.center = transformation(center)
        return result
    }
}

// MARK: - Accidental

/// A sharp/flat/natural accidental at a staff position. Port of `Accidental`.
final class Accidental: SymbolOnStaff {
    let box: BoundingBox
    let position: Int

    init(box: BoundingBox, position: Int) {
        self.box = box
        self.position = position
        super.init(center: box.center)
    }

    override func copy() -> Accidental {
        Accidental(box: box, position: position)
    }
}

// MARK: - Rest

/// A rest symbol. Port of `Rest`. `hasDot` is reset by `copy` (as in Python).
final class Rest: SymbolOnStaff {
    let box: BoundingBox
    var hasDot = false

    init(box: BoundingBox) {
        self.box = box
        super.init(center: box.center)
    }

    override func copy() -> Rest {
        Rest(box: box)
    }
}

// MARK: - Enums

/// Stem orientation. Port of `StemDirection`.
enum StemDirection: Int {
    case up = 1
    case down = 2
}

/// Notehead fill type. Port of `NoteHeadType`, including its string form.
enum NoteHeadType: Int, CustomStringConvertible {
    case hollow = 1
    case solid = 2

    var description: String {
        switch self {
        case .hollow: return "O"
        case .solid: return "*"
        }
    }
}

/// Diatonic note-name lookup (module-level `note_names` in Python).
let noteNames = ["C", "D", "E", "F", "G", "A", "B"]

// MARK: - Note

/// A note head with optional stem, beams and flags. Port of `Note`.
///
/// Mutable fields (`hasDot`, `circleOfFifth`, `beams`, `flags`) are populated by
/// later detection passes and are NOT preserved by `copy` (matching Python).
final class Note: SymbolOnStaff {
    let box: BoundingEllipse
    let position: Int
    var hasDot = false
    let stem: RotatedBoundingBox?
    var circleOfFifth = 0
    let stemDirection: StemDirection?
    var beams: [RotatedBoundingBox] = []
    var flags: [RotatedBoundingBox] = []

    init(box: BoundingEllipse, position: Int, stem: RotatedBoundingBox?, stemDirection: StemDirection?) {
        self.box = box
        self.position = position
        self.stem = stem
        self.stemDirection = stemDirection
        super.init(center: box.center)
    }

    override func copy() -> Note {
        Note(box: box, position: position, stem: stem, stemDirection: stemDirection)
    }
}

// MARK: - BarLine

/// A bar line. Port of `BarLine`.
final class BarLine: SymbolOnStaff {
    let box: RotatedBoundingBox

    init(box: RotatedBoundingBox) {
        self.box = box
        super.init(center: box.center)
    }

    override func copy() -> BarLine {
        BarLine(box: box)
    }
}

// MARK: - Clef

/// A clef. Port of `Clef`.
final class Clef: SymbolOnStaff {
    let box: BoundingBox

    init(box: BoundingBox) {
        self.box = box
        super.init(center: box.center)
    }

    override func copy() -> Clef {
        Clef(box: box)
    }
}

// MARK: - StaffPoint

/// One vertical "slice" of a staff: the y-positions of its lines at a given x.
/// Port of `StaffPoint`.
final class StaffPoint {
    /// X coordinate of this slice.
    let x: Double
    /// Sorted y-coordinates of the staff lines at this x.
    let y: [Double]
    /// Local staff angle (degrees).
    let angle: Double
    /// Mean spacing between adjacent lines (`np.mean(np.diff(y))`).
    let averageUnitSize: Double

    init(x: Double, y: [Double], angle: Double) {
        // Invariant: a staff must consist of 5, 10, ... lines.
        assert(y.count % Constants.numberOfLinesOnAStaff == 0, "A staff must consist of 5, 10, ... lines")
        self.x = x
        self.y = y
        self.angle = angle
        self.averageUnitSize = mean(diff(y))
    }

    /// Combines two slices at the same x into one with all lines (grand staff).
    /// Port of `merge`.
    func merge(_ other: StaffPoint) -> StaffPoint {
        // Invariant: points must be at the same x position.
        assert(abs(x - other.x) <= 1e-3, "Can't merge points at different positions")
        var combined = y
        combined.append(contentsOf: other.y)
        let mergedAngle = (angle + other.angle) / 2
        return StaffPoint(x: x, y: combined.sorted(), angle: mergedAngle)
    }

    /// Computes a symbol's vertical position in half-unit steps relative to the
    /// staff lines at this slice. Port of `find_position_in_unit_sizes`.
    func findPositionInUnitSizes(_ box: AngledBoundingBox) -> Int {
        let center = box.center
        let idxOfClosestY = argmin(y.map { abs($0 - center.y) })
        let distance = y[idxOfClosestY] - center.y
        let distanceInUnitSizes = Int((2 * distance / averageUnitSize).rounded(.toNearestOrEven))
        let position = 2 * (y.count - idxOfClosestY) + distanceInUnitSizes - 1
        return position
    }

    /// Maps every `(x, y_i)` through `transformation`, averaging the resulting x.
    /// Port of `transform_coordinates`.
    func transformCoordinates(_ transformation: (CV.PointF) -> CV.PointF) -> StaffPoint {
        let xy = y.map { transformation(CV.PointF(x, $0)) }
        let averageX = mean(xy.map { $0.x })
        return StaffPoint(x: averageX, y: xy.map { $0.y }, angle: angle)
    }

    /// Represents this slice as a thin vertical `BoundingBox`. Port of `to_bounding_box`.
    func toBoundingBox() -> BoundingBox {
        BoundingBox(
            box: (x1: Int(x), y1: Int(y[0]), x2: Int(x), y2: Int(y[y.count - 1])),
            contours: [],
            debugId: -2
        )
    }
}

// MARK: - Staff

/// A single five-line staff: an ordered grid of `StaffPoint`s plus the symbols
/// assigned to it. Port of `Staff`. Identity (reference) equality is used by
/// `MultiStaff` for de-duplication, matching Python's default object equality.
final class Staff: DebugDrawable {
    var grid: [StaffPoint]
    let minX: Double
    let maxX: Double
    let minY: Double
    let maxY: Double
    let averageUnitSize: Double
    var symbols: [SymbolOnStaff] = []
    var isGrandstaff = false
    private let yTolerance: Double

    init(grid: [StaffPoint]) {
        self.grid = grid
        self.minX = grid[0].x
        self.maxX = grid[grid.count - 1].x
        self.minY = grid.map { $0.y.min()! }.min()!
        self.maxY = grid.map { $0.y.max()! }.max()!
        self.averageUnitSize = median(grid.map { $0.averageUnitSize })
        self.yTolerance = Double(Constants.maxNumberOfLedgerLines) * averageUnitSize
    }

    /// Whether `item`'s centre falls within this staff's vertical zone.
    /// Port of `is_on_staff_zone`.
    func isOnStaffZone(_ item: AngledBoundingBox) -> Bool {
        guard let point = getAt(item.center.x) else {
            return false
        }
        if item.center.y > point.y[point.y.count - 1] + yTolerance
            || item.center.y < point.y[0] - yTolerance {
            return false
        }
        return true
    }

    /// Merges two staffs sharing x-positions into a grand staff. Port of `merge`.
    func merge(_ other: Staff) -> Staff {
        var gridA: [Int: StaffPoint] = [:]
        for p in grid {
            gridA[Int(p.x.rounded(.toNearestOrEven))] = p
        }
        var gridB: [Int: StaffPoint] = [:]
        for p in other.grid {
            gridB[Int(p.x.rounded(.toNearestOrEven))] = p
        }
        let xPositions = Set(gridA.keys).intersection(gridB.keys)
        let mergedGrid = xPositions.sorted().map { gridA[$0]!.merge(gridB[$0]!) }
        let result = Staff(grid: mergedGrid)
        result.symbols.append(contentsOf: symbols)
        result.symbols.append(contentsOf: other.symbols)
        result.isGrandstaff = true
        return result
    }

    /// Port of `add_symbol`.
    func addSymbol(_ symbol: SymbolOnStaff) {
        symbols.append(symbol)
    }

    /// Returns the closest grid slice to `x`, or `nil` if none is within the
    /// staff position tolerance. Port of `get_at`.
    func getAt(_ x: Double) -> StaffPoint? {
        guard let closest = grid.min(by: { abs($0.x - x) < abs($1.x - x) }) else {
            return nil
        }
        if abs(closest.x - x) > Double(Constants.staffPositionTolerance) {
            return nil
        }
        return closest
    }

    /// Minimum vertical distance from `point` to any staff line at that x.
    /// Returns a large sentinel if `point` is off-staff. Port of `y_distance_to`.
    func yDistanceTo(_ point: CV.PointF) -> Double {
        guard let staffPoint = getAt(point.x) else {
            return 1e10 // Something large to mimic infinity.
        }
        return staffPoint.y.map { abs($0 - point.y) }.min()!
    }

    /// Port of `get_bar_lines`.
    func getBarLines() -> [BarLine] {
        symbols.compactMap { $0 as? BarLine }
    }

    /// Port of `get_clefs`.
    func getClefs() -> [Clef] {
        symbols.compactMap { $0 as? Clef }
    }

    /// Port of `get_notes`.
    func getNotes() -> [Note] {
        symbols.compactMap { $0 as? Note }
    }

    /// Returns a copy whose grid is extended to cover `[minX, maxX]`. Port of
    /// `extend_to_x_range`.
    func extendToXRange(_ minXValue: Int, _ maxXValue: Int) -> Staff {
        var newGrid = grid

        if minXValue >= 0 && Double(minXValue) < newGrid[0].x {
            newGrid.insert(StaffPoint(x: Double(minXValue), y: newGrid[0].y, angle: newGrid[0].angle), at: 0)
        }
        if maxXValue >= 0 && Double(maxXValue) > newGrid[newGrid.count - 1].x {
            newGrid.append(StaffPoint(x: Double(maxXValue), y: newGrid[newGrid.count - 1].y, angle: newGrid[newGrid.count - 1].angle))
        }

        return Staff(grid: newGrid)
    }

    /// Port of `get_number_of_notes`.
    func getNumberOfNotes() -> Int {
        symbols.reduce(0) { $0 + ($1 is Note ? 1 : 0) }
    }

    /// Port of `get_all_except_notes`.
    func getAllExceptNotes() -> [SymbolOnStaff] {
        symbols.filter { !($0 is Note) }
    }

    /// Port of `copy`.
    func copy() -> Staff {
        Staff(grid: grid)
    }

    /// Returns a copy with grid and symbols mapped through `transformation`.
    /// Port of `transform_coordinates`.
    func transformCoordinates(_ transformation: (CV.PointF) -> CV.PointF) -> Staff {
        let result = Staff(grid: grid.map { $0.transformCoordinates(transformation) })
        result.symbols = symbols.map { $0.transformCoordinates(transformation) }
        result.isGrandstaff = isGrandstaff
        return result
    }
}

// MARK: - MultiStaff

/// A grand staff or a staff with multiple voices. Port of `MultiStaff`.
final class MultiStaff: DebugDrawable {
    /// Constituent staffs, sorted by `minY`.
    let staffs: [Staff]
    /// Brace/bracket connections linking the staffs.
    let connections: [RotatedBoundingBox]

    init(staffs: [Staff], connections: [RotatedBoundingBox]) {
        self.staffs = staffs.sorted { $0.minY < $1.minY }
        self.connections = connections
    }

    /// Unions two multi-staffs, de-duplicating staffs by identity and
    /// connections by value (matching Python's `not in` semantics). Port of `merge`.
    func merge(_ other: MultiStaff) -> MultiStaff {
        var uniqueStaffs: [Staff] = []
        var uniqueConnections: [RotatedBoundingBox] = []
        for staff in staffs + other.staffs where !uniqueStaffs.contains(where: { $0 === staff }) {
            uniqueStaffs.append(staff)
        }
        for connection in connections + other.connections where !uniqueConnections.contains(connection) {
            uniqueConnections.append(connection)
        }
        return MultiStaff(staffs: uniqueStaffs, connections: uniqueConnections)
    }

    /// Scores how well a brace symbol ties an upper/lower staff into a grand
    /// staff. Port of `_score_brace_with_staff_pair`.
    ///
    /// Rules: the staff's left side must be close (within `5 * unit_size`) to the
    /// brace, and the brace must vertically overlap the staffs by at least 50%.
    /// The score is `y_overlap - x_distance` (higher is better), else 0.
    private func scoreBraceWithStaffPair(_ symbol: RotatedBoundingBox, _ upperStaff: Staff, _ lowerStaff: Staff) -> Double {
        let unitSize = median([upperStaff.averageUnitSize, lowerStaff.averageUnitSize])
        let xDistanceThreshold = Constants.grandstaffXDistanceThresholdFactor * unitSize
        let yOverlapThreshold = Constants.grandstaffYOverlapThresholdFactor * symbol.size.height

        let symbolMinX = symbol.center.x
        let staffMinX = min(upperStaff.minX, lowerStaff.minX)
        let xDistance = abs(staffMinX - symbolMinX)

        let symbolMinY = symbol.center.y - symbol.size.height / 2
        let symbolMaxY = symbol.center.y + symbol.size.height / 2
        let yOverlap = min(symbolMaxY, lowerStaff.maxY) - max(symbolMinY, upperStaff.minY)

        if xDistance < xDistanceThreshold && yOverlap > yOverlapThreshold && yOverlap > xDistance {
            return yOverlap - xDistance
        }
        return 0
    }

    /// A candidate staff pairing with its grand-staff score. Port of nested
    /// `MultiStaff.GrandStaffPair`.
    final class GrandStaffPair {
        let pairIndex: [Int]
        let score: Double

        init(pairIndex: [Int], score: Double) {
            self.pairIndex = pairIndex
            self.score = score
        }

        func getScore() -> Double { score }
        func getIndex() -> [Int] { pairIndex }
    }

    /// Whether the adjacent pair (i, i+1) is a "tight isolated pair" eligible to
    /// become a grand staff. In a system of 3+ staves we require the gap between
    /// the two candidate staves to be clearly smaller than the gap to their
    /// neighbouring staves — this is what separates a real grand staff (piano
    /// treble+bass) from evenly-spaced staves of distinct instruments (e.g. two
    /// oboes bridged by a section bracket). A two-staff system has no neighbours,
    /// so it is always eligible (preserves keyboard scores like the Bourrée).
    private func isTightIsolatedPair(_ i: Int) -> Bool {
        let internalGap = staffs[i + 1].minY - staffs[i].maxY
        var neighborGaps: [Double] = []
        if i - 1 >= 0 { neighborGaps.append(staffs[i].minY - staffs[i - 1].maxY) }
        if i + 2 < staffs.count { neighborGaps.append(staffs[i + 2].minY - staffs[i + 1].maxY) }
        // No neighbours (a lone 2-staff system) → always eligible, as before.
        guard let minNeighborGap = neighborGaps.min() else { return true }
        return internalGap < Constants.grandstaffTightPairGapRatio * minNeighborGap
    }

    /// Selects the best non-conflicting set of staff pairs to merge into grand
    /// staffs. Port of `_select_grandstaffs`.
    private func selectGrandstaffs(_ braceDot: [RotatedBoundingBox]) -> [GrandStaffPair] {
        var pairScores: [GrandStaffPair] = []
        // Step 1: try each adjacent staff pair (0,1), (1,2), ...
        if staffs.count >= 2 {
            for i in 0..<(staffs.count - 1) {
                // Step 1a: skip pairs that aren't a tight, isolated pair within a
                // larger system — prevents over-merging separate instruments
                // (e.g. two same-clef oboe staves) into a spurious grand staff.
                if !isTightIsolatedPair(i) { continue }
                // Step 2: best brace match for this pair.
                // NOTE: Python's `max(generator)` raises on an empty `brace_dot`;
                // here an empty brace list yields no score (0) so no pair is
                // selected, avoiding a crash on iOS.
                let bestScore = braceDot
                    .map { scoreBraceWithStaffPair($0, staffs[i], staffs[i + 1]) }
                    .max() ?? 0
                if bestScore > 0 {
                    pairScores.append(GrandStaffPair(pairIndex: [i, i + 1], score: bestScore))
                }
            }
        }

        // Step 3: greedily keep highest-scoring pairs without reusing a staff.
        var result: [GrandStaffPair] = []
        var usedStaffIndex = Set<Int>()
        for pair in pairScores.sorted(by: { $0.getScore() > $1.getScore() }) {
            if pair.getIndex().contains(where: { usedStaffIndex.contains($0) }) {
                continue
            }
            result.append(pair)
            usedStaffIndex.formUnion(pair.getIndex())
        }
        return result
    }

    /// Merges the staffs flagged by `pairs`, leaving the rest untouched.
    /// Port of `_merge_selected_pairs`.
    private func mergeSelectedPairs(_ pairs: [GrandStaffPair]) -> [Staff] {
        var result: [Staff] = []
        var i = 0
        while i < staffs.count {
            if pairs.contains(where: { $0.getIndex().contains(i) }) {
                result.append(staffs[i].merge(staffs[i + 1]))
                i += 2
                continue
            }
            result.append(staffs[i])
            i += 1
        }
        return result
    }

    /// Builds grand staffs from this multi-staff using detected brace symbols.
    /// Port of `create_grandstaffs`.
    func createGrandstaffs(_ braceDot: [RotatedBoundingBox]) -> MultiStaff {
        if staffs.count < 2 {
            return self
        }
        let pairs = selectGrandstaffs(braceDot)
        if pairs.isEmpty {
            return self
        }
        let mergedStaffs = mergeSelectedPairs(pairs)
        return MultiStaff(staffs: mergedStaffs, connections: connections)
    }

    /// Splits this multi-staff into one single-staff `MultiStaff` per staff.
    /// Port of `break_apart`.
    func breakApart() -> [MultiStaff] {
        staffs.map { MultiStaff(staffs: [$0], connections: []) }
    }
}
