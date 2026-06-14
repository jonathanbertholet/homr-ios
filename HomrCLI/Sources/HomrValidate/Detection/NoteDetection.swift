import Foundation

/// Note detection — port of `homr/note_detection.py`.
///
/// Combines detected noteheads (fitted ellipses) with detected stems
/// (rotated boxes), splits clumps of touching noteheads back into individual
/// heads, and finally assigns the resulting notes to the staffs they sit on.

// MARK: - Rect helper

/// The `cvt.Rect` used throughout this Python module is the 4-tuple
/// `(x1, y1, x2, y2)` (two opposite corners), NOT OpenCV's origin+size `Rect`.
/// We model it as a labelled tuple to keep the porting line-for-line.
private typealias NoteRect = (x1: Int, y1: Int, x2: Int, y2: Int)

/// Python floor division (`//`). Swift integer `/` truncates toward zero, so we
/// correct the quotient for the mixed-sign case to match Python exactly.
private func floorDiv(_ a: Int, _ b: Int) -> Int {
    let q = a / b
    let r = a % b
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1
    }
    return q
}

// MARK: - NoteheadWithStem

/// A notehead paired with the (optional) stem that belongs to it, plus the stem
/// direction. Port of `NoteheadWithStem`. The Python class also implements a
/// debug-only `draw_onto_image`, which is dropped on iOS (inference never draws).
final class NoteheadWithStem {
    /// The fitted-ellipse hull of the notehead.
    let notehead: BoundingEllipse
    /// The stem box, or `nil` when no stem overlaps the notehead.
    let stem: RotatedBoundingBox?
    /// Stem orientation relative to the head, or `nil` when there is no stem.
    let stemDirection: StemDirection?

    init(notehead: BoundingEllipse, stem: RotatedBoundingBox?, stemDirection: StemDirection? = nil) {
        self.notehead = notehead
        self.stem = stem
        self.stemDirection = stemDirection
    }
}

// MARK: - Bounding-box geometry helpers

/// Tightens a candidate box vertically to the actual painted notehead pixels.
///
/// Port of `adjust_bbox`. Looks at the sub-region `noteheads[y1:y2, x1:x2]`,
/// finds the rows that contain any positive pixel (`np.where(region > 0)`), and
/// returns a box whose top/bottom hug those rows (with the same ±1 padding as
/// Python). If the region is empty the box is returned unchanged — Python's
/// comment notes such a box will later be eliminated by its zero height.
private func adjustBbox(_ bbox: NoteRect, _ noteheads: GrayscaleImage) -> NoteRect {
    let width = noteheads.width
    let height = noteheads.height

    // Track the first and last region-relative rows that contain a lit pixel,
    // mirroring `np.min(ys)` / `np.max(ys)` over the sliced region.
    var minRel: Int? = nil
    var maxRel: Int? = nil

    if bbox.y1 < bbox.y2 && bbox.x1 < bbox.x2 {
        for row in bbox.y1..<bbox.y2 where row >= 0 && row < height {
            var hasPositive = false
            for col in bbox.x1..<bbox.x2 where col >= 0 && col < width {
                if noteheads.pixels[row * width + col] > 0 {
                    hasPositive = true
                    break
                }
            }
            if hasPositive {
                let rel = row - bbox.y1
                if minRel == nil { minRel = rel }
                maxRel = rel
            }
        }
    }

    guard let topRel = minRel, let bottomRel = maxRel else {
        // len(ys) == 0: invalid note, returned as-is.
        return bbox
    }

    // top = np.min(ys) + bbox[1] - 1 ; bottom = np.max(ys) + bbox[1] + 1.
    let top = topRel + bbox.y1 - 1
    let bottom = bottomRel + bbox.y1 + 1
    return (x1: bbox.x1, y1: top, x2: bbox.x2, y2: bottom)
}

/// Centre point of a `(x1, y1, x2, y2)` box. Port of `get_center`.
/// Uses banker's rounding (`int(round(...))`) on each averaged coordinate.
private func getCenter(_ bbox: NoteRect) -> (x: Int, y: Int) {
    let cenY = Int((Double(bbox.y1 + bbox.y2) / 2).rounded(.toNearestOrEven))
    let cenX = Int((Double(bbox.x1 + bbox.x2) / 2).rounded(.toNearestOrEven))
    return (x: cenX, y: cenY)
}

/// Recursively splits a candidate box into individual notehead-sized boxes.
///
/// Port of `check_bbox_size` (faithful, including the double-recursion). It works
/// in two phases:
///   1. If the box is roughly twice a notehead wide, split it left/right at the
///      centre, re-tighten each half vertically (`adjust_bbox`), and recurse on
///      both halves. The collected children are then re-checked once more (the
///      Python `if len(new_bbox) > 0` re-run), which is what splits stacked
///      chords sitting side by side.
///   2. Otherwise split the box vertically into `round(h / unit_size)` equal
///      slices, one per stacked note.
private func checkBboxSize(_ bbox: NoteRect, _ noteheads: GrayscaleImage, _ unitSize: Double) -> [NoteRect] {
    let w = bbox.x2 - bbox.x1
    let h = bbox.y2 - bbox.y1
    let cenX = getCenter(bbox).x
    let noteW = Constants.noteheadSizeRatio * unitSize
    let noteH = unitSize

    var newBbox: [NoteRect] = []

    // Closer to "two noteheads wide" than "one notehead wide" => split L/R.
    if abs(Double(w) - noteW) > abs(Double(w) - noteW * 2) {
        var leftBox: NoteRect = (x1: bbox.x1, y1: bbox.y1, x2: cenX, y2: bbox.y2)
        var rightBox: NoteRect = (x1: cenX, y1: bbox.y1, x2: bbox.x2, y2: bbox.y2)

        // Upper/lower bounds could have changed after the horizontal split.
        leftBox = adjustBbox(leftBox, noteheads)
        rightBox = adjustBbox(rightBox, noteheads)

        // adjust_bbox never returns None in Python, so both halves recurse.
        newBbox.append(contentsOf: checkBboxSize(leftBox, noteheads, unitSize))
        newBbox.append(contentsOf: checkBboxSize(rightBox, noteheads, unitSize))
    }

    // Check height.
    if newBbox.count > 0 {
        // Re-run the size check on every child produced by the L/R split.
        var tmpNew: [NoteRect] = []
        for box in newBbox {
            tmpNew.append(contentsOf: checkBboxSize(box, noteheads, unitSize))
        }
        newBbox = tmpNew
    } else {
        // num_notes = int(round(h / note_h)).
        let numNotes = Int((Double(h) / noteH).rounded(.toNearestOrEven))
        if numNotes > 0 {
            // sub_h = h // num_notes (Python floor division).
            let subH = floorDiv(h, numNotes)
            for i in 0..<numNotes {
                let subBox: NoteRect = (
                    x1: bbox.x1,
                    y1: bbox.y1 + i * subH,
                    x2: bbox.x2,
                    y2: bbox.y1 + (i + 1) * subH
                )
                newBbox.append(subBox)
            }
        }
    }

    return newBbox
}

/// Splits a clump of touching noteheads back into one box per head.
///
/// Port of `split_clumps_of_noteheads`. Builds the integer `(x1, y1, x2, y2)`
/// box from the ellipse's axis-aligned extents, runs `check_bbox_size`, and — if
/// more than one box comes back — rebuilds one `NoteheadWithStem` per box. The
/// rebuilt heads share the original contours, debug id, stem and stem direction
/// (the Python code propagates these through its loop variable shadowing).
private func splitClumpsOfNoteheads(
    _ notehead: NoteheadWithStem,
    _ noteheads: GrayscaleImage,
    _ staff: Staff
) -> [NoteheadWithStem] {
    let bbox: NoteRect = (
        x1: Int(notehead.notehead.topLeft.x),
        y1: Int(notehead.notehead.topLeft.y),
        x2: Int(notehead.notehead.bottomRight.x),
        y2: Int(notehead.notehead.bottomRight.y)
    )
    let splitBoxes = checkBboxSize(bbox, noteheads, staff.averageUnitSize)
    if splitBoxes.count <= 1 {
        return [notehead]
    }

    var result: [NoteheadWithStem] = []
    // `current` mirrors Python's reassigned `notehead` loop variable; contours,
    // debug id, stem and direction are carried forward unchanged each step.
    var current = notehead
    for box in splitBoxes {
        let center = getCenter(box)
        let size = CV.Size(width: Double(box.x2 - box.x1), height: Double(box.y2 - box.y1))
        let rebuilt = NoteheadWithStem(
            notehead: BoundingEllipse(
                box: CV.RotatedRect(
                    center: CV.PointF(Double(center.x), Double(center.y)),
                    size: size,
                    angle: 0
                ),
                contours: current.notehead.contours,
                debugId: current.notehead.debugId
            ),
            stem: current.stem,
            stemDirection: current.stemDirection
        )
        result.append(rebuilt)
        current = rebuilt
    }
    return result
}

// MARK: - Combining noteheads with stems

/// Pairs each notehead with the first stem that overlaps it.
///
/// Port of `combine_noteheads_with_stems`. Noteheads are processed top-to-bottom
/// (sorted by the rotated-rect centre y, `notehead.box[0][1]`). A notehead is
/// "thickened" by 15px before testing overlap so a slightly detached stem still
/// matches. The stem direction is `.up` when the stem centre is above the head,
/// otherwise `.down`. Heads with no overlapping stem get a stem-less entry.
///
/// `used_stems` is a `Set` of matched stems (Python tracks it but never reads it
/// back); stems are `RotatedBoundingBox`, which is `Hashable` by its box value.
func combineNoteheadsWithStems(
    noteheads: [BoundingEllipse],
    stems: [RotatedBoundingBox]
) -> [NoteheadWithStem] {
    var result: [NoteheadWithStem] = []

    // Stable sort by rotated-rect centre y (Python `sorted` is stable; Swift's
    // `sorted(by:)` is not, so tie-break on the original index to match).
    let sortedNoteheads = noteheads.enumerated()
        .sorted { lhs, rhs in
            if lhs.element.box.center.y != rhs.element.box.center.y {
                return lhs.element.box.center.y < rhs.element.box.center.y
            }
            return lhs.offset < rhs.offset
        }
        .map { $0.element }

    var usedStems = Set<RotatedBoundingBox>()
    for notehead in sortedNoteheads {
        let thickenedNotehead = notehead.makeBoxThicker(15)
        var foundStem = false
        for stem in stems {
            if stem.isOverlapping(thickenedNotehead) {
                let isStemAbove = stem.center.y < notehead.center.y
                let direction: StemDirection = isStemAbove ? .up : .down
                result.append(NoteheadWithStem(notehead: notehead, stem: stem, stemDirection: direction))
                usedStems.insert(stem)
                foundStem = true
                break
            }
        }
        if !foundStem {
            result.append(NoteheadWithStem(notehead: notehead, stem: nil, stemDirection: nil))
        }
    }
    return result
}

// MARK: - Assigning notes to staffs

/// Assigns notes to the staffs they sit on, splitting clumps as needed.
///
/// Port of `add_notes_to_staffs`. For every staff and every notehead chunk: skip
/// chunks outside the staff zone or off-grid, drop chunks that are too small,
/// then split the chunk into individual heads. Each head that passes the
/// per-staff size gate becomes a `Note`, which is appended to the returned list
/// AND added to the staff in place (`staff.add_symbol`).
///
/// - Note: The Python `symbols` raster argument is unused inside the function;
///   it is kept here only to preserve the original signature. `notehead_pred` is
///   the raster actually consumed by clump splitting.
@discardableResult
func addNotesToStaffs(
    staffs: [Staff],
    noteheads: [NoteheadWithStem],
    symbols: GrayscaleImage,
    noteheadPred: GrayscaleImage
) -> [Note] {
    var result: [Note] = []
    for staff in staffs {
        for noteheadChunk in noteheads {
            if !staff.isOnStaffZone(noteheadChunk.notehead) {
                continue
            }
            let center = noteheadChunk.notehead.center
            // First gate uses the chunk itself against its closest grid slice.
            guard let chunkPoint = staff.getAt(center.x) else {
                continue
            }
            if noteheadChunk.notehead.size.width < 0.5 * chunkPoint.averageUnitSize
                || noteheadChunk.notehead.size.height < 0.5 * chunkPoint.averageUnitSize {
                continue
            }
            for notehead in splitClumpsOfNoteheads(noteheadChunk, noteheadPred, staff) {
                // Re-fetch the grid slice at the SAME chunk centre x (Python
                // reuses `center[0]` here, not the split head's centre).
                guard let point = staff.getAt(center.x) else {
                    continue
                }
                if notehead.notehead.size.width < 0.5 * point.averageUnitSize
                    || notehead.notehead.size.width > 3 * point.averageUnitSize
                    || notehead.notehead.size.height < 0.5 * point.averageUnitSize
                    || notehead.notehead.size.height > 2 * point.averageUnitSize {
                    continue
                }
                let position = point.findPositionInUnitSizes(notehead.notehead)
                let note = Note(
                    box: notehead.notehead,
                    position: position,
                    stem: notehead.stem,
                    stemDirection: notehead.stemDirection
                )
                result.append(note)
                staff.addSymbol(note)
            }
        }
    }

    // Faithful port of the diagnostic tally (`eprint`). No effect on the result.
    var numberOfNotes = 0
    for staff in staffs {
        numberOfNotes += staff.getNotes().count
    }
    print("Found", numberOfNotes, "notes during segmentation")
    return result
}
