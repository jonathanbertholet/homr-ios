import Foundation

/// Brace / bracket / grand-staff detection — port of `homr/brace_dot_detection.py`.
///
/// Isolates tall vertical connectors (braces, brackets, dots) from the symbol
/// mask, then uses them to decide which neighbouring staffs belong together as a
/// `MultiStaff` (multiple voices) or are fused into a grand staff.

/// Pre-processes the symbol mask to isolate tall vertical connectors.
///
/// Port of `prepare_brace_dot_image`. Steps:
///   1. `cv2.subtract(symbols, staff)` removes the staff lines from the symbols.
///   2. Erode with an elliptical SE of `cv2.getStructuringElement(MORPH_ELLIPSE, (1, 5))`
///      — OpenCV's ksize is `(width, height)`, so width = 1, height = 5. This
///      kills thin horizontal remnants.
///   3. Dilate with `MORPH_ELLIPSE, (5, 35)` (width = 5, height = 35) to regrow
///      and connect the surviving tall elements.
func prepareBraceDotImage(symbols: GrayscaleImage, staff: GrayscaleImage) -> GrayscaleImage {
    // Remove the staff lines from the symbol mask.
    let braceDot = CV.subtract(symbols, staff)

    // Erode away thin horizontal remnants (cv2 ksize = (width: 1, height: 5)).
    let erodeKernel = CV.getStructuringElement(.ellipse, (width: 1, height: 5))
    let out = CV.erode(braceDot, erodeKernel)

    // Regrow / connect the tall elements (cv2 ksize = (width: 5, height: 35)).
    let dilateKernel = CV.getStructuringElement(.ellipse, (width: 5, height: 35))
    return CV.dilate(out, dilateKernel)
}

/// Keeps only the brace/dot candidates that are tall enough to be connectors.
///
/// Port of `_filter_for_tall_elements`. Two-stage filter: a rough pass using the
/// first staff's unit size to cut the data down, then a precise pass using the
/// unit size of the closest staff (so page warping is accounted for). Index
/// mapping: `size[1]` is height, `size[0]` is width.
private func filterForTallElements(
    _ braceDot: [RotatedBoundingBox],
    _ staffs: [Staff]
) -> [RotatedBoundingBox] {
    let roughUnitSize = staffs[0].averageUnitSize
    let symbolsLargerThanRoughEstimate = braceDot.filter { symbol in
        symbol.size.height > Constants.minHeightForBraceRough(roughUnitSize)
            && symbol.size.width < Constants.maxWidthForBraceRough(roughUnitSize)
    }

    var result: [RotatedBoundingBox] = []
    for symbol in symbolsLargerThanRoughEstimate {
        // min(staffs, key=staff.y_distance_to(symbol.center)); Swift's min(by:)
        // returns the first minimal element, matching Python's min().
        guard let closestStaff = staffs.min(by: {
            $0.yDistanceTo(symbol.center) < $1.yDistanceTo(symbol.center)
        }) else {
            continue
        }
        let unitSize = closestStaff.averageUnitSize
        if symbol.size.height > Constants.minHeightForBrace(unitSize) {
            result.append(symbol)
        }
    }
    return result
}

/// Brace/dot symbols that bridge the two staffs through their bar lines.
///
/// Port of `_get_connections_between_staffs_at_bar_lines`. A symbol connects when
/// (thickened by 30px) it overlaps at least one bar line on each staff. `line.box`
/// is a `RotatedBoundingBox` (an `AnyPolygon`), passed straight to `isOverlapping`.
private func getConnectionsBetweenStaffsAtBarLines(
    _ staff1: Staff,
    _ staff2: Staff,
    _ braceDot: [RotatedBoundingBox]
) -> [RotatedBoundingBox] {
    let barLines1 = staff1.getBarLines()
    let barLines2 = staff2.getBarLines()
    var result: [RotatedBoundingBox] = []
    for symbol in braceDot {
        let symbolThicker = symbol.makeBoxThicker(30)
        let firstOverlappingStaff1 = barLines1.filter { symbolThicker.isOverlapping($0.box) }
        let firstOverlappingStaff2 = barLines2.filter { symbolThicker.isOverlapping($0.box) }
        if firstOverlappingStaff1.count >= 1 && firstOverlappingStaff2.count >= 1 {
            result.append(symbol)
        }
    }
    return result
}

/// Brace/dot symbols that bridge the two staffs through their clefs.
///
/// Port of `_get_connections_between_staffs_at_clefs`. The symbol is thickened by
/// `tolerance_for_staff_at_any_point` (which is 0) before overlap testing.
/// `clef.box` is a `BoundingBox` (an `AnyPolygon`).
private func getConnectionsBetweenStaffsAtClefs(
    _ staff1: Staff,
    _ staff2: Staff,
    _ braceDot: [RotatedBoundingBox]
) -> [RotatedBoundingBox] {
    let clefs1 = staff1.getClefs()
    let clefs2 = staff2.getClefs()
    var result: [RotatedBoundingBox] = []
    for symbol in braceDot {
        let symbolThicker = symbol.makeBoxThicker(
            Constants.toleranceForStaffAtAnyPoint(staff1.averageUnitSize)
        )
        let firstOverlappingStaff1 = clefs1.filter { symbolThicker.isOverlapping($0.box) }
        let firstOverlappingStaff2 = clefs2.filter { symbolThicker.isOverlapping($0.box) }
        if firstOverlappingStaff1.count >= 1 && firstOverlappingStaff2.count >= 1 {
            result.append(symbol)
        }
    }
    return result
}

/// Brace/dot symbols that bridge the two staffs through their staff lines.
///
/// Port of `_get_connections_between_staffs_at_lines`. The symbol is thickened by
/// `tolerance_for_touching_clefs` and must overlap the thin line-box of each
/// staff at the symbol's centre x. Staffs with no grid slice at that x are skipped.
private func getConnectionsBetweenStaffsAtLines(
    _ staff1: Staff,
    _ staff2: Staff,
    _ braceDot: [RotatedBoundingBox]
) -> [RotatedBoundingBox] {
    var result: [RotatedBoundingBox] = []
    for symbol in braceDot {
        let symbolThicker = symbol.makeBoxThicker(
            Constants.toleranceForTouchingClefs(staff1.averageUnitSize)
        )
        guard let point1 = staff1.getAt(symbol.center.x),
              let point2 = staff2.getAt(symbol.center.x) else {
            continue
        }
        if symbolThicker.isOverlapping(point1.toBoundingBox())
            && symbolThicker.isOverlapping(point2.toBoundingBox()) {
            result.append(symbol)
        }
    }
    return result
}

/// All brace/dot connectors between two staffs (bar lines + clefs + lines).
/// Port of `_get_connections_between_staffs`.
private func getConnectionsBetweenStaffs(
    _ staff1: Staff,
    _ staff2: Staff,
    _ braceDot: [RotatedBoundingBox]
) -> [RotatedBoundingBox] {
    var result: [RotatedBoundingBox] = []
    result.append(contentsOf: getConnectionsBetweenStaffsAtBarLines(staff1, staff2, braceDot))
    result.append(contentsOf: getConnectionsBetweenStaffsAtClefs(staff1, staff2, braceDot))
    result.append(contentsOf: getConnectionsBetweenStaffsAtLines(staff1, staff2, braceDot))
    return result
}

/// Merges any two `MultiStaff`s that share a staff into a single `MultiStaff`.
///
/// Port of `_merge_multi_staff_if_they_share_a_staff`. Python relies on `Staff`'s
/// identity equality inside a `set` intersection; `Staff` is a `final class` with
/// no value `Equatable`, so we reproduce that with a `Set<ObjectIdentifier>` of
/// the staff instances — two multi-staffs share a staff iff they contain the
/// exact same `Staff` object.
private func mergeMultiStaffIfTheyShareAStaff(_ staffs: [MultiStaff]) -> [MultiStaff] {
    var result: [MultiStaff] = []
    for staff in staffs {
        var anyMerged = false
        for existing in result {
            let staffIds = Set(staff.staffs.map(ObjectIdentifier.init))
            let existingIds = Set(existing.staffs.map(ObjectIdentifier.init))
            if !staffIds.intersection(existingIds).isEmpty {
                if let index = result.firstIndex(where: { $0 === existing }) {
                    result.remove(at: index)
                }
                result.append(existing.merge(staff))
                anyMerged = true
                break
            }
        }
        if !anyMerged {
            result.append(staff)
        }
    }
    return result
}

/// Turns each `MultiStaff` into a grand staff where a brace warrants it.
/// Port of `_create_grandstaffs`.
private func createGrandstaffs(
    _ staffs: [MultiStaff],
    _ braceDot: [RotatedBoundingBox]
) -> [MultiStaff] {
    if staffs.isEmpty {
        return staffs
    }
    return staffs.map { $0.createGrandstaffs(braceDot) }
}

/// Connects staffs into braces / brackets / grand staffs.
///
/// Port of `find_braces_brackets_and_grand_staff_lines` (the Python `Debug`
/// parameter is dropped — all drawing is debug-only on iOS). For each staff it
/// inspects the previous and next staff: if enough connectors bridge them, a
/// two-staff `MultiStaff` is recorded; staffs with no connected neighbour become
/// a single-staff `MultiStaff`. The overlapping results are then de-duplicated by
/// shared staff and promoted to grand staffs where appropriate.
func findBracesBracketsAndGrandStaffLines(
    staffs: [Staff],
    braceDot: [RotatedBoundingBox]
) -> [MultiStaff] {
    let filteredBraceDot = filterForTallElements(braceDot, staffs)
    var result: [MultiStaff] = []
    for (i, staff) in staffs.enumerated() {
        var neighbors: [Staff] = []
        if i > 0 {
            neighbors.append(staffs[i - 1])
        }
        if i < staffs.count - 1 {
            neighbors.append(staffs[i + 1])
        }
        var anyConnectedNeighbor = false
        for neighbor in neighbors {
            let connections = getConnectionsBetweenStaffs(staff, neighbor, filteredBraceDot)
            if connections.count >= Constants.minimumConnectionsToFormCombinedStaff {
                result.append(MultiStaff(staffs: [staff, neighbor], connections: connections))
                anyConnectedNeighbor = true
            }
        }
        if !anyConnectedNeighbor {
            result.append(MultiStaff(staffs: [staff], connections: []))
        }
    }

    return createGrandstaffs(mergeMultiStaffIfTheyShareAStaff(result), filteredBraceDot)
}
