import Foundation

// MARK: - StaffRegions
//
// Port of `homr/staff_regions.py`.
//
// Given every detected staff (flattened out of the `MultiStaff` voices), this
// records each staff's vertical extent as a `(minY, maxY)` pair and answers two
// neighbour queries used when carving out the region around a staff for
// inference: where does the closest staff ABOVE a given y end, and where does
// the closest staff BELOW a given y start. The crop in `StaffParsing` uses these
// so a staff's region never bleeds into its neighbours.

/// Vertical extent of a single staff: `(minY, maxY)`.
/// Mirrors Python's `get_center_min_max_y(staff) -> (staff.min_y, staff.max_y)`.
private struct StaffCenter {
    let minY: Double
    let maxY: Double
}

final class StaffRegions {
    /// `(minY, maxY)` for every staff across every multi-staff voice, in the same
    /// flattened order Python produces with its nested comprehension.
    private let centers: [StaffCenter]

    /// Port of `__init__`: flatten all staffs out of all multi-staffs and record
    /// each one's vertical extent.
    init(_ staffs: [MultiStaff]) {
        var collected: [StaffCenter] = []
        for ms in staffs {
            for s in ms.staffs {
                collected.append(StaffCenter(minY: s.minY, maxY: s.maxY))
            }
        }
        self.centers = collected
    }

    /// Port of `get_start_of_closest_staff_above`.
    ///
    /// Among staffs whose top (`minY`, stored as `c[0]`) is above `y`, return the
    /// largest bottom (`maxY`, `c[1]`). If none qualifies, return 0.
    func getStartOfClosestStaffAbove(_ y: Double) -> Double {
        let staffsAbove = centers.filter { $0.minY < y }.map { $0.maxY }
        if staffsAbove.isEmpty {
            return 0
        }
        return staffsAbove.max()!
    }

    /// Port of `get_start_of_closest_staff_below`.
    ///
    /// Among staffs whose bottom (`maxY`, `c[1]`) is below `y`, return the
    /// smallest top (`minY`, `c[0]`). If none qualifies, return 1e12 (a value
    /// larger than the height of any reasonable image), matching Python.
    func getStartOfClosestStaffBelow(_ y: Double) -> Double {
        let staffsBelow = centers.filter { $0.maxY > y }.map { $0.minY }
        if staffsBelow.isEmpty {
            return 1e12
        }
        return staffsBelow.min()!
    }
}
