import Foundation

/// Tuning constants and small unit-size-relative helpers for symbol detection.
///
/// Direct port of `homr/constants.py`. The module-level Python constants become
/// `static let`s and the small `def`-functions (which derive a threshold from a
/// staff's `unit_size`) become `static func`s, all namespaced under `Constants`.
enum Constants {

    /// Number of lines that make up a single staff (Python `number_of_lines_on_a_staff`).
    static let numberOfLinesOnAStaff = 5

    /// Maximum count of ledger lines considered above/below a staff.
    static let maxNumberOfLedgerLines = 4

    /// Vertical slack used when matching staff-line segments.
    static func toleranceForStaffLineDetection(_ unitSize: Double) -> Double {
        unitSize / 3
    }

    /// Largest horizontal gap that two collinear line fragments may have and
    /// still be considered one line.
    static func maxLineGapSize(_ unitSize: Double) -> Double {
        5 * unitSize
    }

    /// Threshold below which a line is "short".
    static func isShortLine(_ unitSize: Double) -> Double {
        unitSize / 5
    }

    /// Threshold below which a connected line is "short".
    static func isShortConnectedLine(_ unitSize: Double) -> Double {
        2 * unitSize
    }

    /// Rough minimum height for a brace candidate.
    static func minHeightForBraceRough(_ unitSize: Double) -> Double {
        2 * unitSize
    }

    /// Rough maximum width for a brace candidate.
    static func maxWidthForBraceRough(_ unitSize: Double) -> Double {
        3 * unitSize
    }

    /// Final minimum height for a brace.
    static func minHeightForBrace(_ unitSize: Double) -> Double {
        4 * unitSize
    }

    /// Pixel tolerance used to decide whether two clefs touch.
    /// Python: `int(round(unit_size * 2))` — banker's rounding then truncation.
    static func toleranceForTouchingClefs(_ unitSize: Double) -> Int {
        Int((unitSize * 2).rounded(.toNearestOrEven))
    }

    /// Tolerance for accepting a staff at an arbitrary point (always 0 in Python).
    static func toleranceForStaffAtAnyPoint(_ unitSize: Double) -> Int {
        0
    }

    /// Horizontal grouping tolerance for notes.
    static func toleranceNoteGrouping(_ unitSize: Double) -> Double {
        1 * unitSize
    }

    /// Maximum width for something to be treated as a bar line.
    static func barLineMaxWidth(_ unitSize: Double) -> Double {
        2 * unitSize
    }

    /// Minimum height for something to be treated as a bar line.
    static func barLineMinHeight(_ unitSize: Double) -> Double {
        3 * unitSize
    }

    /// Size threshold above which a black spot is removed as noise.
    static func blackSpotRemovalThreshold(_ unitSize: Double) -> Double {
        2 * unitSize
    }

    /// X tolerance (pixels) when stitching staff-line segments.
    static let staffLineSegmentXTolerance = 10

    /// Minimum number of cross-staff connections required to combine staffs.
    ///
    /// We don't have to worry about mis-detections, because if not all staffs
    /// group the same way then we break the staffs up again.
    static let minimumConnectionsToFormCombinedStaff = 1

    /// Duration value assigned to a quarter note.
    static let durationOfQuarter = 16

    /// Connected-component size below which a blob is considered image noise.
    static let imageNoiseLimit = 50

    /// Maximum distance (pixels) for a symbol to still belong to a staff.
    static let staffPositionTolerance = 50

    /// Maximum angle (degrees) for two lines to be considered parallel.
    static let maxAngleForLinesToBeParallel = 10

    /// Expected notehead width/height ratio.
    static let noteheadSizeRatio = 1.285714

    /// Multiplier on unit size for the grand-staff brace x-distance threshold.
    /// Kept as `Double` because it is always multiplied with a `Double` unit size.
    static let grandstaffXDistanceThresholdFactor = 5.0

    /// Fraction of a brace's height that must overlap the staffs for a grand staff.
    static let grandstaffYOverlapThresholdFactor = 0.5

    /// Inside a system of 3+ staves, only merge an adjacent pair into a grand
    /// staff when the gap *between* the two candidate staves is at most this
    /// fraction of the gap to their neighbouring staves. A real grand staff
    /// (piano treble+bass) is a tight, isolated pair; evenly-spaced staves of
    /// separate instruments (e.g. two oboes under a bracket) are NOT. Two-staff
    /// systems have no neighbours and are always allowed to merge (unchanged).
    static let grandstaffTightPairGapRatio = 0.7
}
