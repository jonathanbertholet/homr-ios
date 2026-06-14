import Foundation

// MARK: - find_peaks (scipy-free 1-D peak finder)
//
// Direct port of `homr/find_peaks.py`. The Python module already avoids scipy
// and implements the local-maxima / plateau / prominence / distance logic by
// hand, so this is a faithful line-by-line re-implementation.
//
// The only adaptation is the `properties` dictionary: Python returns an empty
// `dict` purely "for compatibility"; here it is an empty `[String: Any]`.

/// Finds peaks in a 1-D signal without scipy. Port of `find_peaks`.
///
/// - Parameters:
///   - x: 1-D signal.
///   - height: optional minimum peak height (`x[peak] >= height`).
///   - distance: optional minimum spacing (in samples) between kept peaks;
///     when peaks are too close the higher ones win.
///   - prominence: optional minimum topographic prominence.
/// - Returns: the peak indices and an (always empty) properties dictionary,
///   matching the Python return shape `tuple[NDArray, dict]`.
func findPeaks(
    _ x: [Double],
    height: Double? = nil,
    distance: Double? = nil,
    prominence: Double? = nil
) -> (peaks: [Int], properties: [String: Any]) {
    // Python: `if len(x) < 3: return np.array([]), {}`.
    if x.count < 3 {
        return ([], [:])
    }

    // ---------------------------------------------------------------
    // Find local maxima, handling flat regions (plateaus). This mirrors the
    // Python `while i < len(x) - 1` walk exactly, including the plateau
    // midpoint selection `(i + j) // 2` (floor division).
    // ---------------------------------------------------------------
    var peaksList: [Int] = []
    let n = x.count
    var i = 1
    while i < n - 1 {
        if x[i] > x[i - 1] {
            // Rising edge: find the end of any plateau starting at i.
            var j = i
            while j < n - 1 && x[j] == x[j + 1] {
                j += 1
            }
            // The plateau is a peak only if it then descends.
            if j < n - 1 && x[j] > x[j + 1] {
                let peakIdx = (i + j) / 2 // floor division (i, j >= 0)
                peaksList.append(peakIdx)
                i = j + 1
            } else {
                i = j + 1
            }
        } else if x[i] == x[i - 1] {
            // Flat region entered from the left at equal height.
            var j = i
            while j < n - 1 && x[j] == x[j + 1] {
                j += 1
            }
            // Peak if it is higher than the left side (implied) and descends.
            if j < n - 1 && x[j] > x[j + 1] {
                let peakIdx = (i + j) / 2
                peaksList.append(peakIdx)
            }
            i = j + 1
        } else {
            i += 1
        }
    }

    var peaks = peaksList

    if peaks.isEmpty {
        return ([], [:])
    }

    // ---------------------------------------------------------------
    // Height threshold: keep peaks whose value is >= height.
    // ---------------------------------------------------------------
    if let height = height {
        peaks = peaks.filter { x[$0] >= height }
    }

    if peaks.isEmpty {
        return ([], [:])
    }

    // ---------------------------------------------------------------
    // Prominence filter. For each peak, walk left and right collecting the
    // lowest value until a strictly higher sample is reached; the prominence
    // is the peak height above the higher of the two side minima.
    // ---------------------------------------------------------------
    if let prominence = prominence {
        var validPeaks: [Int] = []
        for peak in peaks {
            var leftMin = x[peak]
            var k = peak - 1
            while k >= 0 {
                if x[k] > x[peak] { break }
                leftMin = min(leftMin, x[k])
                k -= 1
            }

            var rightMin = x[peak]
            k = peak + 1
            while k < n {
                if x[k] > x[peak] { break }
                rightMin = min(rightMin, x[k])
                k += 1
            }

            let peakProminence = x[peak] - max(leftMin, rightMin)
            if peakProminence >= prominence {
                validPeaks.append(peak)
            }
        }
        peaks = validPeaks
    }

    if peaks.isEmpty {
        return ([], [:])
    }

    // ---------------------------------------------------------------
    // Distance constraint: greedily keep the highest peaks first, dropping any
    // that fall within `distance` samples of an already-kept peak.
    //
    // Python: `sorted_indices = np.argsort(x[peaks])[::-1]`. As instructed, we
    // treat this as a STABLE sort by descending value (ascending stable argsort
    // then reversed), so ties resolve deterministically.
    // ---------------------------------------------------------------
    if let distance = distance, peaks.count > 1 {
        let values = peaks.map { x[$0] }
        // Ascending, stable argsort (ties broken by original index), reversed.
        let sortedIndices = Array(0..<values.count)
            .sorted { a, b in
                if values[a] != values[b] { return values[a] < values[b] }
                return a < b
            }
            .reversed()
        let sortedPeaks = sortedIndices.map { peaks[$0] }

        var keep: [Int] = []
        for peak in sortedPeaks {
            // Keep when far enough from every already-kept peak.
            if keep.isEmpty || keep.allSatisfy({ Double(abs($0 - peak)) >= distance }) {
                keep.append(peak)
            }
        }
        // Restore ascending order (`np.array(sorted(keep))`).
        peaks = keep.sorted()
    }

    return (peaks, [:])
}
