import Foundation

// Module-level grouping / cleanup helpers from `homr/transformer/vocabulary.py`.
//
// These operate on flat symbol streams and the nested chord / measure groupings
// derived from them:
//   - `[EncodedSymbol]`            — a flat symbol stream
//   - `[[EncodedSymbol]]`          — chords
//   - `[[[EncodedSymbol]]]`        — measures of chords
//
// Several functions contain subtle quirks in the Python original; those are
// replicated bug-for-bug and called out in the doc comments.

/// Port of `_remove_redudant_clefs_keys_and_time_signatures` (original spelling
/// preserved).
///
/// Drops clef / key-signature / time-signature symbols that repeat the most
/// recent value (tracked separately for upper vs lower clef). All other symbols
/// pass through.
func _remove_redudant_clefs_keys_and_time_signatures(
    _ chords: [[EncodedSymbol]]
) -> [[EncodedSymbol]] {
    var clefUpper = ""
    var clefLower = ""
    var key = ""
    var time = ""
    var resultChords: [[EncodedSymbol]] = []
    for chord in chords {
        var result: [EncodedSymbol] = []
        for symbol in chord {
            if symbol.rhythm.hasPrefix("clef") {
                if symbol.position == "upper" {
                    if symbol.rhythm != clefUpper {
                        clefUpper = symbol.rhythm
                        result.append(symbol)
                    }
                } else if symbol.rhythm != clefLower {
                    clefLower = symbol.rhythm
                    result.append(symbol)
                }
            } else if symbol.rhythm.hasPrefix("keySignature") {
                if symbol.rhythm != key {
                    key = symbol.rhythm
                    result.append(symbol)
                }
            } else if symbol.rhythm.hasPrefix("timeSignature") {
                if symbol.rhythm != time {
                    time = symbol.rhythm
                    result.append(symbol)
                }
            } else {
                result.append(symbol)
            }
        }
        resultChords.append(result)
    }
    return resultChords
}

/// Port of `_remove_duplicated_piches` (original spelling preserved).
///
/// Collapses duplicate notes within a chord, keyed by `"pitch position"`,
/// keeping first appearance order.
///
/// Bug-for-bug note: when a duplicate key is seen and the new symbol is longer,
/// the Python code writes `by_pitch[symbol.pitch] = symbol` — keyed by `pitch`
/// alone, NOT by the `"pitch position"` key it reads back. That write therefore
/// lands under a different key and never affects the returned list. Replicated
/// exactly.
func _remove_duplicated_piches(_ chord: [EncodedSymbol]) -> [EncodedSymbol] {
    if chord.count <= 1 || !(chord[0].rhythm.hasPrefix("note") || chord[0].rhythm.hasPrefix("rest")) {
        return chord
    }
    var byPitch: [String: EncodedSymbol] = [:]
    var orderOfAppearance: [String] = []
    for symbol in chord {
        let key = symbol.pitch + " " + symbol.position
        if let existing = byPitch[key] {
            if symbol.get_duration().fraction > existing.get_duration().fraction {
                // Bug-for-bug: keyed by `symbol.pitch`, not `key`.
                byPitch[symbol.pitch] = symbol
            }
        } else {
            byPitch[key] = symbol
            orderOfAppearance.append(key)
        }
    }

    return orderOfAppearance.map { byPitch[$0]! }
}

/// Port of `_group_into_chords`.
///
/// A `chord` rhythm marks that the following symbol joins the previous group;
/// otherwise each symbol starts a new group.
func _group_into_chords(_ symbols: [EncodedSymbol]) -> [[EncodedSymbol]] {
    var chords: [[EncodedSymbol]] = []
    var isInChord = false
    for symbol in symbols {
        if symbol.rhythm == "chord" {
            isInChord = true
        } else if isInChord && chords.count > 0 {
            chords[chords.count - 1].append(symbol)
            isInChord = false
        } else {
            chords.append([symbol])
        }
    }
    return chords
}

/// Port of `_flatten_chords`.
///
/// Re-inserts `chord` separator symbols between members of a chord group.
///
/// Bug-for-bug note: the guard inside the loop checks `len(chords) == 0` (the
/// outer list), not `len(chord)`. Since the loop only iterates when `chords` is
/// non-empty, the guard never triggers. Replicated exactly.
func _flatten_chords(_ chords: [[EncodedSymbol]]) -> [EncodedSymbol] {
    var result: [EncodedSymbol] = []
    for chord in chords {
        if chords.isEmpty {  // bug-for-bug: checks the outer list, never true here
            continue
        }
        var isInChord = false
        for symbol in chord {
            if isInChord {
                result.append(EncodedSymbol("chord"))
            }
            result.append(symbol)
            isInChord = true
        }
    }
    return result
}

/// Port of `_group_into_measures`.
///
/// Closes a measure after a chord whose first symbol's rhythm contains
/// `"barline"` or `"repeat"`.
func _group_into_measures(_ chords: [[EncodedSymbol]]) -> [[[EncodedSymbol]]] {
    var measures: [[[EncodedSymbol]]] = []
    var currentMeasure: [[EncodedSymbol]] = []
    for chord in chords {
        currentMeasure.append(chord)
        if chord.count > 0 && (chord[0].rhythm.contains("barline") || chord[0].rhythm.contains("repeat")) {
            measures.append(currentMeasure)
            currentMeasure = []
        }
    }
    if currentMeasure.count > 0 {
        measures.append(currentMeasure)
    }
    return measures
}

/// Port of `_flatten_measures`: concatenate all chords across measures.
func _flatten_measures(_ measures: [[[EncodedSymbol]]]) -> [[EncodedSymbol]] {
    measures.flatMap { $0 }
}

/// Port of `_get_duration_of_measure`.
///
/// Sums, over each chord, the shortest positive note/rest duration in that chord
/// (i.e. the chord's effective onset duration).
func _get_duration_of_measure(_ measure: [[EncodedSymbol]]) -> Fraction {
    var totalDuration = Fraction(0)
    for chord in measure {
        var duration = Fraction(0)
        for symbol in chord {
            if symbol.rhythm.hasPrefix("note") || symbol.rhythm.hasPrefix("rest") {
                let fraction = symbol.get_duration().fraction
                if fraction > Fraction(0) && (fraction < duration || duration == Fraction(0)) {
                    duration = fraction
                }
            }
        }
        totalDuration = totalDuration + duration
    }
    return totalDuration
}

/// Port of `_get_typical_duration_of_measures`: the upper-median measure
/// duration (`sorted(durations)[len // 2]`).
func _get_typical_duration_of_measures(_ measures: [[[EncodedSymbol]]]) -> Fraction {
    let durations = measures.map { _get_duration_of_measure($0) }
    if durations.count == 0 {
        return Fraction(0)
    }
    return durations.sorted()[durations.count / 2]
}

/// Port of `_remove_tuplets`: strip tuplet scaling from every symbol.
func _remove_tuplets(_ measure: [[EncodedSymbol]]) -> [[EncodedSymbol]] {
    measure.map { chord in chord.map { $0.remove_tuplet() } }
}

/// Port of `_fix_over_eager_tuplets`.
///
/// The transformer tends to over-predict tuplets. Measures shorter than the
/// typical measure duration have their tuplets removed.
func _fix_over_eager_tuplets(_ chords: [[EncodedSymbol]]) -> [[EncodedSymbol]] {
    let measures = _group_into_measures(chords)
    let mean = _get_typical_duration_of_measures(measures)
    var result: [[[EncodedSymbol]]] = []
    for (i, measure) in measures.enumerated() {
        if _get_duration_of_measure(measure) < mean {
            eprint("Removing tuplets from measure #", i + 1)
            result.append(_remove_tuplets(measure))
        } else {
            result.append(measure)
        }
    }
    return _flatten_measures(result)
}

/// Port of `_only_keep_lower_staff_if_there_is_a_clef`.
///
/// Until a lower-staff clef is seen (within the first 5 chords), all symbols are
/// forced to the upper position. Once a lower clef appears, symbols pass through
/// unchanged.
///
/// Bug-for-bug note: `delta = len(chords) - len(all_results)` is always 0 (the
/// output has one entry per input chord), so the "no matching clef" warning
/// never fires. Replicated exactly.
func _only_keep_lower_staff_if_there_is_a_clef(
    _ chords: [[EncodedSymbol]]
) -> [[EncodedSymbol]] {
    var hasLowerClef = false
    var allResults: [[EncodedSymbol]] = []
    for (i, chord) in chords.enumerated() {
        var result: [EncodedSymbol] = []
        for symbol in chord {
            if hasLowerClef {
                result.append(symbol)
            } else if i < 5 && symbol.rhythm.hasPrefix("clef") && symbol.position == "lower" {
                hasLowerClef = true
                result.append(symbol)
            } else {
                result.append(symbol.to_upper_position())
            }
        }
        allResults.append(result)
    }
    let delta = chords.count - allResults.count
    if delta > 0 {
        eprint("Removed", delta, "results as there was no matching clef")
    }
    return allResults
}

/// Port of the public `remove_duplicated_symbols`.
///
/// Pipeline: group into chords → (optionally) fix over-eager tuplets and resolve
/// staff position → drop duplicate pitches per chord → drop redundant
/// clefs/keys/time signatures → flatten back to a symbol stream.
func remove_duplicated_symbols(
    _ symbols: [EncodedSymbol], cleanupTuplets: Bool = true
) -> [EncodedSymbol] {
    var chords = _group_into_chords(symbols)
    if cleanupTuplets {
        chords = _fix_over_eager_tuplets(chords)
        chords = _only_keep_lower_staff_if_there_is_a_clef(chords)
    }
    chords = chords.map { _remove_duplicated_piches($0) }
    chords = _remove_redudant_clefs_keys_and_time_signatures(chords)
    return _flatten_chords(chords)
}

/// Port of the public `sort_token_chords`.
///
/// Groups symbols into chords (like `_group_into_chords`, optionally retaining
/// an explicit `chord` separator) and returns each chord sorted. Sorting uses
/// `EncodedSymbol`'s REVERSED `<`, so each chord ends up ordered by descending
/// string representation.
func sort_token_chords(
    _ symbols: [EncodedSymbol], keepChordSymbol: Bool = false
) -> [[EncodedSymbol]] {
    var chords: [[EncodedSymbol]] = []
    var isInChord = false
    for symbol in symbols {
        if symbol.rhythm == "chord" {
            isInChord = true
        } else if isInChord && chords.count > 0 {
            if keepChordSymbol {
                chords[chords.count - 1].append(EncodedSymbol("chord"))
            }
            chords[chords.count - 1].append(symbol)
            isInChord = false
        } else {
            chords.append([symbol])
        }
    }

    return chords.map { $0.stableSorted() }
}
