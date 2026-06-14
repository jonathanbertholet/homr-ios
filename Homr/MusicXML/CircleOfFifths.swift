import Foundation

// MARK: - Circle of fifths and accidental bookkeeping
//
// Direct port of `homr/circle_of_fifths.py`. Maps between MusicXML key
// signatures (the number of sharps/flats, a.k.a. the "circle of fifths" value)
// and human key names, and tracks which notes carry an accidental within a
// measure so the engraver can decide when an explicit accidental must be drawn.

/// Maps a circle-of-fifths integer (-7..7) to its major key name.
/// Negative = flats, positive = sharps. Port of `definition`.
let definition: [Int: String] = [
    -7: "CbM",
    -6: "GbM",
    -5: "DbM",
    -4: "AbM",
    -3: "EbM",
    -2: "BbM",
    -1: "FM",
    0: "CM",
    1: "GM",
    2: "DM",
    3: "AM",
    4: "EM",
    5: "BM",
    6: "F#M",
    7: "C#M",
]

/// Inverse of `definition`: key name -> circle-of-fifths integer.
/// Port of `inv_definition = {v: k for k, v in definition.items()}`.
let inv_definition: [String: Int] = Dictionary(
    uniqueKeysWithValues: definition.map { ($0.value, $0.key) }
)

/// Note letters in the order sharps are added going around the circle.
/// Port of `circle_of_fifth_notes_positive`.
let circle_of_fifth_notes_positive = ["F", "C", "G", "D", "A", "E", "B"]

/// Note letters in the order flats are added (reverse of the sharp order).
/// Port of `circle_of_fifth_notes_negative = list(reversed(...))`.
let circle_of_fifth_notes_negative = Array(circle_of_fifth_notes_positive.reversed())

/// Port of `key_signature_to_circle_of_fifth`: look up a key name's
/// circle-of-fifths integer, warning and defaulting to C (0) if unknown.
func key_signature_to_circle_of_fifth(_ key_signature: String) -> Int {
    guard let value = inv_definition[key_signature] else {
        eprint("Warning: Unknown key signature", key_signature)
        return 0
    }
    return value
}

/// Port of `repeat_note_for_all_octaves`: expand each note letter into its
/// concrete pitch across octaves 0..10 (e.g. `"F"` -> `"F0"..."F10"`), matching
/// the `EncodedSymbol.pitch` spelling used to test membership.
func repeat_note_for_all_octaves(_ notes: [String]) -> [String] {
    var result: [String] = []
    for note in notes {
        for octave in 0..<11 {
            result.append(note + String(octave))
        }
    }
    return result
}

/// Port of the `AbstractKeyTransformation` ABC: an accidental tracker.
protocol AbstractKeyTransformation: AnyObject {
    /// Register `accidental` on `note` and return the accidental that must be
    /// drawn (empty string when it is redundant).
    func add_accidental(_ note: String, _ accidental: String?) -> String
    /// Return a fresh tracker for the next measure (accidentals reset at the
    /// bar line).
    func reset_at_end_of_measure() -> AbstractKeyTransformation
}

/// Port of `NoKeyTransformation`: tracks accidentals with no key context.
///
/// A class because it carries mutable per-measure accidental state.
final class NoKeyTransformation: AbstractKeyTransformation {
    /// note -> last accidental seen this measure. Port of `current_accidentals`.
    var current_accidentals: [String: String] = [:]

    init() {}

    /// Return `accidental` only the first time it differs from what is already
    /// in force for `note`; otherwise return `""`. Port of `add_accidental`.
    func add_accidental(_ note: String, _ accidental: String?) -> String {
        if let accidental = accidental, accidental != "",
            current_accidentals[note] != accidental
        {
            current_accidentals[note] = accidental
            return accidental
        }
        return ""
    }

    func reset_at_end_of_measure() -> AbstractKeyTransformation {
        NoKeyTransformation()
    }
}

/// Port of `KeyTransformation`: tracks accidentals relative to a key signature.
///
/// A class because it carries mutable `sharps`/`flats` sets that change as
/// accidentals are added and consumed through the measure.
final class KeyTransformation: AbstractKeyTransformation {
    /// The key signature as a circle-of-fifths integer.
    let circle_of_fifth: Int
    /// Concrete pitches currently sharpened (key signature + applied sharps).
    var sharps: Set<String> = []
    /// Concrete pitches currently flattened (key signature + applied flats).
    var flats: Set<String> = []

    /// Seed `sharps`/`flats` from the key signature. Positive -> the first
    /// `circle_of_fifth` sharp letters across all octaves; negative -> the
    /// first `|circle_of_fifth|` flat letters. Port of `__init__`.
    init(_ circle_of_fifth: Int) {
        self.circle_of_fifth = circle_of_fifth
        if circle_of_fifth > 0 {
            sharps = Set(
                repeat_note_for_all_octaves(
                    Array(circle_of_fifth_notes_positive[0..<circle_of_fifth])
                )
            )
        } else if circle_of_fifth < 0 {
            flats = Set(
                repeat_note_for_all_octaves(
                    Array(circle_of_fifth_notes_negative[0..<abs(circle_of_fifth)])
                )
            )
        }
    }

    /// Port of `KeyTransformation.add_accidental`.
    ///
    /// For an explicit accidental (`#`/`b`/`N`): remember the previous state of
    /// `note`, update the sets, and return the accidental only if it actually
    /// changes the note (returns `""` when it matches what was already in
    /// force). For any other value: cancel a previously applied sharp/flat by
    /// returning a natural `"N"`, otherwise return `""`.
    func add_accidental(_ note: String, _ accidental: String?) -> String {
        if let accidental = accidental, ["#", "b", "N"].contains(accidental) {
            var previous_accidental = "N"
            if sharps.contains(note) {
                sharps.remove(note)
                previous_accidental = "#"
            }
            if flats.contains(note) {
                flats.remove(note)
                previous_accidental = "b"
            }
            if accidental == "#" {
                sharps.insert(note)
            } else if accidental == "b" {
                flats.insert(note)
            }
            return accidental != previous_accidental ? accidental : ""
        } else {
            if sharps.contains(note) {
                sharps.remove(note)
                return "N"
            }
            if flats.contains(note) {
                flats.remove(note)
                return "N"
            }
            return ""
        }
    }

    func reset_at_end_of_measure() -> AbstractKeyTransformation {
        KeyTransformation(circle_of_fifth)
    }
}

/// Port of `maintain_accidentals_during_measure`.
///
/// The PrIMuS dataset does not maintain accidentals through a measure, whereas
/// `homr` and the other datasets do. This rewrites each symbol's `lift` so an
/// accidental stays in force until the bar line: explicit accidentals update
/// the running key state; otherwise the note inherits the sharp/flat implied by
/// the (here always C major) key. The bar line resets the tracker.
func maintain_accidentals_during_measure(_ symbols: [EncodedSymbol]) -> [EncodedSymbol] {
    var results: [EncodedSymbol] = []

    // PrIMuS treats keys as we expect, so we ignore the real key and use C
    // major (no accidentals) as the baseline.
    var key = KeyTransformation(0)

    for symbol in symbols {
        if symbol.rhythm.contains("barline") {
            // Equivalent to `key.reset_at_end_of_measure()`, which returns a
            // fresh `KeyTransformation(self.circle_of_fifth)`.
            key = KeyTransformation(key.circle_of_fifth)
            results.append(symbol)
        } else if symbol.lift != nonote {
            // In engraving the lift may be empty (implied by key/previous
            // accidental); for sounding we need the actual pitch.
            let lift: String? = symbol.lift != empty ? symbol.lift : nil
            let actual_accidental: String

            if let lift = lift, ["#", "b", "N"].contains(lift) {
                actual_accidental = lift
                // Record that this accidental now applies for later notes.
                _ = key.add_accidental(symbol.pitch, lift)
            } else if key.sharps.contains(symbol.pitch) {
                actual_accidental = "#"
            } else if key.flats.contains(symbol.pitch) {
                actual_accidental = "b"
            } else {
                actual_accidental = empty
            }

            results.append(symbol.change_lift(actual_accidental))
        } else {
            results.append(symbol)
        }
    }

    return results
}

/// Port of `strip_naturals`: drop explicit natural signs, turning a `lift` of
/// `"N"` into the empty sentinel.
func strip_naturals(_ symbols: [EncodedSymbol]) -> [EncodedSymbol] {
    symbols.map { symbol in
        symbol.lift == "N" ? symbol.change_lift(empty) : symbol
    }
}
