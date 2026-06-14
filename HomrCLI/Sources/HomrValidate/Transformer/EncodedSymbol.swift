import Foundation

// MARK: - Duration parsing
//
// Port of the duration helpers from `homr/transformer/vocabulary.py`. The kern
// duration grammar comes from Humdrum **kern (https://www.humdrum.org/rep/kern/).

/// A parsed Humdrum **kern duration.
///
/// Mirrors Python `SymbolDuration`. The initializer reduces
/// `Fraction(actual_notes, normal_notes)` and exposes the resulting numerator /
/// denominator as `actualNotes` / `normalNotes`, then precomputes the final
/// duration `fraction` (relative to a whole note) via `_to_fraction`.
struct SymbolDuration {
    /// Base note value before dots/tuplets, e.g. `Fraction(1, 4)` for a quarter.
    let baseDuration: Fraction
    /// Number of augmentation dots.
    let dots: Int
    /// Tuplet "actual" count = numerator of the reduced `actual/normal` ratio.
    let actualNotes: Int
    /// Tuplet "normal" count = denominator of the reduced `actual/normal` ratio.
    let normalNotes: Int
    /// Final exact duration relative to a whole note.
    let fraction: Fraction
    /// Original kern base integer (passed through verbatim from the parser).
    let kern: Int

    /// Mirrors `SymbolDuration.__init__`: reduce the tuplet ratio, then compute
    /// the final fraction from base value + dots + tuplet scaling.
    init(baseDuration: Fraction, dots: Int, actualNotes: Int, normalNotes: Int, kern: Int) {
        self.baseDuration = baseDuration
        self.dots = dots
        let actionNormal = Fraction(actualNotes, normalNotes)
        self.actualNotes = actionNormal.numerator
        self.normalNotes = actionNormal.denominator
        self.kern = kern
        self.fraction = SymbolDuration.toFraction(
            baseDuration: baseDuration,
            dots: dots,
            actualNotes: actionNormal.numerator,
            normalNotes: actionNormal.denominator
        )
    }

    /// Port of `SymbolDuration._to_fraction`.
    ///
    /// Applies dots (`dur += add; add /= 2` accumulation) and tuplet scaling
    /// (`dur *= Fraction(normal, actual)` when `actual != normal`).
    private static func toFraction(
        baseDuration: Fraction, dots: Int, actualNotes: Int, normalNotes: Int
    ) -> Fraction {
        var dur = baseDuration

        // Apply dots.
        var add = dur / Fraction(2)
        for _ in 0..<dots {
            dur = dur + add
            add = add / Fraction(2)
        }

        // Apply tuplet scaling.
        if actualNotes != normalNotes {
            dur = dur * Fraction(normalNotes, actualNotes)
        }

        return dur
    }
}

/// Port of `prior_power_of_two`: the largest power of two `<= n`.
///
/// Returns 1 for `n < 1` (Python comment: "produces wrong rhythms, but at least
/// it produces something").
func prior_power_of_two(_ n: Int) -> Int {
    if n < 1 {
        return 1
    }
    // Equivalent to Python `1 << (n.bit_length() - 1)`.
    let bitLength = Int.bitWidth - n.leadingZeroBitCount
    return 1 << (bitLength - 1)
}

/// Port of `kern_to_symbol_duration`: parse a kern duration string.
///
/// Bug-for-bug note: the Python `kern.endswith("m")` (multirest) branch
/// constructs a `SymbolDuration(...)` but never returns it — it is dead code and
/// execution falls through to the generic numeric parse below. This port
/// preserves that exactly (the constructed value is discarded).
func kern_to_symbol_duration(_ kern: String) -> SymbolDuration {
    if kern.hasSuffix("m") {
        // Multirest — dead code in the original (computed, not returned). Kept
        // for fidelity; the result is intentionally discarded.
        _ = SymbolDuration(baseDuration: Fraction(1), dots: 0, actualNotes: 1, normalNotes: 1, kern: 4)
    }

    // Extract the leading numeric prefix (can be more than one digit).
    let chars = Array(kern)
    var i = 0
    while i < chars.count, chars[i] >= "0", chars[i] <= "9" {
        i += 1
    }
    let baseStr = String(chars[0..<i])
    let rest = String(chars[i...])

    let base = baseStr.isEmpty ? 4 : (Int(baseStr) ?? 4)  // default quarter
    let dots = rest.filter { $0 == "." }.count

    if kern.contains("G") {
        // Grace note.
        return SymbolDuration(baseDuration: Fraction(0), dots: dots, actualNotes: 1, normalNotes: 1, kern: base)
    }
    if base == 0 {
        // Special: whole-measure rest.
        return SymbolDuration(baseDuration: Fraction(1), dots: dots, actualNotes: 1, normalNotes: 1, kern: base)
    }

    // If base is a power of two, it's a normal note.
    if base & (base - 1) == 0 {
        let baseDuration = Fraction(1, base)
        return SymbolDuration(baseDuration: baseDuration, dots: dots, actualNotes: 1, normalNotes: 1, kern: base)
    } else {
        // Tuplet case: find the prior power of two.
        let normalNotes = prior_power_of_two(base)
        let baseDuration = Fraction(1, normalNotes)
        let actualNotes = base
        return SymbolDuration(
            baseDuration: baseDuration, dots: dots,
            actualNotes: actualNotes, normalNotes: normalNotes, kern: normalNotes
        )
    }
}

// MARK: - EncodedSymbol

/// A musical symbol split into the separate decoder branches.
///
/// Port of Python `EncodedSymbol`. Implemented as a `struct` (value type) so the
/// many `copy.copy(self)` + mutate-one-field patterns become trivial: each
/// helper takes `var result = self`, mutates one field, and returns it.
///
/// Equality / hashing cover only the six string fields (rhythm, pitch, lift,
/// articulation, slur, position) — never `coordinates` or the cached duration —
/// matching Python `__eq__` / `__hash__`.
///
/// `Comparable` is intentionally REVERSED, mirroring Python `__lt__`
/// (`return str(self) > str(other)`). `sort_token_chords` relies on this.
struct EncodedSymbol: Equatable, Hashable, Comparable, CustomStringConvertible {
    var rhythm: String
    var pitch: String
    var lift: String
    var articulation: String
    var slur: String
    var position: String

    /// Optional 2D coordinates derived from transformer attention.
    ///
    /// In Python this stores an attention array, but downstream code only ever
    /// uses it as an optional pair of coordinates (or `None`). The decoder agent
    /// sets/ignores it; it is excluded from equality and hashing.
    var coordinates: (Double, Double)?

    /// Reference-typed lazy duration cache.
    ///
    /// Python memoizes the parsed duration in the instance attribute
    /// `_duration`. To preserve that memoization across a value type without
    /// forcing every call site to use `var`, the cache lives in a class box.
    /// Copies share the box, which is safe because the only field that affects
    /// duration is `rhythm`, and the only operation that changes `rhythm`
    /// (`remove_tuplet`) installs a fresh box (mirroring `result._duration = None`).
    private final class DurationBox {
        var value: SymbolDuration?
    }
    private var durationBox = DurationBox()

    init(
        _ rhythm: String,
        pitch: String = nonote,
        lift: String = nonote,
        articulation: String = nonote,
        slur: String = nonote,
        position: String = nonote,
        coordinates: (Double, Double)? = nil
    ) {
        self.rhythm = rhythm
        self.pitch = pitch
        self.lift = lift
        self.articulation = articulation
        self.slur = slur
        self.position = position
        self.coordinates = coordinates
    }

    /// Port of `is_control_symbol`.
    func is_control_symbol() -> Bool {
        rhythm == "BOS" || rhythm == "EOS" || rhythm == "PAD"
    }

    /// Port of `is_tuplet`: true when stripping the tuplet changes the rhythm.
    func is_tuplet() -> Bool {
        remove_tuplet().rhythm != rhythm
    }

    /// Port of `remove_tuplet`.
    ///
    /// Mirrors the regex `(note|rest)_(\d+)(.*)` via manual parsing and the exact
    /// `% 3` / `% 5` / `% 7` reduction. Returns `self` unchanged if the rhythm
    /// does not match or the duration is not a tuplet multiple.
    func remove_tuplet() -> EncodedSymbol {
        // Match group 1 = "note"|"rest".
        let prefix: String
        if rhythm.hasPrefix("note_") {
            prefix = "note"
        } else if rhythm.hasPrefix("rest_") {
            prefix = "rest"
        } else {
            return self
        }

        // Match group 2 = \d+ (greedy run of digits after the underscore).
        let chars = Array(rhythm)
        var idx = prefix.count + 1  // skip "note_" / "rest_"
        let digitStart = idx
        while idx < chars.count, chars[idx] >= "0", chars[idx] <= "9" {
            idx += 1
        }
        // \d+ requires at least one digit.
        if idx == digitStart {
            return self
        }
        guard var duration = Int(String(chars[digitStart..<idx])) else {
            return self
        }
        // Match group 3 = .* (the remainder, e.g. dots).
        let suffix = String(chars[idx...])

        if duration % 3 == 0 {
            duration = duration / 3 * 2
        } else if duration % 5 == 0 {
            duration = duration / 5 * 4
        } else if duration % 7 == 0 {
            duration = duration / 7 * 4
        } else {
            return self
        }

        var result = self
        result.rhythm = prefix + "_" + String(duration) + suffix
        result.durationBox = DurationBox()  // mirrors result._duration = None
        return result
    }

    /// Port of `change_lift`.
    func change_lift(_ lift: String) -> EncodedSymbol {
        var result = self
        result.lift = lift
        return result
    }

    /// Port of `to_upper_position`.
    func to_upper_position() -> EncodedSymbol {
        if position != "lower" {
            return self
        }
        var result = self
        result.position = "upper"
        return result
    }

    /// Port of `is_valid`.
    ///
    /// A note-like rhythm (note/rest/clef) must have every decoration field set
    /// (`!= nonote`); a non-note rhythm must have every field unset (`== nonote`).
    func is_valid() -> Bool {
        let hasPosition = has_rhythm_symbol_a_position(rhythm)
        let isNote = [lift, articulation, pitch, slur, position].map { $0 != nonote }
        return isNote.allSatisfy { $0 == hasPosition }
    }

    /// Port of `add_articulations`: merge in new articulations and re-sort.
    ///
    /// `sorted` on `[String]` is lexicographic by Unicode scalar, matching
    /// Python's `sorted` for these ASCII tokens.
    func add_articulations(_ articulations: [String]) -> EncodedSymbol {
        var all = articulations
        all.append(contentsOf: EncodedSymbol.splitNonEmpty(self.articulation))
        var result = self
        result.articulation = all.sorted().joined(separator: "_")
        return result
    }

    /// Port of `add_slurs`: merge in new slurs and re-sort.
    func add_slurs(_ slurs: [String]) -> EncodedSymbol {
        var all = slurs
        all.append(contentsOf: EncodedSymbol.splitNonEmpty(self.slur))
        var result = self
        result.slur = all.sorted().joined(separator: "_")
        return result
    }

    /// Port of `strip_articulations`.
    ///
    /// Returns the stripped tokens (in original order, not sorted) and a copy
    /// whose articulation is the remaining tokens, or `empty` if none remain.
    func strip_articulations(
        _ toBeRemoved: [String], removeAll: Bool = false
    ) -> ([String], EncodedSymbol) {
        var stripped: [String] = []
        var remaining: [String] = []
        for articulation in EncodedSymbol.splitNonEmpty(self.articulation) {
            if removeAll || toBeRemoved.contains(articulation) {
                stripped.append(articulation)
            } else {
                remaining.append(articulation)
            }
        }
        var result = self
        result.articulation = remaining.isEmpty ? empty : remaining.joined(separator: "_")
        return (stripped, result)
    }

    /// Port of `strip_slurs`.
    func strip_slurs(
        _ toBeRemoved: [String], removeAll: Bool = false
    ) -> ([String], EncodedSymbol) {
        var stripped: [String] = []
        var remaining: [String] = []
        for slur in EncodedSymbol.splitNonEmpty(self.slur) {
            if removeAll || toBeRemoved.contains(slur) {
                stripped.append(slur)
            } else {
                remaining.append(slur)
            }
        }
        var result = self
        result.slur = remaining.isEmpty ? empty : remaining.joined(separator: "_")
        return (stripped, result)
    }

    /// Port of `get_duration`.
    ///
    /// Returns a zero-duration placeholder for non note/rest rhythms (the Python
    /// code logs a warning via `eprint`). Otherwise parses `rhythm.split("_")[1]`.
    /// Caches into `durationBox` (mirrors Python's `_duration` memoization).
    func get_duration() -> SymbolDuration {
        if let cached = durationBox.value {
            return cached
        }

        if !(rhythm.hasPrefix("note") || rhythm.hasPrefix("rest")) {
            eprint("Warning, invalid symbol in group: Only notes and rests have durations")
            return SymbolDuration(baseDuration: Fraction(0), dots: 0, actualNotes: 1, normalNotes: 1, kern: 1)
        }
        // Python `rhythm.split("_")[1]` — second underscore-separated component.
        let kern = rhythm.components(separatedBy: "_")[1]

        let duration = kern_to_symbol_duration(kern)
        durationBox.value = duration
        return duration
    }

    /// Port of `__str__`: space-joined six fields.
    var description: String {
        [rhythm, pitch, lift, articulation, slur, position].joined(separator: " ")
    }

    /// Port of `__eq__` / `__hash__`: the six string fields only.
    static func == (lhs: EncodedSymbol, rhs: EncodedSymbol) -> Bool {
        lhs.rhythm == rhs.rhythm
            && lhs.pitch == rhs.pitch
            && lhs.lift == rhs.lift
            && lhs.articulation == rhs.articulation
            && lhs.slur == rhs.slur
            && lhs.position == rhs.position
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(rhythm)
        hasher.combine(pitch)
        hasher.combine(lift)
        hasher.combine(articulation)
        hasher.combine(slur)
        hasher.combine(position)
    }

    /// Port of `__lt__`, intentionally REVERSED (`str(self) > str(other)`).
    /// `sorted(...)` therefore orders symbols by descending `description`.
    static func < (lhs: EncodedSymbol, rhs: EncodedSymbol) -> Bool {
        lhs.description > rhs.description
    }

    /// Helper mirroring Python `[a for a in value.split("_") if a]`:
    /// split on `_` and drop empty components.
    private static func splitNonEmpty(_ value: String) -> [String] {
        value.split(separator: "_", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { !$0.isEmpty }
    }
}

extension Array where Element == EncodedSymbol {
    /// Stable sort matching Python's `sorted` (which is guaranteed stable).
    ///
    /// Swift's `sorted()` is NOT guaranteed stable, so symbols with an equal
    /// `description` (the `<` key) could be reordered, which would scramble the
    /// `coordinates` the decoder depends on. This decorates each element with its
    /// original index and breaks ties on it, preserving input order — exactly
    /// like CPython's Timsort.
    func stableSorted() -> [EncodedSymbol] {
        enumerated()
            .sorted { lhs, rhs in
                if lhs.element < rhs.element { return true }
                if rhs.element < lhs.element { return false }
                return lhs.offset < rhs.offset
            }
            .map { $0.element }
    }
}
