import Foundation

// Port of `homr/transformer/vocabulary.py` — the token vocabularies that bridge
// the neural transformer output and the MusicXML generator.
//
// CRITICAL: the insertion ORDER of every builder must match the order used when
// the ONNX model was trained, since the token index is the model's class id.
// `itertools.product` and `reversed(...)` orderings are reproduced exactly.

/// Sentinel for "no note" / absent field. Python `nonote = "."`.
let nonote = "."
/// Sentinel for "no decoration" on a note. Python `empty = "_"`.
let empty = "_"

/// Lightweight stderr logger replacing Python's `eprint`. Diagnostics only.
func eprint(_ items: Any...) {
    let message = items.map { "\($0)" }.joined(separator: " ")
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

/// Port of `build_dict`: assign each token an index in order, validating the
/// build-time invariants. These are training-vocab invariants, so violations are
/// programmer errors — `preconditionFailure` mirrors Python's `raise ValueError`.
func build_dict(_ tokens: [String]) -> [String: Int] {
    var result: [String: Int] = [:]
    for (i, t) in tokens.enumerated() {
        if result[t] != nil {
            preconditionFailure("Duplicated entry for " + t)
        }
        if t.trimmingCharacters(in: .whitespaces).isEmpty {
            preconditionFailure("Tokens must not be a whitespace, this makes parsing easier")
        }
        if t.trimmingCharacters(in: .whitespaces) != t {
            preconditionFailure("Tokens must not contain a whitespace, this makes parsing easier")
        }
        if t.contains("&") {
            preconditionFailure("& is reserved as alternatives for chords")
        }
        result[t] = i
    }
    return result
}

/// Port of `build_rhythm`. Builds the rhythm vocabulary in exact training order.
func build_rhythm() -> [String: Int] {
    var rhythm: [String] = []

    // Sequence symbols.
    rhythm.append(contentsOf: ["PAD", "BOS", "EOS"])
    rhythm.append("chord")

    // Bar lines.
    rhythm.append(contentsOf: ["barline", "doublebarline", "bolddoublebarline"])
    rhythm.append(contentsOf: ["repeatStart", "repeatEnd", "repeatEndStart"])
    rhythm.append(contentsOf: ["voltaStart", "voltaStop", "voltaDiscontinue"])

    // Clefs.
    rhythm.append(contentsOf: (3..<6).map { "clef_F\($0)" })
    rhythm.append(contentsOf: (1..<6).map { "clef_C\($0)" })
    rhythm.append(contentsOf: (1..<3).map { "clef_G\($0)" })

    // Signatures.
    rhythm.append(contentsOf: (-7..<8).map { "keySignature_\($0)" })
    rhythm.append(contentsOf: [1, 2, 3, 4, 6, 8, 12, 16, 32, 48].map { "timeSignature/\($0)" })

    // Rhythm — kern durations per https://www.humdrum.org/rep/kern/.
    rhythm.append(contentsOf: (2..<11).map { "rest_\($0)m" })  // multirests
    let kernBaseDurations = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128]
    let dots = ["", ".", ".."]
    let grace = ["", "G"]
    // itertools.product(kernBaseDurations, grace, dots): outer = base, then
    // grace, then dots. f"{d}{g}{dot}".
    var kernValues: [String] = []
    for d in kernBaseDurations {
        for g in grace {
            for dot in dots {
                kernValues.append("\(d)\(g)\(dot)")
            }
        }
    }

    // Durations coming from tuplets.
    let irregularDurations = [7, 11, 13, 18, 20, 21, 22, 24, 26, 28, 30, 34, 36, 40, 48, 56, 96]

    rhythm.append(contentsOf: kernValues.map { "note_\($0)" })
    rhythm.append(contentsOf: irregularDurations.map { "note_\($0)" })
    rhythm.append(contentsOf: kernValues.map { "rest_\($0)" })
    rhythm.append(contentsOf: irregularDurations.map { "rest_\($0)" })

    return build_dict(rhythm)
}

/// Port of `build_lift`.
func build_lift() -> [String: Int] {
    let lifts = [nonote, empty, "#", "##", "N", "b", "bb"]
    return build_dict(lifts)
}

/// Port of `build_position`. The staff position, applies to notes/rests/clefs.
func build_position() -> [String: Int] {
    let positions = [nonote, "upper", "lower"]
    return build_dict(positions)
}

/// Port of `build_articulation`. The superset of articulations (from the lieder
/// dataset), in exact order.
func build_articulation() -> [String: Int] {
    var articulation = [nonote, empty]

    let articulationsLieder = [
        "accent",
        "accent_arpeggiate",
        "accent_arpeggiate_fermata",
        "accent_arpeggiate_staccato",
        "accent_arpeggiate_tenuto",
        "accent_breathMark",
        "accent_breathMark_fermata",
        "accent_fermata",
        "accent_fermata_staccato",
        "accent_staccatissimo",
        "accent_staccato",
        "accent_staccato_tenuto",
        "accent_tenuto",
        "accent_tremolo",
        "accent_trill",
        "accent_fermata_trill",
        "arpeggiate",
        "arpeggiate_breathMark_fermata",
        "arpeggiate_fermata",
        "arpeggiate_fermata_staccato",
        "arpeggiate_staccatissimo",
        "arpeggiate_staccato",
        "arpeggiate_staccato_tenuto",
        "arpeggiate_tenuto",
        "arpeggiate_tremolo",
        "arpeggiate_trill",
        "breathMark",
        "breathMark_fermata",
        "breathMark_fermata_tenuto",
        "breathMark_staccato",
        "breathMark_tenuto",
        "breathMark_trill",
        "breathMark_tremolo",
        "breathMark_staccato_tenuto",
        "fermata",
        "fermata_staccato",
        "fermata_staccato_tenuto",
        "fermata_tenuto",
        "fermata_tremolo",
        "fermata_trill",
        "fermata_turn",
        "spiccato",
        "staccatissimo",
        "staccato",
        "staccato_tenuto",
        "staccato_tremolo",
        "staccato_trill",
        "staccato_turn",
        "tenuto",
        "tremolo",
        "trill",
        "turn",
    ]

    articulation.append(contentsOf: articulationsLieder)

    return build_dict(articulation)
}

/// Port of `build_slur`.
func build_slur() -> [String: Int] {
    var slur = [nonote, empty]
    slur.append(contentsOf: ["slurStart_slurStop", "slurStart", "slurStop"])
    return build_dict(slur)
}

/// Port of `build_pitch`.
///
/// Reproduces `reversed([f"{n}{octave}" for octave, n in
/// itertools.product(range(10), note_names)])`: product's first operand is the
/// octave (0..9), second is the note name (C,D,E,F,G,A,B), and the string is
/// `n + str(octave)` (e.g. "C0"). The full list is then reversed.
func build_pitch() -> [String: Int] {
    var pitch = [nonote, empty]
    let noteNames = ["C", "D", "E", "F", "G", "A", "B"]
    var names: [String] = []
    for octave in 0..<10 {
        for n in noteNames {
            names.append("\(n)\(octave)")
        }
    }
    pitch.append(contentsOf: names.reversed())
    return build_dict(pitch)
}

/// Port of `has_rhythm_symbol_a_position`: note/rest/clef carry a staff position.
func has_rhythm_symbol_a_position(_ rhythm: String) -> Bool {
    rhythm.hasPrefix("note") || rhythm.hasPrefix("rest") || rhythm.hasPrefix("clef")
}

/// Port of the `Vocabulary` class.
///
/// Holds the six token→index sub-vocabularies. Each has a matching computed
/// inverse (`index→token`) that the decoder agent uses to turn model class ids
/// back into tokens.
struct Vocabulary {
    /// token → index maps, in training order.
    let rhythm: [String: Int]
    let lift: [String: Int]
    let articulation: [String: Int]
    let pitch: [String: Int]
    let slur: [String: Int]
    let position: [String: Int]

    init() {
        self.rhythm = build_rhythm()
        self.lift = build_lift()
        self.articulation = build_articulation()
        self.pitch = build_pitch()
        self.slur = build_slur()
        self.position = build_position()
    }

    /// index → token inverse maps (for decoding model output back to tokens).
    var rhythmInverse: [Int: String] { Vocabulary.invert(rhythm) }
    var liftInverse: [Int: String] { Vocabulary.invert(lift) }
    var articulationInverse: [Int: String] { Vocabulary.invert(articulation) }
    var pitchInverse: [Int: String] { Vocabulary.invert(pitch) }
    var slurInverse: [Int: String] { Vocabulary.invert(slur) }
    var positionInverse: [Int: String] { Vocabulary.invert(position) }

    /// Invert a token→index map. Indices are unique by construction of
    /// `build_dict`, so the inverse is well defined.
    private static func invert(_ map: [String: Int]) -> [Int: String] {
        Dictionary(uniqueKeysWithValues: map.map { ($0.value, $0.key) })
    }
}
