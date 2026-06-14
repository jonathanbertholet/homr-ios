import Foundation

// MARK: - Instrument identity
//
// The OMR pipeline only separates parts geometrically (one per staff position);
// it never reads which instrument a staff belongs to. This type plus the catalog
// below close that gap: an OCR'd staff label (e.g. "Oboe d'amore I", "Basso",
// "Continuo") is matched to a General MIDI voice so the exported MusicXML/MIDI
// and the in-app synth can render each part with a distinct timbre.

/// A recognised instrument for one part.
struct Instrument: Sendable, Equatable, Codable, Hashable {
    /// Display name used for the MusicXML `<part-name>` / `<instrument-name>`.
    let name: String
    /// MusicXML `<instrument-sound>` id (e.g. `strings.cello`).
    let sound: String
    /// General MIDI program, 0-based (0 = Acoustic Grand Piano), as used by the
    /// synth and by Standard MIDI File program-change events.
    let gmProgram: Int

    /// MusicXML `<midi-program>` is 1-based.
    var midiProgram1Based: Int { gmProgram + 1 }
}

/// Maps free-text instrument labels (English / Italian / German / French terms
/// common on engraved scores) to General MIDI voices.
enum InstrumentCatalog {
    /// A keyword → instrument rule. Keywords are matched against a normalised,
    /// lowercased label; order matters (most specific terms come first so e.g.
    /// "double bass" wins over "bass", and "english horn" over "horn").
    private struct Rule {
        let keywords: [String]
        let instrument: Instrument
    }

    private static func inst(_ name: String, _ sound: String, _ gm0: Int) -> Instrument {
        Instrument(name: name, sound: sound, gmProgram: gm0)
    }

    // GM programs below are 0-based (subtract 1 from the familiar 1-based numbers).
    private static let rules: [Rule] = [
        // Double reeds / woodwinds — specific before generic.
        Rule(keywords: ["oboe d'amore", "oboe damore", "oboe d amore"], instrument: inst("Oboe d'amore", "wind.reed.oboe.oboe-damore", 68)),
        Rule(keywords: ["english horn", "cor anglais", "corno inglese"], instrument: inst("English Horn", "wind.reed.english-horn", 69)),
        Rule(keywords: ["oboe", "hautbois"], instrument: inst("Oboe", "wind.reed.oboe", 68)),
        Rule(keywords: ["clarinet", "clarinetto", "klarinette", "clarinette"], instrument: inst("Clarinet", "wind.reed.clarinet", 71)),
        Rule(keywords: ["bassoon", "fagotto", "fagott", "basson"], instrument: inst("Bassoon", "wind.reed.bassoon", 70)),
        Rule(keywords: ["piccolo", "ottavino"], instrument: inst("Piccolo", "wind.flutes.flute.piccolo", 72)),
        Rule(keywords: ["flute", "flauto", "flöte", "flote", "traverso"], instrument: inst("Flute", "wind.flutes.flute", 73)),
        Rule(keywords: ["recorder", "blockflöte", "flauto dolce"], instrument: inst("Recorder", "wind.flutes.recorder", 74)),
        // Strings — specific before generic.
        Rule(keywords: ["violoncello", "violoncelle", "cello"], instrument: inst("Cello", "strings.cello", 42)),
        Rule(keywords: ["contrabass", "double bass", "kontrabass", "contrabbasso", "contrebasse", "violone"], instrument: inst("Contrabass", "strings.contrabass", 43)),
        Rule(keywords: ["viola", "bratsche", "alto"], instrument: inst("Viola", "strings.viola", 41)),
        Rule(keywords: ["violin", "violino", "violine", "violon", "viol"], instrument: inst("Violin", "strings.violin", 40)),
        Rule(keywords: ["harp", "harfe", "arpa", "harpe"], instrument: inst("Harp", "pluck.harp", 46)),
        // Continuo / bass lines (figured bass is not realised; we voice the bass line).
        Rule(keywords: ["continuo", "basso continuo", "b.c."], instrument: inst("Continuo", "strings.cello", 42)),
        Rule(keywords: ["basso", "bass"], instrument: inst("Bass", "strings.contrabass", 43)),
        // Brass.
        Rule(keywords: ["trumpet", "tromba", "trompete", "trompette"], instrument: inst("Trumpet", "brass.trumpet", 56)),
        Rule(keywords: ["trombone", "posaune", "trombone"], instrument: inst("Trombone", "brass.trombone", 57)),
        Rule(keywords: ["french horn", "corno", "horn", "cor"], instrument: inst("Horn", "brass.french-horn", 60)),
        Rule(keywords: ["tuba"], instrument: inst("Tuba", "brass.tuba", 58)),
        // Keyboards.
        Rule(keywords: ["harpsichord", "cembalo", "clavecin", "clavicembalo"], instrument: inst("Harpsichord", "keyboard.harpsichord", 6)),
        Rule(keywords: ["organ", "orgel", "organo", "orgue"], instrument: inst("Organ", "keyboard.organ", 19)),
        Rule(keywords: ["piano", "klavier", "pianoforte"], instrument: inst("Piano", "keyboard.piano", 0)),
        // Voices.
        Rule(keywords: ["soprano", "sopran"], instrument: inst("Soprano", "voice.soprano", 53)),
        Rule(keywords: ["alto", "contralto"], instrument: inst("Alto", "voice.alto", 53)),
        Rule(keywords: ["tenor", "ténor"], instrument: inst("Tenor", "voice.tenor", 53)),
        Rule(keywords: ["voice", "canto", "voce", "stimme", "coro", "choir"], instrument: inst("Voice", "voice", 53)),
    ]

    /// Curated, de-duplicated instrument list for the arrangement editor's
    /// instrument picker (one representative per General MIDI voice we support).
    static let pickerPresets: [Instrument] = [
        inst("Piano", "keyboard.piano", 0),
        inst("Harpsichord", "keyboard.harpsichord", 6),
        inst("Organ", "keyboard.organ", 19),
        inst("Violin", "strings.violin", 40),
        inst("Viola", "strings.viola", 41),
        inst("Cello", "strings.cello", 42),
        inst("Contrabass", "strings.contrabass", 43),
        inst("Harp", "pluck.harp", 46),
        inst("Trumpet", "brass.trumpet", 56),
        inst("Trombone", "brass.trombone", 57),
        inst("Tuba", "brass.tuba", 58),
        inst("Horn", "brass.french-horn", 60),
        inst("Oboe", "wind.reed.oboe", 68),
        inst("Oboe d'amore", "wind.reed.oboe.oboe-damore", 68),
        inst("English Horn", "wind.reed.english-horn", 69),
        inst("Bassoon", "wind.reed.bassoon", 70),
        inst("Clarinet", "wind.reed.clarinet", 71),
        inst("Piccolo", "wind.flutes.flute.piccolo", 72),
        inst("Flute", "wind.flutes.flute", 73),
        inst("Recorder", "wind.flutes.recorder", 74),
        inst("Voice", "voice", 53),
    ]

    /// Best-matching instrument for a raw OCR'd label, or nil if none recognised.
    static func match(label: String) -> Instrument? {
        let normalised = normalise(label)
        guard !normalised.isEmpty else { return nil }
        for rule in rules where rule.keywords.contains(where: { normalised.contains($0) }) {
            return rule.instrument
        }
        return nil
    }

    /// Lowercase, fold diacritics where helpful but keep the umlaut forms our
    /// keywords also list, collapse whitespace. Roman-numeral / digit part
    /// suffixes ("I", "II", "1") don't interfere because matching is substring.
    private static func normalise(_ label: String) -> String {
        let lowered = label
            .lowercased()
            .replacingOccurrences(of: "\n", with: " ")
        // Collapse runs of whitespace to single spaces.
        let collapsed = lowered.split(whereSeparator: { $0 == " " || $0 == "\t" }).joined(separator: " ")
        return collapsed.trimmingCharacters(in: .whitespaces)
    }
}
