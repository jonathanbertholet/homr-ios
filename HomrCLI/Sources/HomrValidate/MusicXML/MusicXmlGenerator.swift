import Foundation

// MARK: - MusicXML generator
//
// Direct port of `homr/music_xml_generator.py`. Turns a recognised, cleaned-up
// stream of `EncodedSymbol`s (one stream per detected voice/part) into a
// MusicXML 4.0 partwise document. The Python original builds the document with
// the `musicxml` package; here every `mxl.XML*` element becomes an `XMLNode`
// (see `MusicXmlBuilder.swift`) and the element nesting / attribute names are
// reproduced exactly.
//
// Snake_case names are kept to mirror the Python source 1:1, matching the
// convention already used across the ported transformer files.

// MARK: - Small numeric helpers

extension Fraction {
    /// Truncate toward zero, mirroring Python's `int(Fraction)`.
    ///
    /// Swift integer division truncates toward zero, and `Fraction` always keeps
    /// a positive denominator, so `numerator / denominator` reproduces
    /// `int(Fraction(numerator, denominator))` for both signs.
    var truncatedToInt: Int {
        numerator / denominator
    }
}

/// Euclidean GCD on absolute values, mirroring `math.gcd`.
private func gcd_int(_ a: Int, _ b: Int) -> Int {
    var x = abs(a)
    var y = abs(b)
    while y != 0 {
        (x, y) = (y, x % y)
    }
    return x
}

/// Median of a list of fractions, reproducing `np.median` on a list of
/// `Fraction` objects (object dtype): the middle value for odd counts, the
/// exact mean of the two central values for even counts. Only called with a
/// non-empty list.
private func median_fraction(_ values: [Fraction]) -> Fraction {
    let sorted = values.sorted()
    let n = sorted.count
    if n % 2 == 1 {
        return sorted[n / 2]
    }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / Fraction(2)
}

// MARK: - Conversion state

/// Mutable per-part bookkeeping carried through measure generation.
///
/// Port of Python `ConversionState`. A class because the state (current beats,
/// tremolo toggle, volta numbering) is mutated as symbols are processed.
final class ConversionState {
    /// Fallback duration (in divisions) for whole-measure rests; starts at one
    /// 4/4 measure and is updated by every time signature.
    var beats: Int
    /// Divisions per whole note used for all `<duration>` values.
    let division: Int
    /// Median measure length (in whole notes) used to derive the time signature.
    let nominator: Fraction
    /// Alternating "start"/"stop" used to pair up tremolo marks.
    var tremolo_state: String
    /// Current volta (1st/2nd ending) number.
    var volta_number: Int
    /// Measure number where the last volta ended (for consecutive-volta detection).
    var last_volta_measure: Int

    init(_ division: Int, _ nominator: Fraction) {
        self.beats = 4 * Constants.durationOfQuarter
        self.division = division
        self.nominator = nominator
        self.tremolo_state = "stop"
        self.volta_number = 1
        self.last_volta_measure = -10
    }

    /// Bump the volta number when this volta directly follows the previous one,
    /// else restart at 1. Port of `start_volta`.
    func start_volta(_ measure_no: Int) -> Int {
        if measure_no == last_volta_measure + 1 {
            volta_number += 1
        } else {
            volta_number = 1
        }
        return volta_number
    }

    /// Record where a volta ended. Port of `stop_volta`.
    func stop_volta(_ measure_no: Int) -> Int {
        last_volta_measure = measure_no
        return volta_number
    }

    /// Flip and return the tremolo state. Port of `toggle_tremolo_state`.
    func toggle_tremolo_state() -> String {
        tremolo_state = tremolo_state == "start" ? "stop" : "start"
        return tremolo_state
    }
}

// MARK: - Symbol chord

/// A vertical group of simultaneous symbols (a chord, or a single bar
/// line/clef/etc.), optionally tagged with a tuplet bracket mark.
///
/// Port of Python `SymbolChord`. A class because `TupletParser` mutates
/// `tuplet_mark` in place on shared chord instances.
final class SymbolChord {
    var symbols: [EncodedSymbol]
    /// "" / "start" / "stop" — the tuplet bracket boundary, if any.
    var tuplet_mark: String

    init(_ symbols: [EncodedSymbol], tuplet_mark: String = "") {
        self.symbols = symbols
        self.tuplet_mark = tuplet_mark
    }

    /// True if this group is a bar line or repeat. Port of `is_barline`.
    func is_barline() -> Bool {
        guard let first = symbols.first else { return false }
        let first_rhythm = first.rhythm
        return first_rhythm.contains("barline") || first_rhythm.contains("repeat")
    }

    /// The chord's onset duration = shortest note/rest length (0 if none).
    /// Port of `get_duration`.
    func get_duration() -> Fraction {
        let notes_rests = symbols
            .filter { $0.rhythm.hasPrefix("note") || $0.rhythm.hasPrefix("rest") }
            .map { $0.get_duration().fraction }
        return notes_rests.min() ?? Fraction(0)
    }

    /// Split into upper-staff and lower-staff chords for grand-staff parts.
    ///
    /// Port of `into_positions`. Symbols default to the lower staff unless
    /// explicitly `"upper"`. If the lower staff holds only rests, the two
    /// chords are swapped so the upper (melodic) staff is emitted first. Empty
    /// chords are dropped.
    func into_positions() -> [SymbolChord] {
        var upper: [EncodedSymbol] = []
        var lower: [EncodedSymbol] = []
        var lower_is_only_rest = true
        for symbol in symbols {
            if symbol.position == "upper" {
                upper.append(symbol)
            } else {
                lower.append(symbol)
                lower_is_only_rest = lower_is_only_rest && symbol.rhythm.hasPrefix("rest")
            }
        }
        var chords = [
            SymbolChord(upper, tuplet_mark: tuplet_mark),
            SymbolChord(lower, tuplet_mark: tuplet_mark),
        ]
        if lower_is_only_rest {
            chords = [chords[1], chords[0]]
        }
        return chords.filter { !$0.symbols.isEmpty }
    }
}

// MARK: - Generator arguments

/// Options controlling the generated document.
///
/// Port of Python `XmlGeneratorArguments`. `largePage` emits an oversized page
/// layout (good for electronic display); `metronome`/`tempo` add a metronome
/// direction to the first part.
struct XmlGeneratorArguments {
    let largePage: Bool?
    let metronome: Int?
    let tempo: Int?

    init(largePage: Bool? = nil, metronome: Int? = nil, tempo: Int? = nil) {
        self.largePage = largePage
        self.metronome = metronome
        self.tempo = tempo
    }
}

// MARK: - Public entry point

/// Generate a MusicXML document from per-part symbol streams.
///
/// Public wrapper over the internal tree builder, returning a `MusicXmlDocument`
/// (root `<score-partwise>`) that can be serialised or written to disk.
/// Equivalent to Python `generate_xml(args, staffs, title)` plus its `.write`.
func generateXml(
    _ args: XmlGeneratorArguments, staffs: [[EncodedSymbol]], title: String
) -> MusicXmlDocument {
    MusicXmlDocument(root: generate_xml(args, staffs, title))
}

/// Identification/encoding block so validators and apps (e.g. MuseScore) can
/// attribute the file. Port of `build_identification`.
func build_identification() -> XMLNode {
    let ident = XMLNode("identification")
    let enc = XMLNode("encoding")
    enc.addChild(XMLNode("software", value: "homr"))
    ident.addChild(enc)
    return ident
}

/// Build the `<score-partwise>` root: work title, identification, defaults,
/// part list, and one `<part>` per voice. Port of `generate_xml`.
func generate_xml(
    _ args: XmlGeneratorArguments, _ staffs: [[EncodedSymbol]], _ title: String
) -> XMLNode {
    let root = XMLNode("score-partwise")
    root.setAttribute("version", "4.0")
    root.addChild(build_work(title))
    root.addChild(build_identification())
    root.addChild(build_defaults(args))
    let has_two_staves_by_part = staffs.map { _voice_has_two_staves($0) }
    root.addChild(build_part_list(has_two_staves_by_part))
    for (index, staff) in staffs.enumerated() {
        root.addChild(build_part(args, staff, index, has_two_staves_by_part[index]))
    }
    return root
}

/// True if any symbol uses the lower staff (e.g. piano left hand / bass clef).
/// Port of `_voice_has_two_staves`.
func _voice_has_two_staves(_ voice: [EncodedSymbol]) -> Bool {
    voice.contains { $0.position == "lower" }
}

/// Build one `<part>` and its measures. Port of `build_part`.
func build_part(
    _ args: XmlGeneratorArguments, _ voice: [EncodedSymbol], _ index: Int, _ has_two_staves: Bool
) -> XMLNode {
    let part = XMLNode("part")
    part.setAttribute("id", get_part_id(index))
    let is_first_part = index == 0
    let measures = build_measures(args, voice, is_first_part, has_two_staves)
    for measure in measures {
        part.addChild(measure)
    }
    return part
}

// MARK: - Measure construction

/// Convert a voice's symbol stream into a list of `<measure>` elements.
///
/// Port of `build_measures`. Walks chord groups, emitting notes/rests, clefs,
/// key/time signatures, bar lines, repeats and voltas, opening a new measure at
/// each bar line/repeat boundary. Attribute reuse (`build_or_get_attributes`)
/// mirrors the Python state machine exactly, including resetting the pending
/// `attributes` to `nil` each iteration.
func build_measures(
    _ args: XmlGeneratorArguments,
    _ voice: [EncodedSymbol],
    _ is_first_part: Bool,
    _ has_two_staves: Bool = false
) -> [XMLNode] {
    var measures: [XMLNode] = []
    var measure_number = 1
    var current_measure = XMLNode("measure")
    current_measure.setAttribute("number", String(measure_number))

    // Finalise the current measure: rebalance voices, then store it. Captures
    // `current_measure`/`measures` by reference (they are reassigned below).
    func close_current_measure() {
        rebalance_measure_voices(current_measure)
        measures.append(current_measure)
    }

    let groups = add_tuplet_start_stop(group_into_chords(voice))
    let (division, nominator) = find_division_and_time_signature_nominator(groups)
    let state = ConversionState(division, nominator)

    let first_attributes = build_or_get_attributes(current_measure, nil)
    first_attributes.addChild(build_divisions(division))
    if has_two_staves {
        first_attributes.addChild(XMLNode("staves", value: 2))
        first_attributes.addChild(XMLNode("part-symbol", value: "brace"))
    }
    if is_first_part {
        if let direction = build_add_time_direction(args) {
            current_measure.addChild(direction)
        }
    }
    var attributes: XMLNode? = first_attributes
    for (group_no, group) in groups.enumerated() {
        let symbol = group.symbols[0]
        let rhythm = symbol.rhythm
        let last_attributes = attributes
        attributes = nil
        if rhythm.hasPrefix("note") || rhythm.hasPrefix("rest") {
            if group.symbols.count == 1 && rhythm.hasSuffix("m") {
                let attrs = build_or_get_attributes(current_measure, last_attributes)
                attributes = attrs
                build_multi_measure_rest(symbol, attrs)
            } else {
                let staff_positions = group.into_positions()
                for (pos_no, staff_pos) in staff_positions.enumerated() {
                    let chord_duration =
                        pos_no == staff_positions.count - 1
                        ? group.get_duration() : Fraction(0)
                    for note_xml in build_note_chord(staff_pos, state, chord_duration) {
                        current_measure.addChild(note_xml)
                    }
                }
            }
            continue
        }
        if rhythm == "newline" {
            let is_last_measure = group_no == groups.count - 1
            if !is_last_measure {
                let printNode = XMLNode("print")
                printNode.setAttribute("new-system", "yes")
                current_measure.addChild(printNode)
            }
        } else if rhythm.hasPrefix("clef") {
            let attrs = build_or_get_attributes(current_measure, last_attributes, force_new: true)
            attributes = attrs
            for should_be_clef in group.symbols where should_be_clef.rhythm.hasPrefix("clef") {
                build_clef(should_be_clef, attrs)
            }
        } else if rhythm.hasPrefix("keySignature") {
            let attrs = build_or_get_attributes(current_measure, last_attributes)
            attributes = attrs
            build_key(symbol, attrs)
        } else if rhythm.hasPrefix("timeSignature") {
            let attrs = build_or_get_attributes(current_measure, last_attributes)
            attributes = attrs
            build_time_signature(symbol, attrs, state)
        } else if rhythm.contains("barline") {
            if rhythm != "barline" {
                // Standard bar lines don't need extra handling.
                let barline = build_or_get_barline(current_measure, "right")
                build_barline_style(symbol, barline)
            }
            close_current_measure()
            measure_number += 1
            current_measure = XMLNode("measure")
            current_measure.setAttribute("number", String(measure_number))
        } else if rhythm == "repeatStart" {
            close_current_measure()
            measure_number += 1
            current_measure = XMLNode("measure")
            current_measure.setAttribute("number", String(measure_number))

            let barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)
        } else if rhythm == "repeatEnd" {
            let barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = XMLNode("measure")
            current_measure.setAttribute("number", String(measure_number))
        } else if rhythm == "repeatEndStart" {
            let barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatEnd"), barline)

            close_current_measure()
            measure_number += 1
            current_measure = XMLNode("measure")
            current_measure.setAttribute("number", String(measure_number))
            let barline2 = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatStart"), barline2)
        } else if rhythm.hasPrefix("voltaStart") {
            let volta_number = state.start_volta(measure_number)
            let barline = build_or_get_barline(current_measure, "left")
            build_barline_ending(symbol, barline, volta_number)
        } else if rhythm.hasPrefix("voltaStop") || rhythm.hasPrefix("voltaDiscontinue") {
            let volta_number = state.stop_volta(measure_number)
            let barline = build_or_get_barline(current_measure, "right")
            build_barline_ending(symbol, barline, volta_number)
        } else {
            eprint("Symbol isn't supported yet ", symbol)
        }
    }

    if !current_measure.children.isEmpty {
        close_current_measure()
    }
    return measures
}

// MARK: - Header / metadata builders

/// Build the `<work>` block with the score title. Port of `build_work`.
func build_work(_ title_text: String) -> XMLNode {
    let work = XMLNode("work")
    let title = XMLNode("work-title")
    title.value = title_text
    work.addChild(title)
    return work
}

/// Build `<defaults>`. With `largePage`, emit an oversized page layout so we
/// only break systems at each detected staff (good for electronic formats;
/// print output may need scaling). Port of `build_defaults`.
func build_defaults(_ args: XmlGeneratorArguments) -> XMLNode {
    if !(args.largePage ?? false) {
        return XMLNode("defaults")
    }
    // Units are tenths (MusicXML page-height/page-width). Larger than A4/letter
    // so a single system per detected staff fits without page breaks.
    let page_width = 110
    let page_height = 300
    let defaults = XMLNode("defaults")
    let page_layout = XMLNode("page-layout")
    page_layout.addChild(XMLNode("page-height", value: page_height))
    page_layout.addChild(XMLNode("page-width", value: page_width))
    defaults.addChild(page_layout)
    return defaults
}

/// Part id for part `index` (0-based): `P1`, `P2`, ... Port of `get_part_id`.
func get_part_id(_ index: Int) -> String {
    "P" + String(index + 1)
}

/// Classify a part by staff layout, returning
/// `(part_name, instrument_name, instrument_sound, midi_program)`.
///
/// Single-staff -> Voice, two-staff -> Piano. `midi_program` is 1-based
/// (1 = Acoustic Grand Piano, 54 = Voice Oohs). Port of `_part_metadata`.
func _part_metadata(_ has_two_staves: Bool) -> (String, String, String, Int) {
    if has_two_staves {
        return ("Piano", "Piano", "keyboard.piano", 1)
    }
    return ("Voice", "Voice", "voice", 54)
}

/// Build `<part-list>` with a `<score-part>` (name, instrument, MIDI) per part.
/// Port of `build_part_list`.
func build_part_list(_ has_two_staves_by_part: [Bool]) -> XMLNode {
    let part_list = XMLNode("part-list")
    for (part, has_two_staves) in has_two_staves_by_part.enumerated() {
        let part_id = get_part_id(part)
        let (part_name_str, instrument_name_str, instrument_sound_str, midi_program) =
            _part_metadata(has_two_staves)
        let score_part = XMLNode("score-part")
        score_part.setAttribute("id", part_id)
        score_part.addChild(XMLNode("part-name", value: part_name_str))
        let score_instrument = XMLNode("score-instrument")
        score_instrument.setAttribute("id", part_id + "-I1")
        score_instrument.addChild(XMLNode("instrument-name", value: instrument_name_str))
        score_instrument.addChild(XMLNode("instrument-sound", value: instrument_sound_str))
        score_part.addChild(score_instrument)
        let midi_instrument = XMLNode("midi-instrument")
        midi_instrument.setAttribute("id", part_id + "-I1")
        midi_instrument.addChild(XMLNode("midi-channel", value: part + 1))
        midi_instrument.addChild(XMLNode("midi-program", value: midi_program))
        midi_instrument.addChild(XMLNode("volume", value: 100))
        midi_instrument.addChild(XMLNode("pan", value: 0))
        score_part.addChild(midi_instrument)
        part_list.addChild(score_part)
    }
    return part_list
}

// MARK: - Attributes / bar lines

/// Reuse the last `<attributes>` element (so e.g. key+time share one block) or
/// create a fresh one. `force_new` always creates a new element (used for
/// clefs). Port of `build_or_get_attributes`.
func build_or_get_attributes(
    _ measure: XMLNode, _ last_attributes: XMLNode?, force_new: Bool = false
) -> XMLNode {
    if let last = last_attributes, !force_new {
        return last
    }
    let attributes = XMLNode("attributes")
    measure.addChild(attributes)
    return attributes
}

/// Find an existing `<barline>` at `location` in the measure, or create one.
/// Port of `build_or_get_barline`.
func build_or_get_barline(_ measure: XMLNode, _ location: String) -> XMLNode {
    for child in measure.children(named: "barline") where child.attribute("location") == location {
        return child
    }
    let barline = XMLNode("barline")
    barline.setAttribute("location", location)
    measure.addChild(barline)
    return barline
}

/// Build a `<key>` from a `keySignature_<n>` symbol, where `<n>` is the
/// circle-of-fifths integer. Port of `build_key`.
func build_key(_ model_key: EncodedSymbol, _ attributes: XMLNode) {
    let key = XMLNode("key")
    let circle_of_fifth = model_key.rhythm.components(separatedBy: "_")[1]
    let fifth = XMLNode("fifths", value: Int(circle_of_fifth)!)
    attributes.addChild(key)
    key.addChild(fifth)
}

/// Staff number for a symbol: lower = 2, otherwise 1. Port of `get_staff`.
func get_staff(_ symbol: EncodedSymbol) -> Int {
    symbol.position == "lower" ? 2 : 1
}

/// Build a part-global, stable MusicXML voice number.
///
/// Voices are part-global, so staff number alone can merge independent layers.
/// Reserve 4 voices per staff: staff 1 -> 1..4, staff 2 -> 5..8.
/// Port of `get_xml_voice`.
func get_xml_voice(_ staff_num: Int, _ rhythmic_layer: Int) -> Int {
    (staff_num - 1) * 4 + rhythmic_layer + 1
}

/// A run of notes occupying `[start, end)` on a staff, for voice assignment.
/// Port of the `TimedNoteEvent` dataclass. A class so `notes` can grow as chord
/// tones are merged.
private final class TimedNoteEvent {
    let staff_num: Int
    let start: Int
    let end: Int
    var notes: [XMLNode]

    init(_ staff_num: Int, _ start: Int, _ end: Int, _ notes: [XMLNode]) {
        self.staff_num = staff_num
        self.start = start
        self.end = end
        self.notes = notes
    }
}

/// Assign stable, non-overlapping voices per staff across a whole measure.
///
/// Port of `rebalance_measure_voices`. Walks the measure tracking time via
/// `<backup>`/`<forward>`/`<duration>`, groups notes into timed events (merging
/// chord tones), then greedily assigns the lowest free local voice per staff
/// and rewrites each note's `<voice>` value.
func rebalance_measure_voices(_ measure: XMLNode) {
    var timed_events: [TimedNoteEvent] = []
    var current_time = 0
    var last_note_start = 0
    for child in measure.children {
        if child.name == "backup" {
            let durations = child.children(named: "duration")
            if let first = durations.first, let value = Int(first.value ?? "") {
                current_time -= value
            }
            continue
        }
        if child.name == "forward" {
            let durations = child.children(named: "duration")
            if let first = durations.first, let value = Int(first.value ?? "") {
                current_time += value
            }
            continue
        }
        if child.name != "note" {
            continue
        }

        let duration_nodes = child.children(named: "duration")
        let duration = duration_nodes.first.flatMap { Int($0.value ?? "") } ?? 0
        let staff_nodes = child.children(named: "staff")
        let staff_num = staff_nodes.first.flatMap { Int($0.value ?? "") } ?? 1
        let is_chord_tone = !child.children(named: "chord").isEmpty
        let start = is_chord_tone ? last_note_start : current_time
        let end = start + duration
        if is_chord_tone, let last = timed_events.last,
            last.staff_num == staff_num, last.start == start, last.end == end
        {
            last.notes.append(child)
        } else if is_chord_tone {
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))
        } else {
            last_note_start = start
            current_time += duration
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))
        }
    }

    var by_staff: [Int: [TimedNoteEvent]] = [:]
    var staff_order: [Int] = []
    for event in timed_events {
        if by_staff[event.staff_num] == nil {
            staff_order.append(event.staff_num)
        }
        by_staff[event.staff_num, default: []].append(event)
    }

    for staff_num in staff_order {
        let events = by_staff[staff_num]!
        // Stable sort by (start, end) — preserves input order on ties, matching
        // Python's stable `sorted`.
        let sorted_events =
            events.enumerated()
            .sorted { lhs, rhs in
                if lhs.element.start != rhs.element.start {
                    return lhs.element.start < rhs.element.start
                }
                if lhs.element.end != rhs.element.end {
                    return lhs.element.end < rhs.element.end
                }
                return lhs.offset < rhs.offset
            }
            .map { $0.element }
        var active: [(end: Int, voice: Int)] = []
        for event in sorted_events {
            active = active.filter { $0.end > event.start }
            let used_voices = Set(active.map { $0.voice })
            var voice_no = 1
            while used_voices.contains(voice_no) {
                voice_no += 1
            }
            active.append((event.end, voice_no))
            let xml_voice = String(get_xml_voice(staff_num, voice_no - 1))
            for note in event.notes {
                if let voice_node = note.children(named: "voice").first {
                    voice_node.value = xml_voice
                }
            }
        }
    }
}

/// Build a `<clef>` from a `clef_<sign><line>` symbol (e.g. `clef_G2` ->
/// sign G, line 2). Port of `build_clef`.
func build_clef(_ model_clef: EncodedSymbol, _ attributes: XMLNode) {
    let sign_and_line = Array(model_clef.rhythm.components(separatedBy: "_")[1])
    let sign = String(sign_and_line[0])
    let line = String(sign_and_line[1])
    let clef = XMLNode("clef")
    clef.setAttribute("number", get_staff(model_clef))
    attributes.addChild(clef)
    clef.addChild(XMLNode("sign", value: sign))
    clef.addChild(XMLNode("line", value: Int(line)!))
}

/// Build a `<time>` from a `timeSignature/<denominator>` symbol. `beats` =
/// median measure length × denominator (at least 1). Port of
/// `build_time_signature`.
func build_time_signature(
    _ model_time_signature: EncodedSymbol, _ attributes: XMLNode, _ state: ConversionState
) {
    let time = XMLNode("time")
    let denominator = model_time_signature.rhythm.components(separatedBy: "/")[1]
    attributes.addChild(time)
    let beats = max((state.nominator * Fraction(Int(denominator)!)).truncatedToInt, 1)
    time.addChild(XMLNode("beats", value: String(beats)))
    time.addChild(XMLNode("beat-type", value: denominator))
    state.beats = beats
}

/// Add a `<bar-style>` to a bar line (heavy-heavy for a bold double bar line,
/// else light-light). Port of `build_barline_style`.
func build_barline_style(_ barline: EncodedSymbol, _ xml: XMLNode) {
    let style_value = barline.rhythm == "bolddoublebarline" ? "heavy-heavy" : "light-light"
    xml.addChild(XMLNode("bar-style", value: style_value))
}

/// Add an `<ending>` (volta) start/stop/discontinue to a bar line.
/// Port of `build_barline_ending`.
func build_barline_ending(_ volta: EncodedSymbol, _ xml: XMLNode, _ volta_number: Int) {
    let ending = XMLNode("ending")
    if volta.rhythm.hasPrefix("voltaStart") {
        ending.setAttribute("type", "start")
    } else if volta.rhythm.hasPrefix("voltaStop") {
        ending.setAttribute("type", "stop")
    } else if volta.rhythm.hasPrefix("voltaDiscontinue") {
        ending.setAttribute("type", "discontinue")
    } else {
        preconditionFailure("Unknown ending " + String(describing: volta))
    }
    ending.setAttribute("number", String(volta_number))
    xml.addChild(ending)
}

/// Add a `<repeat>` (forward for repeatStart, else backward) to a bar line,
/// skipping if one already exists. Port of `build_repeat`.
func build_repeat(_ barline: EncodedSymbol, _ xml: XMLNode) {
    if !xml.children(named: "repeat").isEmpty {
        eprint("barline already has a repeat")
        return
    }
    let repeatNode = XMLNode("repeat")
    let direction = barline.rhythm == "repeatStart" ? "forward" : "backward"
    repeatNode.setAttribute("direction", direction)
    xml.addChild(repeatNode)
}

// MARK: - Note construction

/// Pitch alteration per lift token. Port of `LIFT_TO_ALTER`.
let LIFT_TO_ALTER: [String: Int] = [
    "N": 0,
    "#": 1,
    "##": 2,
    "b": -1,
    "bb": -2,
]

/// Note-type name per kern base duration. Port of `DURATION_NAMES`.
let DURATION_NAMES: [Int: String] = [
    0: "breve",
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
    128: "128th",
]

/// Build a note's `<notations>` (always added, even if empty), populating
/// articulations, ornaments, fermata/arpeggiate, slur/tie marks and an optional
/// tuplet bracket. Port of `build_articulations`.
///
/// Note: the Python original contains duplicate `arpeggiate`/`fermata` branches
/// that are unreachable (the first match wins); only the reachable behaviour is
/// reproduced here.
func build_articulations(
    _ note: XMLNode, _ articualations: String, _ tuplet_mark: String, _ state: ConversionState
) {
    let notation = XMLNode("notations")
    note.addChild(notation)

    var xml_articulations: [XMLNode] = []
    var xml_ornaments: [XMLNode] = []

    for articulation in articualations.components(separatedBy: "_") {
        switch articulation {
        case "":
            continue
        case nonote:
            eprint("WARNING note without valid articulation", articualations)
        case "fermata":
            notation.addChild(XMLNode("fermata"))
        case "arpeggiate":
            notation.addChild(XMLNode("arpeggiate"))
        case "accent":
            xml_articulations.append(XMLNode("accent"))
        case "staccato":
            xml_articulations.append(XMLNode("staccato"))
        case "staccatissimo":
            xml_articulations.append(XMLNode("staccatissimo"))
        case "tenuto":
            xml_articulations.append(XMLNode("tenuto"))
        case "tremolo":
            let tremolo = XMLNode("tremolo", value: 3)
            tremolo.setAttribute("type", state.toggle_tremolo_state())
            xml_ornaments.append(tremolo)
        case "trill":
            xml_ornaments.append(XMLNode("trill-mark"))
        case "breathMark":
            xml_articulations.append(XMLNode("breath-mark"))
        case "turn":
            xml_ornaments.append(XMLNode("inverted-turn"))
        case "caesura":
            xml_articulations.append(XMLNode("caesura"))
        case "doit":
            xml_articulations.append(XMLNode("doit"))
        case "slurStart":
            let slur = XMLNode("slur")
            slur.setAttribute("type", "start")
            notation.addChild(slur)
        case "slurStop":
            let slur = XMLNode("slur")
            slur.setAttribute("type", "stop")
            notation.addChild(slur)
        case "tieStart":
            let tied = XMLNode("tied")
            tied.setAttribute("type", "start")
            notation.addChild(tied)
        case "tieStop":
            let tied = XMLNode("tied")
            tied.setAttribute("type", "stop")
            notation.addChild(tied)
        default:
            preconditionFailure("Unsupported articulation " + articulation)
        }
    }

    if tuplet_mark != "" {
        let tuplet_xml = XMLNode("tuplet")
        tuplet_xml.setAttribute("type", tuplet_mark)
        notation.addChild(tuplet_xml)
    }

    if !xml_articulations.isEmpty {
        let parent = XMLNode("articulations")
        for child in xml_articulations {
            parent.addChild(child)
        }
        notation.addChild(parent)
    }

    if !xml_ornaments.isEmpty {
        let parent = XMLNode("ornaments")
        for child in xml_ornaments {
            parent.addChild(child)
        }
        notation.addChild(parent)
    }
}

/// Add `<slur>` marks to a note, reusing its `<notations>` if present.
///
/// For `slurStart_slurStop` the stop is added before the start (otherwise the
/// slur would start and immediately stop). Port of `build_slurs`.
func build_slurs(_ note: XMLNode, _ slurs: String, _ slur_number: Int) {
    let notation: XMLNode
    if let existing = note.children(named: "notations").first {
        notation = existing
    } else {
        notation = XMLNode("notations")
        note.addChild(notation)
    }

    if slurs == "_" || slurs == "" {
        // No slur.
    } else if slurs == nonote {
        eprint("WARNING note without valid articulation", slurs)
    } else if slurs == "slurStart" {
        let slur = XMLNode("slur")
        slur.setAttribute("type", "start")
        slur.setAttribute("number", slur_number)
        notation.addChild(slur)
    } else if slurs == "slurStop" {
        let slur = XMLNode("slur")
        slur.setAttribute("type", "stop")
        slur.setAttribute("number", slur_number)
        notation.addChild(slur)
    } else if slurs == "slurStart_slurStop" {
        // Stop first, then start, so the slur spans across this note.
        let stop = XMLNode("slur")
        stop.setAttribute("type", "stop")
        stop.setAttribute("number", slur_number)
        notation.addChild(stop)
        let start = XMLNode("slur")
        start.setAttribute("type", "start")
        start.setAttribute("number", slur_number)
        notation.addChild(start)
    } else {
        preconditionFailure("Unsupported slur " + slurs)
    }
}

/// Build a single `<note>` (or rest). Port of `build_note_or_rest`.
///
/// Handles chord membership, pitch/rest, accidental (`<alter>`), grace notes,
/// note `<type>` and `<duration>` (= fraction × divisions), staff, voice, dots,
/// tuplet `<time-modification>`, and finally notations (articulations + slurs).
func build_note_or_rest(
    _ model_note: EncodedSymbol,
    _ rhythmic_layer: Int,
    _ is_chord: Bool,
    _ state: ConversionState,
    _ tuplet_mark: String
) -> XMLNode {
    let note = XMLNode("note")
    if is_chord {
        note.addChild(XMLNode("chord"))
    }
    let model_pitch = model_note.pitch
    let model_duration = model_note.get_duration()
    if model_pitch == empty {
        let rest = XMLNode("rest")
        if model_duration.fraction.numerator == 0 {
            rest.setAttribute("measure", "yes")
        }
        note.addChild(rest)
    } else if model_pitch == nonote {
        eprint("WARNING note without pitch", model_note)
        note.addChild(XMLNode("rest"))
    } else {
        let pitch = XMLNode("pitch")
        let chars = Array(model_pitch)
        pitch.addChild(XMLNode("step", value: String(chars[0])))
        pitch.addChild(XMLNode("octave", value: Int(String(chars[1]))!))
        if model_note.lift == nonote {
            eprint("WARNING note with invalid lift", model_note)
        } else if model_note.lift != empty {
            pitch.addChild(XMLNode("alter", value: LIFT_TO_ALTER[model_note.lift]!))
        }
        note.addChild(pitch)
    }

    if model_note.rhythm.contains("G") {
        note.addChild(XMLNode("grace"))
        let base_duration = model_duration.kern
        note.addChild(XMLNode("type", value: DURATION_NAMES[base_duration]!))
    } else if model_duration.fraction.numerator > 0 {
        let base_duration = model_duration.kern == 0 ? 1 : model_duration.kern
        note.addChild(XMLNode("type", value: DURATION_NAMES[base_duration]!))
        note.addChild(
            XMLNode("duration", value: (model_duration.fraction * Fraction(state.division)).truncatedToInt)
        )
    } else {
        note.addChild(XMLNode("type", value: DURATION_NAMES[0]!))
        note.addChild(XMLNode("duration", value: state.beats))
    }

    let staff_num = get_staff(model_note)
    let slur_number = staff_num
    note.addChild(XMLNode("staff", value: staff_num))
    note.addChild(XMLNode("voice", value: String(get_xml_voice(staff_num, rhythmic_layer))))
    for _ in 0..<model_duration.dots {
        note.addChild(XMLNode("dot"))
    }
    if model_duration.actualNotes != model_duration.normalNotes {
        let time_modification = XMLNode("time-modification")
        time_modification.addChild(XMLNode("actual-notes", value: model_duration.actualNotes))
        time_modification.addChild(XMLNode("normal-notes", value: model_duration.normalNotes))
        note.addChild(time_modification)
        build_articulations(note, model_note.articulation, tuplet_mark, state)
        build_slurs(note, model_note.slur, slur_number)
    } else {
        build_articulations(note, model_note.articulation, "", state)
        build_slurs(note, model_note.slur, slur_number)
    }

    return note
}

/// Add a multi-measure rest `<measure-style>` from a `rest_<n>m` symbol.
/// Port of `build_multi_measure_rest`.
func build_multi_measure_rest(_ symbol: EncodedSymbol, _ attributes: XMLNode) {
    if !attributes.children(named: "measure-style").isEmpty {
        eprint("Measure already has a multi rest")
        return
    }
    let raw = symbol.rhythm.components(separatedBy: "_")[1].replacingOccurrences(of: "m", with: "")
    let duration = Int(raw)!
    let style = XMLNode("measure-style")
    style.addChild(XMLNode("multiple-rest", value: duration))
    attributes.addChild(style)
}

/// Build all `<note>` elements for one staff-position chord, with `<backup>`
/// elements separating distinct rhythmic layers and a final `<backup>` to
/// realign to the chord's onset. Port of `build_note_chord`.
func build_note_chord(
    _ note_chord: SymbolChord, _ state: ConversionState, _ chord_duration: Fraction
) -> [XMLNode] {
    let by_duration = _group_notes(note_chord.symbols)
    var result: [XMLNode] = []
    var final_duration = Fraction(0)
    let sorted_durations = by_duration.keys.sorted()
    for (i, group_duration) in sorted_durations.enumerated() {
        var is_first = true
        for note_loop in by_duration[group_duration]! {
            result.append(
                build_note_or_rest(note_loop, i, !is_first, state, note_chord.tuplet_mark)
            )
            is_first = false
        }
        if i != sorted_durations.count - 1 && group_duration > Fraction(0) {
            let backup = XMLNode("backup")
            backup.addChild(
                XMLNode("duration", value: (group_duration * Fraction(state.division)).truncatedToInt)
            )
            result.append(backup)
        }

        final_duration = group_duration
    }

    // Reset the position to match the chord position.
    if chord_duration < final_duration {
        let backup = XMLNode("backup")
        backup.addChild(
            XMLNode(
                "duration",
                value: ((final_duration - chord_duration) * Fraction(state.division)).truncatedToInt
            )
        )
        result.append(backup)
    }
    return result
}

/// Group a chord's notes by their effective duration: grace notes -> 0,
/// whole-measure rests -> the chord's max duration, others -> their fraction.
/// Port of `_group_notes`.
func _group_notes(_ notes: [EncodedSymbol]) -> [Fraction: [EncodedSymbol]] {
    var groups_by_duration: [Fraction: [EncodedSymbol]] = [:]
    let max_duration = notes.map { $0.get_duration().fraction }.max() ?? Fraction(0)
    for note in notes {
        let duration = note.get_duration()
        let is_grace = note.rhythm.contains("G")
        let fraction: Fraction
        if is_grace {
            fraction = Fraction(0)
        } else if duration.fraction.numerator == 0 {
            // Whole measure rest.
            fraction = max_duration
        } else {
            fraction = duration.fraction
        }
        groups_by_duration[fraction, default: []].append(note)
    }
    return groups_by_duration
}

// MARK: - Directions / divisions

/// Build a metronome `<direction>` (beat-unit quarter + per-minute), plus a
/// `<sound tempo=...>`. Returns nil when no metronome is requested.
/// Port of `build_add_time_direction`.
func build_add_time_direction(_ args: XmlGeneratorArguments) -> XMLNode? {
    guard let metronome = args.metronome, metronome != 0 else {
        return nil
    }
    let direction = XMLNode("direction")
    let direction_type = XMLNode("direction-type")
    direction.addChild(direction_type)
    let metronomeNode = XMLNode("metronome")
    direction_type.addChild(metronomeNode)
    metronomeNode.addChild(XMLNode("beat-unit", value: "quarter"))
    metronomeNode.addChild(XMLNode("per-minute", value: String(metronome)))
    let sound = XMLNode("sound")
    if let tempo = args.tempo, tempo != 0 {
        sound.setAttribute("tempo", tempo)
    } else {
        sound.setAttribute("tempo", metronome)
    }
    direction.addChild(sound)
    return direction
}

/// Smallest divisions value so every duration is an integer multiple (LCM of
/// the positive durations' denominators). Port of `find_common_division`.
func find_common_division(_ durations: [Fraction]) -> Int {
    func lcm(_ a: Int, _ b: Int) -> Int {
        abs(a * b) / gcd_int(a, b)
    }

    let denominators = durations.filter { $0 > Fraction(0) }.map { $0.denominator }
    guard let first = denominators.first else {
        return 1
    }
    var common = first
    for d in denominators.dropFirst() {
        common = lcm(common, d)
    }
    return common
}

/// Derive the divisions value and the time-signature nominator (median measure
/// length in whole notes). Port of `find_division_and_time_signature_nominator`.
func find_division_and_time_signature_nominator(_ voice: [SymbolChord]) -> (Int, Fraction) {
    var durations: [Fraction] = [Fraction(1, 4)]
    var duration_in_measure = Fraction(0)
    var measure_duration: [Fraction] = []
    for chord in voice {
        if chord.is_barline() && duration_in_measure > Fraction(0) {
            measure_duration.append(duration_in_measure)
            duration_in_measure = Fraction(0)
        } else {
            let duration = chord.get_duration()
            if duration > Fraction(0) {
                durations.append(duration)
                duration_in_measure = duration_in_measure + duration
            }
        }
    }

    if duration_in_measure > Fraction(0) {
        measure_duration.append(duration_in_measure)
        duration_in_measure = Fraction(0)
    }

    if measure_duration.isEmpty {
        return (find_common_division(durations), Fraction(1))
    }

    let nominator = median_fraction(measure_duration)
    return (find_common_division(durations), nominator)
}

/// Group a voice into chords (sorted within each chord). Port of
/// `group_into_chords`.
func group_into_chords(_ voice: [EncodedSymbol]) -> [SymbolChord] {
    sort_token_chords(voice).map { SymbolChord($0) }
}

// MARK: - Tuplet detection

/// Detects complete tuplets within each measure and marks their start/stop
/// chords. Port of the Python `TupletParser`.
enum TupletParser {
    /// Mark tuplet brackets, measure by measure. If a measure's tuplets can't be
    /// completed, its marks are restored (left unmarked). Port of `parse`.
    static func parse(_ groups: [SymbolChord]) -> [SymbolChord] {
        // Split into measures first: if a tuplet in some measure can't be
        // completed, we can skip that measure and continue with the next.
        for measure_groups in split_into_measures(groups) {
            let saved_marks = measure_groups.map { $0.tuplet_mark }
            if add_tuplets(measure_groups) {
                continue
            }
            // Tuplet parsing failed for this measure; restore original marks.
            for (group, mark) in zip(measure_groups, saved_marks) {
                group.tuplet_mark = mark
            }
        }
        return groups
    }

    /// The tuplet duration of the first tuplet note/rest in a chord, if any.
    /// Port of `get_tuplet_duration`.
    static func get_tuplet_duration(_ group: SymbolChord) -> SymbolDuration? {
        for symbol in group.symbols
        where symbol.rhythm.hasPrefix("note") || symbol.rhythm.hasPrefix("rest") {
            let duration = symbol.get_duration()
            if duration.normalNotes != duration.actualNotes {
                return duration
            }
        }
        return nil
    }

    /// Split chords into measures at bar lines. Port of `split_into_measures`.
    static func split_into_measures(_ groups: [SymbolChord]) -> [[SymbolChord]] {
        var measures: [[SymbolChord]] = []
        var current_measure: [SymbolChord] = []
        for group in groups {
            current_measure.append(group)
            if group.is_barline() {
                measures.append(current_measure)
                current_measure = []
            }
        }
        if !current_measure.isEmpty {
            measures.append(current_measure)
        }
        return measures
    }

    /// Scan one measure, marking each complete run of `actual_notes` tuplet
    /// chords (all sharing the same actual/normal ratio) with start/stop.
    /// Returns false if any tuplet run can't be completed. Port of `add_tuplets`.
    static func add_tuplets(_ groups: [SymbolChord]) -> Bool {
        var cursor = 0
        while cursor < groups.count {
            let duration = get_tuplet_duration(groups[cursor])

            // Tuplet not found, skip.
            guard let duration = duration else {
                cursor += 1
                continue
            }

            let start = cursor
            let tuplet_format = (duration.actualNotes, duration.normalNotes)
            let tuplet_size = duration.actualNotes

            // Try to find a complete tuplet.
            while cursor - start < tuplet_size {
                // Three sanity checks first.
                if cursor >= groups.count {
                    return false
                }
                guard let current_duration = get_tuplet_duration(groups[cursor]) else {
                    return false
                }
                let current_format = (current_duration.actualNotes, current_duration.normalNotes)
                if current_format != tuplet_format {
                    return false
                }
                // Confident the note is within the tuplet.
                cursor += 1
            }

            groups[start].tuplet_mark = "start"
            groups[cursor - 1].tuplet_mark = "stop"
        }

        return true
    }
}

/// Mark tuplet start/stop on the chord groups. Port of `add_tuplet_start_stop`.
func add_tuplet_start_stop(_ groups: [SymbolChord]) -> [SymbolChord] {
    TupletParser.parse(groups)
}

/// Build `<divisions>` = divisions-per-whole-note / 4 (i.e. divisions per
/// quarter note). Port of `build_divisions`.
func build_divisions(_ division: Int) -> XMLNode {
    // The divisions element indicates how many divisions per quarter(!) note
    // are used to express a note's duration.
    XMLNode("divisions", value: division / 4)
}
