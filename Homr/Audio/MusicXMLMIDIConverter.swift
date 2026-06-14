import Foundation

// MARK: - MusicXML → MIDI conversion
//
// Walks the in-memory MusicXML element tree (`XMLNode`, the same structure the
// generator builds) and produces a `MIDISequence`. Working off the tree avoids
// re-parsing the serialised string and keeps the mapping exact.
//
// Timing model (standard MusicXML semantics):
//   * `<divisions>` (inside `<attributes>`) is the number of duration units per
//     quarter note. We convert every `<duration>` to quarter-note beats
//     immediately, so a mid-piece divisions change is handled correctly.
//   * A running `cursor` (in beats) tracks the current insertion point within a
//     measure. A plain note advances it by its duration; `<backup>` rewinds it
//     and `<forward>` advances it (this is how multi-voice / grand-staff parts
//     are encoded).
//   * A `<chord>` note shares the onset of the previous note (no extra advance).
//   * The next measure starts at the furthest point any voice reached, so
//     measures never overlap even with backups.
//
// Each `<part>` is independent and restarts at beat 0 on its own MIDI channel,
// so multiple voices play simultaneously.
//
// Deliberate v1 simplifications (documented, not silent):
//   * Repeats/voltas are NOT expanded — the score plays through once.
//   * Grace notes (zero duration) are skipped.
//   * Ties are not merged; a tied note simply re-articulates (audible but not
//     held). Slurs/articulations don't affect playback.
enum MusicXMLMIDIConverter {
    /// Default tempo when the document carries no `<sound tempo=…>` (the
    /// generator only emits one when a metronome is requested, which we don't).
    static let defaultTempoBPM: Double = 90

    /// Convert a generated document into a playable/exportable MIDI sequence.
    static func convert(_ document: MusicXmlDocument, defaultTempo: Double = defaultTempoBPM) -> MIDISequence {
        convert(root: document.root, defaultTempo: defaultTempo)
    }

    /// Convert from a `<score-partwise>` root node.
    static func convert(root: XMLNode, defaultTempo: Double = defaultTempoBPM) -> MIDISequence {
        let parts = root.children(named: "part")
        var notes: [MIDINote] = []
        for (index, part) in parts.enumerated() {
            // One channel per part; clamp into the 0–15 MIDI range.
            notes.append(contentsOf: convertPart(part, channel: min(index, 15)))
        }
        let tempo = firstTempo(in: root) ?? defaultTempo
        return MIDISequence(
            notes: notes,
            tempoBPM: tempo,
            partCount: parts.count,
            channelPrograms: channelPrograms(in: root)
        )
    }

    /// Read each `<score-part>`'s `<midi-program>` (1-based) from `<part-list>`,
    /// keyed by channel (= part order, clamped to 15) as 0-based GM programs.
    private static func channelPrograms(in root: XMLNode) -> [Int: Int] {
        guard let partList = root.children(named: "part-list").first else { return [:] }
        var programs: [Int: Int] = [:]
        for (index, scorePart) in partList.children(named: "score-part").enumerated() {
            guard let midi = scorePart.children(named: "midi-instrument").first,
                  let program1Based = intChild(midi, "midi-program") else { continue }
            programs[min(index, 15)] = max(0, program1Based - 1)
        }
        return programs
    }

    // MARK: - Part / measure walk

    private static func convertPart(_ part: XMLNode, channel: Int) -> [MIDINote] {
        var notes: [MIDINote] = []
        var divisions = 1          // duration units per quarter note
        var measureStart = 0.0     // absolute onset of the current measure, in beats

        for measure in part.children(named: "measure") {
            var cursor = measureStart      // current insertion point (beats)
            var measureEnd = measureStart  // furthest point reached this measure
            var lastOnset = measureStart   // onset of the previous note (for chords)

            for child in measure.children {
                switch child.name {
                case "attributes":
                    if let d = intChild(child, "divisions") {
                        divisions = max(1, d)
                    }

                case "note":
                    // Grace notes carry no <duration>; skip them for v1.
                    if !child.children(named: "grace").isEmpty { continue }

                    let durationBeats = Double(intChild(child, "duration") ?? 0) / Double(divisions)
                    let isChord = !child.children(named: "chord").isEmpty
                    let onset = isChord ? lastOnset : cursor

                    // Emit a note only when there is an actual pitch (rests don't).
                    if let pitch = child.children(named: "pitch").first,
                       let midi = midiNumber(from: pitch) {
                        notes.append(MIDINote(
                            noteNumber: midi,
                            startBeats: onset,
                            durationBeats: max(durationBeats, 0),
                            velocity: 80,
                            channel: channel
                        ))
                    }

                    if isChord {
                        measureEnd = max(measureEnd, onset + durationBeats)
                    } else {
                        lastOnset = onset
                        cursor = onset + durationBeats
                        measureEnd = max(measureEnd, cursor)
                    }

                case "backup":
                    cursor -= Double(intChild(child, "duration") ?? 0) / Double(divisions)

                case "forward":
                    cursor += Double(intChild(child, "duration") ?? 0) / Double(divisions)
                    measureEnd = max(measureEnd, cursor)

                default:
                    break
                }
            }

            measureStart = measureEnd
        }

        return notes
    }

    // MARK: - Helpers

    /// Pitch → MIDI note number. `(octave + 1) * 12 + step + alter`, clamped 0–127.
    private static func midiNumber(from pitch: XMLNode) -> Int? {
        guard let step = pitch.children(named: "step").first?.value?.trimmingCharacters(in: .whitespaces),
              let semitone = stepSemitone[step.uppercased()],
              let octave = intChild(pitch, "octave") else {
            return nil
        }
        let alter = intChild(pitch, "alter") ?? 0
        let midi = (octave + 1) * 12 + semitone + alter
        return min(127, max(0, midi))
    }

    /// Diatonic step → semitone offset within an octave.
    private static let stepSemitone: [String: Int] = [
        "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
    ]

    /// Integer value of the first named child element, if parseable.
    private static func intChild(_ node: XMLNode, _ name: String) -> Int? {
        guard let raw = node.children(named: name).first?.value else { return nil }
        return Int(raw.trimmingCharacters(in: .whitespaces))
    }

    /// First `<sound tempo="…">` anywhere in the tree (depth-first), if any.
    private static func firstTempo(in node: XMLNode) -> Double? {
        if node.name == "sound", let tempo = node.attribute("tempo"), let value = Double(tempo) {
            return value
        }
        for child in node.children {
            if let found = firstTempo(in: child) {
                return found
            }
        }
        return nil
    }
}
