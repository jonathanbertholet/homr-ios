import Foundation

// MARK: - Standard MIDI File writer
//
// Serialises a `MIDISequence` into a real type-0 Standard MIDI File (`.mid`):
// one track holding a tempo meta event plus note-on/note-off pairs across all
// channels. This is the literal "MIDI" artefact of the MusicXML → MIDI step and
// can be shared/opened in any DAW or notation app.
//
// The synthesizer (`ScorePlayer`) plays the `MIDISequence` directly and does
// not depend on this file; the two are independent consumers of the sequence.
enum StandardMIDIFile {
    /// Pulses-per-quarter-note used as the SMF time division. 480 is the common
    /// notation-software default and divides cleanly into typical note values.
    static let ticksPerQuarter = 480

    /// Render `sequence` as `.mid` file bytes.
    static func data(from sequence: MIDISequence) -> Data {
        var track = Data()

        // Tempo meta event at tick 0: FF 51 03 <microseconds-per-quarter>.
        let usPerQuarter = Int(60_000_000.0 / max(sequence.tempoBPM, 1))
        appendVariableLength(&track, 0)
        track.append(contentsOf: [0xFF, 0x51, 0x03])
        track.append(UInt8((usPerQuarter >> 16) & 0xFF))
        track.append(UInt8((usPerQuarter >> 8) & 0xFF))
        track.append(UInt8(usPerQuarter & 0xFF))

        // Program-change events at tick 0, one per channel that has a program,
        // so DAWs render each part with its identified instrument.
        for (channel, program) in sequence.channelPrograms.sorted(by: { $0.key < $1.key }) {
            let ch = UInt8(min(15, max(0, channel)))
            let prog = UInt8(min(127, max(0, program)))
            appendVariableLength(&track, 0)
            track.append(contentsOf: [0xC0 | ch, prog])
        }

        // Expand notes into absolute-tick on/off events.
        struct Event { let tick: Int; let isNoteOff: Bool; let status: UInt8; let data1: UInt8; let data2: UInt8 }
        var events: [Event] = []
        for note in sequence.notes {
            let channel = UInt8(min(15, max(0, note.channel)))
            let key = UInt8(min(127, max(0, note.noteNumber)))
            let velocity = UInt8(min(127, max(1, note.velocity)))
            let startTick = Int((note.startBeats * Double(ticksPerQuarter)).rounded())
            // Guarantee at least 1 tick so the note isn't zero-length.
            let endTick = max(startTick + 1, Int(((note.startBeats + note.durationBeats) * Double(ticksPerQuarter)).rounded()))
            events.append(Event(tick: startTick, isNoteOff: false, status: 0x90 | channel, data1: key, data2: velocity))
            events.append(Event(tick: endTick, isNoteOff: true, status: 0x80 | channel, data1: key, data2: 0))
        }

        // Stable sort by tick, with note-offs before note-ons at the same tick
        // so a repeated pitch releases before it re-attacks (avoids stuck notes).
        let ordered = events.enumerated().sorted { lhs, rhs in
            if lhs.element.tick != rhs.element.tick { return lhs.element.tick < rhs.element.tick }
            if lhs.element.isNoteOff != rhs.element.isNoteOff { return lhs.element.isNoteOff }
            return lhs.offset < rhs.offset
        }.map { $0.element }

        var lastTick = 0
        for event in ordered {
            appendVariableLength(&track, event.tick - lastTick)
            track.append(contentsOf: [event.status, event.data1, event.data2])
            lastTick = event.tick
        }

        // End-of-track meta event.
        appendVariableLength(&track, 0)
        track.append(contentsOf: [0xFF, 0x2F, 0x00])

        // Assemble: MThd header chunk + MTrk track chunk.
        var file = Data()
        file.append(contentsOf: Array("MThd".utf8))
        appendUInt32(&file, 6)            // header length
        appendUInt16(&file, 0)            // format 0 (single multi-channel track)
        appendUInt16(&file, 1)            // one track
        appendUInt16(&file, UInt16(ticksPerQuarter))
        file.append(contentsOf: Array("MTrk".utf8))
        appendUInt32(&file, UInt32(track.count))
        file.append(track)
        return file
    }

    // MARK: - Byte helpers

    /// Append a big-endian UInt32.
    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        data.append(UInt8((value >> 24) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8(value & 0xFF))
    }

    /// Append a big-endian UInt16.
    private static func appendUInt16(_ data: inout Data, _ value: UInt16) {
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8(value & 0xFF))
    }

    /// Append an SMF variable-length quantity (7 bits per byte, MSB = continue).
    private static func appendVariableLength(_ data: inout Data, _ value: Int) {
        var v = UInt32(max(0, value))
        var buffer: [UInt8] = [UInt8(v & 0x7F)]
        v >>= 7
        while v > 0 {
            buffer.append(UInt8((v & 0x7F) | 0x80))
            v >>= 7
        }
        // Bytes were produced least-significant first; emit most-significant first.
        data.append(contentsOf: buffer.reversed())
    }
}
