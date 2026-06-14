import Foundation

// MARK: - MIDI sequence model
//
// The intermediate "MIDI" representation produced from a recognised MusicXML
// document. It is intentionally engine-agnostic: timing is expressed in
// quarter-note beats (not seconds or samples) so the same sequence can drive
// either the in-app synthesizer (`ScorePlayer`) or a Standard MIDI File export
// (`StandardMIDIFile`). All types are value types and `Sendable` so the
// sequence can cross the `OMRProcessor` actor boundary safely.

/// A single sounding note.
///
/// `startBeats`/`durationBeats` are in quarter-note beats from the start of the
/// piece; combined with `MIDISequence.tempoBPM` they convert to wall-clock time.
struct MIDINote: Sendable, Equatable {
    /// MIDI note number, 0–127 (60 = middle C / C4).
    var noteNumber: Int
    /// Onset, in quarter-note beats from the start of the piece.
    var startBeats: Double
    /// Sounding length, in quarter-note beats.
    var durationBeats: Double
    /// MIDI velocity, 1–127.
    var velocity: Int
    /// MIDI channel, 0–15 (one per recognised part/voice, clamped to 15).
    var channel: Int
}

/// A converted score ready for playback or MIDI export.
struct MIDISequence: Sendable {
    /// All notes across every part, in no particular order.
    var notes: [MIDINote]
    /// Tempo in quarter notes per minute.
    var tempoBPM: Double
    /// Number of recognised parts (voices) the notes came from.
    var partCount: Int
    /// MIDI channel → General MIDI program (0-based) from the score's part list.
    /// Drives per-channel timbre in the synth and program-change events in the
    /// exported `.mid`. Channels absent here use the default voice.
    var channelPrograms: [Int: Int] = [:]

    /// Total length in quarter-note beats (latest note end), 0 when empty.
    var durationBeats: Double {
        notes.map { $0.startBeats + $0.durationBeats }.max() ?? 0
    }

    /// Total length in seconds at the sequence tempo.
    var durationSeconds: Double {
        durationBeats * 60.0 / max(tempoBPM, 1)
    }

    var isEmpty: Bool { notes.isEmpty }
}
