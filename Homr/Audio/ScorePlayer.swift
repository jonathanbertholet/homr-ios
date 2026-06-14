import AVFoundation
import Foundation
import Observation
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Score playback (iOS audio path)
//
// Plays a `MIDISequence` through `AVAudioEngine` using a hand-written
// polyphonic synthesizer (`SynthCore`) fed by an `AVAudioSourceNode`. This needs
// no bundled SoundFont and no `AVMIDIPlayer`/`AVAudioUnitSampler` instrument, so
// it produces sound reliably on device and Simulator. The timbre is a mellow
// sine-plus-harmonics voice with a short attack/release envelope to avoid clicks.
//
// `ScorePlayer` is the `@MainActor`, observable transport (play/pause/stop/seek
// + progress). `SynthCore` owns the immutable, precomputed voice schedule and
// runs entirely on the audio render thread.

/// Real-time synthesizer core. All audio-thread state is touched only inside
/// `render(...)`; the main thread mutates it only while the engine is paused
/// (see `ScorePlayer`), so no locking is needed on the render path.
///
/// `@unchecked Sendable`: the render block runs off the main actor, so the
/// compiler can't verify the hand-off. Safety comes from the discipline above —
/// `load`/`seek` are only called while the engine is stopped, and `playhead`/
/// `finished` are simple word-sized reads on the UI side.
final class SynthCore: @unchecked Sendable {
    /// A precomputed sounding note in absolute output frames, carrying its
    /// instrument timbre (harmonics + envelope) so different parts sound distinct.
    struct Voice {
        let startFrame: Int
        let endFrame: Int
        let frequency: Double
        let amplitude: Double
        let timbre: Timbre
    }

    private let sampleRate: Double
    private var voices: [Voice] = []
    /// Frame index of the last note end (read on the main thread to know when a
    /// finished playhead should restart from the top).
    private(set) var totalFrames: Int = 0

    /// Per-buffer mix scratch, preallocated so the render thread never allocates.
    private var mix: [Float]
    private let maxFrames = 8192

    /// Current play position in output frames. Written by the audio thread during
    /// playback; read on the main thread for the progress UI (a benign data race —
    /// aligned 64-bit-ish loads on arm64 — acceptable for a position indicator).
    var playhead: Int = 0
    /// Set true by the audio thread once the schedule (plus release tail) is done.
    var finished: Bool = false

    init(sampleRate: Double) {
        self.sampleRate = sampleRate
        self.mix = [Float](repeating: 0, count: maxFrames)
    }

    /// Install a new schedule and rewind. Call only while the engine is stopped.
    func load(voices: [Voice], totalFrames: Int) {
        self.voices = voices.sorted { $0.startFrame < $1.startFrame }
        self.totalFrames = totalFrames
        self.playhead = 0
        self.finished = false
        // Size the completion tail to the longest release across all timbres.
        let longestRelease = voices.map(\.timbre.release).max() ?? 0.04
        self.maxReleaseFrames = max(longestRelease * sampleRate, 1.0)
    }

    /// Move the play position (clamped). Call only while the engine is stopped.
    func seek(toFrame frame: Int) {
        playhead = min(max(0, frame), totalFrames)
        finished = false
    }

    /// Longest release across timbres, in frames — how far past the last note we
    /// keep rendering so release tails aren't cut off.
    private var maxReleaseFrames = 1760.0

    /// Fill `frameCount` frames into every channel of `buffers`.
    func render(into buffers: UnsafeMutableAudioBufferListPointer, frameCount: Int) {
        let start = playhead
        let count = min(frameCount, maxFrames)

        mix.withUnsafeMutableBufferPointer { scratch in
            // Clear the scratch mix.
            for i in 0..<count { scratch[i] = 0 }

            // Sum every voice that overlaps this buffer window.
            let windowEnd = start + count
            for voice in voices {
                if voice.endFrame <= start { continue }
                if voice.startFrame >= windowEnd { break } // sorted by start → done
                let from = max(voice.startFrame, start)
                let to = min(voice.endFrame, windowEnd)
                let length = Double(voice.endFrame - voice.startFrame)
                let attackFrames = voice.timbre.attack * sampleRate
                let releaseFrames = voice.timbre.release * sampleRate
                let decayRate = voice.timbre.decayRate
                let harmonics = voice.timbre.harmonics
                var frame = from
                while frame < to {
                    let local = Double(frame - voice.startFrame)
                    // Attack/release envelope.
                    var env = 1.0
                    if local < attackFrames {
                        env = local / attackFrames
                    } else if local > length - releaseFrames {
                        env = max(0, (length - local) / releaseFrames)
                    }
                    // Exponential decay for struck/plucked timbres (0 = sustained).
                    if decayRate > 0 {
                        env *= exp(-decayRate * local / sampleRate)
                    }
                    // Additive synthesis over the timbre's harmonic series.
                    let phase = 2.0 * Double.pi * voice.frequency * Double(frame) / sampleRate
                    var tone = 0.0
                    for k in 0..<harmonics.count {
                        tone += harmonics[k] * sin(Double(k + 1) * phase)
                    }
                    scratch[frame - start] += Float(tone * env * voice.amplitude)
                    frame += 1
                }
            }

            // Soft-clip the mix once (tanh) to tame chord peaks without harsh clipping.
            for i in 0..<count {
                scratch[i] = Float(tanh(Double(scratch[i])))
            }

            // Copy the mono mix into each output channel.
            for buffer in buffers {
                guard let dst = buffer.mData?.assumingMemoryBound(to: Float.self) else { continue }
                for i in 0..<count { dst[i] = scratch[i] }
                // Zero any frames beyond our clamp (defensive; count == frameCount normally).
                if count < frameCount {
                    for i in count..<frameCount { dst[i] = 0 }
                }
            }
        }

        playhead = start + frameCount
        // Allow a release tail past the last note before declaring completion.
        if playhead >= totalFrames + Int(maxReleaseFrames) {
            finished = true
        }
    }
}

/// Observable, main-actor transport that drives `SynthCore` via `AVAudioEngine`.
@MainActor
@Observable
final class ScorePlayer {
    private(set) var isPlaying = false
    /// Current playback position, seconds (kept in sync by a UI timer).
    private(set) var currentTime: Double = 0
    /// Total duration, seconds (depends on the sequence and tempo).
    private(set) var duration: Double = 0
    /// Whether a non-empty sequence is loaded and ready to play.
    private(set) var hasSequence = false

    /// Playback tempo (quarter notes per minute). Changing it while stopped
    /// rescales timing on the next play; the slider/stepper in the UI binds here.
    var tempoBPM: Double = MusicXMLMIDIConverter.defaultTempoBPM {
        didSet {
            guard !isPlaying, oldValue != tempoBPM else { return }
            rebuildSchedule(resetPosition: false)
        }
    }

    private let sampleRate: Double = 44_100
    private let engine = AVAudioEngine()
    private let core: SynthCore
    private var sourceNode: AVAudioSourceNode?
    private var sequence: MIDISequence?
    private var progressTimer: Timer?
    private var engineConfigured = false

    init() {
        self.core = SynthCore(sampleRate: sampleRate)
    }

    /// Load a converted sequence. Resets transport to the start.
    func load(_ sequence: MIDISequence) {
        stop()
        self.sequence = sequence
        self.tempoBPM = sequence.tempoBPM
        self.hasSequence = !sequence.isEmpty
        rebuildSchedule(resetPosition: true)
    }

    /// Recompute the frame-accurate voice schedule from the sequence + tempo.
    private func rebuildSchedule(resetPosition: Bool) {
        guard let sequence else {
            core.load(voices: [], totalFrames: 0)
            duration = 0
            currentTime = 0
            return
        }
        let secondsPerBeat = 60.0 / max(tempoBPM, 1)
        // Normalise per-note amplitude by an estimated polyphony so dense chords
        // don't slam the limiter; velocity then scales within that headroom.
        let voiceGain = 0.32
        // Resolve each channel's GM program once into a timbre.
        var timbreByChannel: [Int: Timbre] = [:]
        for (channel, program) in sequence.channelPrograms {
            timbreByChannel[channel] = Timbre.forGMProgram(program)
        }
        var voices: [SynthCore.Voice] = []
        voices.reserveCapacity(sequence.notes.count)
        var maxEnd = 0
        for note in sequence.notes {
            let startFrame = Int(note.startBeats * secondsPerBeat * sampleRate)
            let endFrame = Int((note.startBeats + note.durationBeats) * secondsPerBeat * sampleRate)
            guard endFrame > startFrame else { continue }
            let frequency = 440.0 * pow(2.0, Double(note.noteNumber - 69) / 12.0)
            let amplitude = voiceGain * Double(note.velocity) / 127.0
            let timbre = timbreByChannel[note.channel] ?? .default
            voices.append(SynthCore.Voice(
                startFrame: startFrame, endFrame: endFrame,
                frequency: frequency, amplitude: amplitude, timbre: timbre
            ))
            maxEnd = max(maxEnd, endFrame)
        }
        core.load(voices: voices, totalFrames: maxEnd)
        duration = Double(maxEnd) / sampleRate
        if resetPosition { currentTime = 0 }
    }

    // MARK: - Transport

    func play() {
        guard hasSequence else { return }
        // Restart from the top if we're sitting at the end.
        if core.playhead >= core.totalFrames {
            core.seek(toFrame: 0)
            currentTime = 0
        }
        do {
            try configureEngineIfNeeded()
            activateSessionForPlayback()
            if !engine.isRunning {
                try engine.start()
            }
            isPlaying = true
            startProgressTimer()
        } catch {
            isPlaying = false
        }
    }

    func pause() {
        guard isPlaying else { return }
        engine.pause()
        isPlaying = false
        stopProgressTimer()
    }

    func stop() {
        engine.stop()
        isPlaying = false
        stopProgressTimer()
        core.seek(toFrame: 0)
        currentTime = 0
    }

    /// Seek to `seconds`. Pauses the engine first so the render thread isn't
    /// reading the playhead while we set it.
    func seek(toSeconds seconds: Double) {
        let wasPlaying = isPlaying
        if engine.isRunning { engine.pause() }
        core.seek(toFrame: Int(seconds * sampleRate))
        currentTime = min(max(0, seconds), duration)
        if wasPlaying {
            try? engine.start()
        }
    }

    // MARK: - Engine setup

    private func configureEngineIfNeeded() throws {
        guard !engineConfigured else { return }
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        let node = AVAudioSourceNode(format: format) { [core] _, _, frameCount, audioBufferList in
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            core.render(into: buffers, frameCount: Int(frameCount))
            return noErr
        }
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        sourceNode = node
        engine.prepare()
        engineConfigured = true
    }

    private func activateSessionForPlayback() {
        #if canImport(UIKit)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default)
        try? session.setActive(true)
        #endif
    }

    // MARK: - Progress

    private func startProgressTimer() {
        stopProgressTimer()
        let timer = Timer(timeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tickProgress() }
        }
        RunLoop.main.add(timer, forMode: .common)
        progressTimer = timer
    }

    private func stopProgressTimer() {
        progressTimer?.invalidate()
        progressTimer = nil
    }

    private func tickProgress() {
        currentTime = min(Double(core.playhead) / sampleRate, duration)
        if core.finished {
            // Reached the end: reset to the top and stop.
            stop()
        }
    }
}
