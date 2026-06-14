import Foundation

// MARK: - Synth timbre presets
//
// A lightweight additive-synthesis voice description used by `SynthCore`. Each
// timbre is a set of harmonic amplitudes (normalised so the partials sum to ~1,
// keeping the limiter honest) plus an amplitude envelope. A non-zero `decayRate`
// models struck/plucked instruments (piano, harpsichord, harp) whose tone fades
// while held; sustained instruments (strings, organ, winds) leave it at 0.
//
// Timbres are chosen per General MIDI program family so identified instruments
// sound distinct without bundling any samples or SoundFonts.
struct Timbre: Sendable {
    /// Harmonic amplitudes, index 0 = fundamental. Normalised to sum to ~1.
    let harmonics: [Double]
    /// Attack ramp length, seconds.
    let attack: Double
    /// Release ramp length, seconds.
    let release: Double
    /// Exponential decay rate (1/seconds) applied while the note sounds; 0 = none.
    let decayRate: Double

    init(harmonics: [Double], attack: Double, release: Double, decayRate: Double = 0) {
        // Normalise so dense chords don't slam the soft-clipper.
        let sum = harmonics.reduce(0, +)
        self.harmonics = sum > 0 ? harmonics.map { $0 / sum } : harmonics
        self.attack = attack
        self.release = release
        self.decayRate = decayRate
    }

    /// Neutral fallback voice (mellow sine + a little colour).
    static let `default` = Timbre(harmonics: [1.0, 0.18, 0.08], attack: 0.006, release: 0.05)

    /// Pick a timbre for a General MIDI program (0-based). Grouped by GM family.
    static func forGMProgram(_ program: Int) -> Timbre {
        switch program {
        case 6:        // Harpsichord — bright, fast decay (check before the piano range)
            return Timbre(harmonics: [1.0, 0.7, 0.5, 0.35, 0.2], attack: 0.002, release: 0.1, decayRate: 3.5)
        case 0...7:    // Pianos / keyboards
            return Timbre(harmonics: [1.0, 0.5, 0.28, 0.16, 0.08], attack: 0.004, release: 0.18, decayRate: 2.2)
        case 16...23:  // Organs — steady, rich, no decay
            return Timbre(harmonics: [1.0, 0.9, 0.6, 0.5, 0.4, 0.3], attack: 0.02, release: 0.08)
        case 24...31:  // Guitars — plucked
            return Timbre(harmonics: [1.0, 0.6, 0.4, 0.25, 0.15], attack: 0.003, release: 0.15, decayRate: 2.8)
        case 32...39:  // Basses — plucked, dark
            return Timbre(harmonics: [1.0, 0.45, 0.2, 0.1], attack: 0.004, release: 0.12, decayRate: 2.0)
        case 40...47:  // Strings (bowed) — rich, sustained
            return Timbre(harmonics: [1.0, 0.7, 0.55, 0.42, 0.32, 0.24], attack: 0.045, release: 0.18)
        case 48...55:  // Ensemble / voices
            return Timbre(harmonics: [1.0, 0.5, 0.3, 0.18, 0.1], attack: 0.06, release: 0.22)
        case 56...63:  // Brass — bright, sustained
            return Timbre(harmonics: [1.0, 0.85, 0.65, 0.5, 0.4, 0.28], attack: 0.03, release: 0.1)
        case 64...71:  // Reeds (oboe, clarinet, bassoon, sax)
            return Timbre(harmonics: [1.0, 0.55, 0.5, 0.38, 0.28, 0.18], attack: 0.02, release: 0.1)
        case 72...79:  // Pipes / flutes — nearly pure
            return Timbre(harmonics: [1.0, 0.22, 0.08], attack: 0.035, release: 0.12)
        default:
            return .default
        }
    }
}
