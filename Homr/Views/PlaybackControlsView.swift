import SwiftUI

/// Reusable transport for the synth `ScorePlayer`: play/pause, stop, a scrubber
/// and a tempo stepper. Shared by the scan screen and the score editor so both
/// behave identically.
struct PlaybackControlsView: View {
    let player: ScorePlayer

    var body: some View {
        // Two-way binding: read the live position, write a seek on scrub.
        let position = Binding(
            get: { player.currentTime },
            set: { player.seek(toSeconds: $0) }
        )
        VStack(spacing: 10) {
            HStack(spacing: 20) {
                Button {
                    player.isPlaying ? player.pause() : player.play()
                } label: {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 40))
                }

                Button {
                    player.stop()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.system(size: 40))
                }
                .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Tempo")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Stepper(
                        "\(Int(player.tempoBPM)) BPM",
                        value: Binding(
                            get: { player.tempoBPM },
                            set: { player.tempoBPM = $0 }
                        ),
                        in: 30...240,
                        step: 5
                    )
                    .font(.caption)
                    // Tempo only re-scales the schedule while playback is stopped.
                    .disabled(player.isPlaying)
                }
            }

            HStack(spacing: 8) {
                Text(timeString(player.currentTime))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                Slider(value: position, in: 0...max(player.duration, 0.01))
                Text(timeString(player.duration))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Format seconds as `m:ss` for the transport labels.
    private func timeString(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
