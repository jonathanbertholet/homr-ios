import Observation
import SwiftUI
import UIKit

// MARK: - Score detail
//
// Opens one saved score with three tabs:
//   • Scan  — the recognised page (preprocessed image + detection overlay).
//   • Score — engraved notation rendered from the MusicXML by Verovio.
//   • Parts — the arrangement editor: per-part instrument, mute/solo playback,
//             per-staff part assignment, and export.
//
// Editing mutates the in-memory `Arrangement`, re-assembles MusicXML + MIDI from
// the cached per-staff streams (no model re-run), refreshes the synth and the
// notation, and persists the change back into the library.

/// View model owning the loaded score, its editable arrangement and the player.
@Observable
@MainActor
final class ScoreDetailViewModel {
    /// The loaded score contents (images, staves, arrangement, xml).
    private(set) var detail: ScoreDetail
    /// Current editable arrangement (mutated by the editor).
    var arrangement: Arrangement
    /// Latest serialised MusicXML (drives the Score tab; regenerated on edits).
    private(set) var musicXML: String
    /// Synth player for the whole score / individual parts.
    let player = ScorePlayer()

    private let library: ScoreLibrary
    /// Full multi-channel sequence; per-part playback filters this by channel.
    private var fullSequence: MIDISequence?
    /// Whether the cached per-staff data is present (required for editing).
    var canEdit: Bool { !detail.parsedStaves.isEmpty }

    init(detail: ScoreDetail, library: ScoreLibrary) {
        self.detail = detail
        self.library = library
        self.arrangement = detail.arrangement
        self.musicXML = detail.musicXML
        // Rebuild from cached staves so the player/notation always match the
        // current arrangement; fall back to the stored XML if staves are missing.
        if !detail.parsedStaves.isEmpty {
            regenerate(persist: false)
        }
    }

    // MARK: - Regeneration

    /// Re-assemble MusicXML + MIDI for the current arrangement, refresh the synth
    /// and (optionally) persist the change.
    func regenerate(persist: Bool) {
        guard !detail.parsedStaves.isEmpty else { return }
        let score = buildScore(
            from: detail.parsedStaves, arrangement: arrangement, title: detail.record.title
        )
        musicXML = score.musicXML
        fullSequence = score.midiSequence
        if let sequence = score.midiSequence {
            player.load(sequence)
        } else {
            player.stop()
        }
        if persist {
            try? library.update(
                id: detail.record.id,
                arrangement: arrangement,
                musicXML: score.musicXML,
                midiData: score.midiData
            )
        }
    }

    // MARK: - Editing actions

    /// Assign (or clear) the instrument for a part.
    func setInstrument(_ instrument: Instrument?, for part: Int) {
        if let instrument {
            arrangement.partInstruments[part] = instrument
        } else {
            arrangement.partInstruments[part] = nil
        }
        regenerate(persist: true)
    }

    /// Move a staff into a different part (or exclude it with `part == nil`).
    func setPart(_ part: Int?, forStaff staffID: Int) {
        arrangement.staffToPart[staffID] = part ?? -1
        regenerate(persist: true)
    }

    /// Force a fixed number of voices per system, re-slicing the page top-to-
    /// bottom. Use this when auto-grouping miscounted the parts.
    func setVoiceCount(_ count: Int) {
        guard !detail.parsedStaves.isEmpty else { return }
        let wasSequential = arrangement.playSequentially
        arrangement = Arrangement.forVoiceCount(count, staves: detail.parsedStaves)
        arrangement.playSequentially = wasSequential   // preserve the play mode
        regenerate(persist: true)
    }

    /// Switch between simultaneous (parallel) and sequential part playback.
    func setSequential(_ sequential: Bool) {
        arrangement.playSequentially = sequential
        regenerate(persist: true)
    }

    /// Current playback position as a 0…1 fraction (drives the score playhead +
    /// note highlight). Returns nil when idle (never started / after stop) so the
    /// playhead is hidden, but keeps a value while paused mid-piece.
    var playbackFraction: Double? {
        guard player.hasSequence, player.duration > 0 else { return nil }
        if !player.isPlaying && player.currentTime == 0 { return nil }
        return min(1, max(0, player.currentTime / player.duration))
    }

    /// Toggle whether a part is muted (skipped in "play all" and export MIDI).
    func toggleMute(_ part: Int) {
        if arrangement.mutedParts.contains(part) {
            arrangement.mutedParts.remove(part)
        } else {
            arrangement.mutedParts.insert(part)
        }
        regenerate(persist: true)
    }

    // MARK: - Playback

    /// Play every (non-muted) part together.
    func playAll() {
        regenerate(persist: false)
        player.play()
    }

    /// Solo a single part by filtering the full sequence to its channel.
    func playPart(_ part: Int) {
        guard var sequence = fullSequence else { return }
        sequence.notes = sequence.notes.filter { $0.channel == part }
        player.load(sequence)
        player.play()
    }

    /// Display name for a part (assigned instrument, else "Part N").
    func partName(_ part: Int) -> String {
        arrangement.partInstruments[part]?.name ?? "Part \(part + 1)"
    }

    /// Staves grouped by system in reading order (for the staff assignment list).
    var stavesBySystem: [(system: Int, staves: [ParsedStaff])] {
        let grouped = Dictionary(grouping: detail.parsedStaves, by: \.systemIndex)
        return grouped.keys.sorted().map { key in
            (system: key, staves: grouped[key]!.sorted { $0.orderInSystem < $1.orderInSystem })
        }
    }

    /// Persisted MIDI file URL for export (after the latest regenerate/persist).
    var midiURL: URL? { library.midiURL(for: detail.record) }
}

/// The tabbed detail screen for one saved score.
struct ScoreDetailView: View {
    @Environment(ScoreLibrary.self) private var library
    let record: ScoreRecord

    @State private var viewModel: ScoreDetailViewModel?
    @State private var loadError: String?
    /// Whether the Score-tab playback/voices bottom sheet is presented.
    @State private var showScoreControls = false

    var body: some View {
        Group {
            if let viewModel {
                TabView {
                    // The floating miniplayer is attached per-tab (not to the
                    // TabView) so it floats *above* the system tab bar and
                    // reserves its own space instead of covering the tabs.
                    scanTab(viewModel)
                        .safeAreaInset(edge: .bottom) { miniPlayer(viewModel) }
                        .tabItem { Label("Scan", systemImage: "doc.text.image") }

                    scoreTab(viewModel)
                        .safeAreaInset(edge: .bottom) { miniPlayer(viewModel) }
                        .tabItem { Label("Score", systemImage: "music.note.list") }

                    partsTab(viewModel)
                        .safeAreaInset(edge: .bottom) { miniPlayer(viewModel) }
                        .tabItem { Label("Parts", systemImage: "slider.horizontal.3") }
                }
                // Options button (in the miniplayer) opens the full settings sheet.
                .sheet(isPresented: $showScoreControls) { settingsSheet(viewModel) }
            } else if let loadError {
                ContentUnavailableView("Couldn't open score", systemImage: "exclamationmark.triangle", description: Text(loadError))
            } else {
                ProgressView("Loading…")
            }
        }
        .navigationTitle(record.title)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: loadIfNeeded)
    }

    // MARK: - Miniplayer (floating, persistent across tabs)

    /// Compact floating transport shown above the tab bar on every tab: play/
    /// pause, stop, a thin progress bar, and the options button that opens the
    /// settings sheet. Styled as an inset rounded bar so it never covers content
    /// edge-to-edge; `safeAreaInset` (per tab) reserves its space above the tabs.
    private static let accent = Color(red: 0.91, green: 0.27, blue: 0.17)

    @ViewBuilder
    private func miniPlayer(_ vm: ScoreDetailViewModel) -> some View {
        HStack(spacing: 12) {
            Button {
                vm.player.isPlaying ? vm.player.pause() : vm.player.play()
            } label: {
                Image(systemName: vm.player.isPlaying ? "pause.fill" : "play.fill")
                    .font(.title3)
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel(vm.player.isPlaying ? "Pause" : "Play")

            Button {
                vm.player.stop()
            } label: {
                Image(systemName: "stop.fill")
                    .font(.body)
                    .frame(width: 26, height: 30)
                    .contentShape(Rectangle())
            }
            .foregroundStyle(.secondary)
            .accessibilityLabel("Stop")

            VStack(alignment: .leading, spacing: 4) {
                Text(record.title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                ProgressView(
                    value: min(vm.player.currentTime, max(vm.player.duration, 0.01)),
                    total: max(vm.player.duration, 0.01)
                )
                .progressViewStyle(.linear)
                .tint(Self.accent)
            }

            Spacer(minLength: 4)

            Button {
                showScoreControls = true
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.title3)
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel("Playback and arrangement options")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.regularMaterial)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(.primary.opacity(0.06), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.18), radius: 14, x: 0, y: 6)
        .padding(.horizontal, 16)
        .padding(.bottom, 6)
    }

    /// Load the full score from disk once when the screen appears.
    private func loadIfNeeded() {
        guard viewModel == nil else { return }
        do {
            let detail = try library.detail(for: record)
            viewModel = ScoreDetailViewModel(detail: detail, library: library)
        } catch {
            loadError = error.localizedDescription
        }
    }

    // MARK: - Scan tab

    @ViewBuilder
    private func scanTab(_ vm: ScoreDetailViewModel) -> some View {
        ScrollView {
            VStack(spacing: 16) {
                if let overlay = vm.detail.overlayImage, let base = vm.detail.previewImage {
                    ZStack {
                        Image(uiImage: base).resizable().scaledToFit()
                        Image(uiImage: overlay).resizable().scaledToFit()
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                } else if let base = vm.detail.previewImage {
                    Image(uiImage: base).resizable().scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                Text(vm.detail.record.summary)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding()
        }
    }

    // MARK: - Score tab (Verovio notation)

    @ViewBuilder
    private func scoreTab(_ vm: ScoreDetailViewModel) -> some View {
        if vm.musicXML.isEmpty {
            ContentUnavailableView("No notation", systemImage: "music.note", description: Text("This score has no MusicXML to render."))
        } else {
            // Rendered by the bundled, offline Verovio toolkit in a WKWebView.
            // `highlightFraction` drives the live playhead + note highlight; it
            // reads the player position so the view re-evaluates during playback
            // and follows along across every staff.
            ScoreWebView(musicXML: vm.musicXML, highlightFraction: vm.playbackFraction)
                .ignoresSafeArea(edges: .bottom)
        }
    }

    // MARK: - Settings sheet (full arrangement editor, opened from miniplayer)

    /// Bottom sheet holding every Parts-tab setting: transport, voices/play mode,
    /// per-part instruments, staff assignment and export. Background interaction
    /// stays enabled at the small detent so the live playhead remains visible.
    @ViewBuilder
    private func settingsSheet(_ vm: ScoreDetailViewModel) -> some View {
        NavigationStack {
            Form {
                playbackSection(vm)
                voicesSection(vm)
                instrumentsSection(vm)
                staffSection(vm)
                exportSection(vm)
            }
            .navigationTitle("Playback & Arrangement")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showScoreControls = false }
                }
            }
        }
        .presentationDetents([.height(320), .large])
        .presentationBackgroundInteraction(.enabled(upThrough: .height(320)))
        .presentationContentInteraction(.scrolls)
    }

    // MARK: - Parts tab (editor + playback + export)

    @ViewBuilder
    private func partsTab(_ vm: ScoreDetailViewModel) -> some View {
        Form {
            playbackSection(vm)
            voicesSection(vm)
            instrumentsSection(vm)
            staffSection(vm)
            exportSection(vm)
        }
    }

    // MARK: - Shared settings sections (used by Parts tab + settings sheet)

    @ViewBuilder
    private func playbackSection(_ vm: ScoreDetailViewModel) -> some View {
        Section {
            PlaybackControlsView(player: vm.player)
            Button {
                vm.playAll()
            } label: {
                Label("Play all parts", systemImage: "play.fill")
            }
        } header: {
            Text("Playback")
        }
    }

    @ViewBuilder
    private func voicesSection(_ vm: ScoreDetailViewModel) -> some View {
        if vm.canEdit {
            Section {
                Stepper(
                    "Voices per system: \(vm.arrangement.partCount)",
                    value: Binding(
                        get: { vm.arrangement.partCount },
                        set: { vm.setVoiceCount($0) }
                    ),
                    in: 1...12
                )
                .disabled(vm.player.isPlaying)

                Toggle(
                    "Play voices simultaneously",
                    isOn: Binding(
                        get: { !vm.arrangement.playSequentially },
                        set: { vm.setSequential(!$0) }
                    )
                )
                .disabled(vm.player.isPlaying)
            } header: {
                Text("Voices")
            } footer: {
                Text("How many staves make up one system (e.g. 4 for a quartet). The page is re-sliced top-to-bottom: these voices play together, systems play in sequence. Turn off \u{201C}simultaneously\u{201D} to audition the parts one after another.")
            }
        }
    }

    @ViewBuilder
    private func instrumentsSection(_ vm: ScoreDetailViewModel) -> some View {
        Section {
            ForEach(Array(0..<vm.arrangement.partCount), id: \.self) { part in
                partRow(vm, part: part)
            }
        } header: {
            Text("Parts & instruments")
        } footer: {
            Text("Each part plays in parallel with its own timbre. Mute to drop a part from playback and export.")
        }
    }

    @ViewBuilder
    private func staffSection(_ vm: ScoreDetailViewModel) -> some View {
        if vm.canEdit {
            Section {
                ForEach(vm.stavesBySystem, id: \.system) { group in
                    DisclosureGroup("System \(group.system + 1)") {
                        ForEach(group.staves) { staff in
                            staffRow(vm, staff: staff)
                        }
                    }
                }
            } header: {
                Text("Staff assignment")
            } footer: {
                Text("Reassign a staff if a system was grouped incorrectly. Part = the instrument line it belongs to.")
            }
        }
    }

    @ViewBuilder
    private func exportSection(_ vm: ScoreDetailViewModel) -> some View {
        Section {
            ShareLink(item: vm.musicXML, preview: SharePreview("\(record.title).musicxml")) {
                Label("Export MusicXML", systemImage: "square.and.arrow.up")
            }
            if let midiURL = vm.midiURL {
                ShareLink(item: midiURL, preview: SharePreview("\(record.title).mid")) {
                    Label("Export MIDI", systemImage: "pianokeys")
                }
            }
        } header: {
            Text("Export")
        }
    }

    /// One part: instrument picker + mute + solo-play.
    @ViewBuilder
    private func partRow(_ vm: ScoreDetailViewModel, part: Int) -> some View {
        HStack {
            // Instrument picker (current instrument, or "Auto").
            Menu {
                Button("Auto / none") { vm.setInstrument(nil, for: part) }
                Divider()
                ForEach(InstrumentCatalog.pickerPresets, id: \.self) { instrument in
                    Button(instrument.name) { vm.setInstrument(instrument, for: part) }
                }
            } label: {
                VStack(alignment: .leading, spacing: 2) {
                    Text(vm.partName(part)).font(.body)
                    Text("Part \(part + 1)").font(.caption2).foregroundStyle(.secondary)
                }
            }

            Spacer()

            Button {
                vm.playPart(part)
            } label: {
                Image(systemName: "play.circle")
            }
            .buttonStyle(.borderless)

            Button {
                vm.toggleMute(part)
            } label: {
                Image(systemName: vm.arrangement.mutedParts.contains(part) ? "speaker.slash.fill" : "speaker.wave.2.fill")
                    .foregroundStyle(vm.arrangement.mutedParts.contains(part) ? .red : .secondary)
            }
            .buttonStyle(.borderless)
        }
    }

    /// One staff: which part it is assigned to.
    @ViewBuilder
    private func staffRow(_ vm: ScoreDetailViewModel, staff: ParsedStaff) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(staff.label ?? "Staff \(staff.orderInSystem + 1)").font(.subheadline)
                if let label = staff.label {
                    Text(label).font(.caption2).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Picker("Part", selection: Binding(
                get: { vm.arrangement.staffToPart[staff.id] ?? -1 },
                set: { vm.setPart($0 < 0 ? nil : $0, forStaff: staff.id) }
            )) {
                Text("Exclude").tag(-1)
                ForEach(Array(0..<vm.arrangement.partCount), id: \.self) { part in
                    Text(vm.partName(part)).tag(part)
                }
            }
            .labelsHidden()
        }
    }
}
