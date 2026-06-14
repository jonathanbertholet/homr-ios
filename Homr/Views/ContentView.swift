import Observation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// Main screen: model setup, image import, and OMR processing.
struct ContentView: View {
    @Environment(ModelStore.self) private var modelStore
    @Environment(ScoreLibrary.self) private var library
    /// Called when the user opens a freshly-saved score (the Library handles the
    /// actual navigation so the detail isn't hidden behind this scanner sheet).
    var onOpen: (ScoreRecord) -> Void = { _ in }
    @State private var viewModel = ScanViewModel()
    @State private var selectedPhotos: [PhotosPickerItem] = []
    @State private var isImportingFile = false

    var body: some View {
        // No NavigationStack here: this screen is pushed inside the Library's
        // stack, so it inherits that stack (and its `navigationDestination`).
        ScrollView {
            VStack(spacing: 24) {
                header
                modelStatusCard
                importButtons
                queueSection
                resultSection

                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                }
            }
            .padding()
        }
        .navigationTitle("New Scan")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await modelStore.prepareModelsIfNeeded()
        }
        .onChange(of: selectedPhotos) { _, newItems in
            guard !newItems.isEmpty else { return }
            // Snapshot + clear the picker selection so the same photos can be
            // re-added later, then enqueue them (no auto-scan).
            let items = newItems
            selectedPhotos = []
            Task { await viewModel.addPhotos(items) }
        }
        .fileImporter(
            isPresented: $isImportingFile,
            allowedContentTypes: SheetMusicImporter.supportedTypes,
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                guard !urls.isEmpty else { return }
                Task { await viewModel.addFiles(urls) }
            case .failure(let error):
                viewModel.errorMessage = error.localizedDescription
            }
        }
    }

    /// Photo library + document picker (PDF, PNG, JPEG, HEIC, TIFF). Both ADD to
    /// the queue (you can mix sources and add repeatedly); scanning is manual.
    private var importButtons: some View {
        VStack(spacing: 12) {
            PhotosPicker(
                selection: $selectedPhotos,
                selectionBehavior: .ordered,
                matching: .images
            ) {
                Label("Add Photos", systemImage: "photo.on.rectangle.angled")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(!modelStore.isReady || viewModel.isProcessing)

            Button {
                isImportingFile = true
            } label: {
                Label("Add Files (PDF or Image)", systemImage: "doc.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(!modelStore.isReady || viewModel.isProcessing)
        }
    }

    /// The pending page queue: selection shortcuts, thumbnails (tap to include /
    /// exclude), live per-page scan progress, and the manual "Scan" confirmation.
    /// Only the selected pages are recognised into a single combined score.
    @ViewBuilder
    private var queueSection: some View {
        if !viewModel.pages.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                queueHeader
                queueThumbnails
                if viewModel.isProcessing {
                    scanProgressBar
                } else {
                    scanButton
                }
            }
            .padding()
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
            .animation(.default, value: viewModel.isProcessing)
        }
    }

    /// Title + selection-count, with a shortcuts menu (Select All / None / Clear).
    private var queueHeader: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Pages to scan")
                    .font(.subheadline.weight(.semibold))
                Text("\(viewModel.selectedCount) of \(viewModel.pages.count) selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                Button { viewModel.selectAll() } label: {
                    Label("Select All", systemImage: "checkmark.circle")
                }
                .disabled(viewModel.selectedCount == viewModel.pages.count)

                Button { viewModel.selectNone() } label: {
                    Label("Select None", systemImage: "circle")
                }
                .disabled(viewModel.selectedCount == 0)

                Divider()

                Button(role: .destructive) { viewModel.clearQueue() } label: {
                    Label("Remove All", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.title3)
            }
            .disabled(viewModel.isProcessing)
        }
    }

    /// Horizontally-scrolling thumbnails. Tap toggles selection (when idle); during
    /// a scan each shows its live status (waiting / spinner / done).
    private var queueThumbnails: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(Array(viewModel.pages.enumerated()), id: \.offset) { index, image in
                    queueThumbnail(index: index, image: image)
                }
            }
            .padding(.vertical, 2)
        }
    }

    /// The prominent confirm button — scans only the selected pages.
    private var scanButton: some View {
        Button {
            Task { await viewModel.scan(library: library) }
        } label: {
            Label(
                viewModel.selectedCount == 1 ? "Scan page" : "Scan \(viewModel.selectedCount) pages",
                systemImage: "wand.and.stars"
            )
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!modelStore.isReady || viewModel.selectedCount == 0)
    }

    /// Determinate progress while scanning, with the current stage label.
    private var scanProgressBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: viewModel.scanProgress)
                .progressViewStyle(.linear)
            Text(viewModel.stageLabel.isEmpty ? "Scanning…" : viewModel.stageLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    /// One queued-page thumbnail: page number, selection ring / scan status, and a
    /// remove button (when idle).
    private func queueThumbnail(index: Int, image: UIImage) -> some View {
        let status = viewModel.pageStatus[index]
        let isSelected = viewModel.selectedPages.contains(index)
        let dimmed = viewModel.isProcessing ? (status == nil) : !isSelected

        return Image(uiImage: image)
            .resizable()
            .scaledToFill()
            .frame(width: 76, height: 100)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .opacity(dimmed ? 0.4 : 1)
            .overlay(alignment: .topLeading) {
                Text("\(index + 1)")
                    .font(.caption2.weight(.bold))
                    .padding(5)
                    .background(.black.opacity(0.55), in: Circle())
                    .foregroundStyle(.white)
                    .padding(4)
            }
            .overlay(alignment: .bottomTrailing) { statusBadge(status: status, isSelected: isSelected) }
            .overlay(alignment: .topTrailing) {
                if !viewModel.isProcessing {
                    Button {
                        viewModel.removePage(at: index)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .black.opacity(0.55))
                    }
                    .padding(3)
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        status == .scanning ? Color.accentColor
                            : (!viewModel.isProcessing && isSelected ? Color.accentColor : Color.primary.opacity(0.12)),
                        lineWidth: status == .scanning || (!viewModel.isProcessing && isSelected) ? 2.5 : 1
                    )
            }
            .contentShape(RoundedRectangle(cornerRadius: 10))
            .onTapGesture {
                guard !viewModel.isProcessing else { return }
                viewModel.toggleSelection(index)
            }
    }

    /// Corner badge: a selection check (idle) or live scan status (during a scan).
    @ViewBuilder
    private func statusBadge(status: ScanViewModel.PageStatus?, isSelected: Bool) -> some View {
        if viewModel.isProcessing {
            switch status {
            case .scanning:
                ProgressView()
                    .controlSize(.small)
                    .padding(5)
                    .background(.thinMaterial, in: Circle())
                    .padding(4)
            case .done:
                badgeIcon("checkmark.circle.fill", color: .green)
            default:
                EmptyView()
            }
        } else if isSelected {
            badgeIcon("checkmark.circle.fill", color: .accentColor)
        }
    }

    /// A small filled SF Symbol badge used in the thumbnail corner.
    private func badgeIcon(_ name: String, color: Color) -> some View {
        Image(systemName: name)
            .symbolRenderingMode(.palette)
            .foregroundStyle(.white, color)
            .font(.body)
            .padding(4)
    }

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "music.note.list")
                .font(.system(size: 44))
                .foregroundStyle(.tint)
            Text("Optical Music Recognition")
                .font(.headline)
            Text("Scan sheet music and convert it to MusicXML")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        if let overlay = viewModel.overlayImage, let base = viewModel.previewImage {
            VStack(spacing: 12) {
                Picker("View", selection: $viewModel.displayMode) {
                    Text("Original").tag(ScanViewModel.DisplayMode.original)
                    Text("Detected").tag(ScanViewModel.DisplayMode.overlay)
                }
                .pickerStyle(.segmented)

                ZStack {
                    Image(uiImage: base)
                        .resizable()
                        .scaledToFit()
                    if viewModel.displayMode == .overlay {
                        Image(uiImage: overlay)
                            .resizable()
                            .scaledToFit()
                    }
                }
                .frame(maxHeight: 360)
                .clipShape(RoundedRectangle(cornerRadius: 12))

                if let summary = viewModel.resultMessage {
                    Text(summary)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                // Once saved, jump straight into the score (notation + editor).
                // Navigation is performed by the Library so the detail screen is
                // pushed on top of the bookcase rather than behind this sheet.
                if let record = viewModel.savedRecord {
                    Button {
                        onOpen(record)
                    } label: {
                        Label("Open score & editor", systemImage: "music.note.list")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }

                // Audio playback of the recognised score (synth-based).
                if viewModel.player.hasSequence {
                    PlaybackControlsView(player: viewModel.player)
                        .padding()
                        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                }

                // Offer the recognised MusicXML for export when available.
                if let musicXML = viewModel.musicXML {
                    ShareLink(
                        item: musicXML,
                        preview: SharePreview("score.musicxml")
                    ) {
                        Label("Export MusicXML", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
            }
        } else if let preview = viewModel.previewImage {
            Image(uiImage: preview)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 280)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    @ViewBuilder
    private var modelStatusCard: some View {
        switch modelStore.state {
        case .idle:
            ProgressView("Checking models…")
        case .downloading(let progress, let label):
            VStack(spacing: 8) {
                ProgressView(value: progress)
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .ready:
            Label("Models ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .font(.caption)
        }
    }
}

/// View state for photo loading and OMR processing.
@Observable
@MainActor
final class ScanViewModel {
    enum DisplayMode { case original, overlay }

    var previewImage: UIImage?
    var overlayImage: UIImage?
    var displayMode: DisplayMode = .overlay
    var isProcessing = false
    var stageLabel = ""
    var errorMessage: String?
    var resultMessage: String?
    /// The recognised MusicXML document, surfaced for export via ShareLink.
    var musicXML: String?
    /// Temp `.mid` file URL for export via ShareLink (nil when nothing sounds).
    var midiFileURL: URL?
    /// Synth-based player for the recognised score's audio playback.
    let player = ScorePlayer()
    /// The library record created for the last successful scan (drives "Open score").
    var savedRecord: ScoreRecord?

    /// Per-page progress state shown on each queued thumbnail during a scan.
    enum PageStatus: Equatable { case waiting, scanning, done }

    /// Queued page images. Multiple imports (PDFs and/or photos) accumulate here
    /// and are recognised into ONE combined score when the user confirms — the
    /// scan does not auto-start.
    var pages: [UIImage] = []
    /// Indices (into `pages`) the user has selected to include in the next scan.
    /// New imports start selected; tapping a thumbnail toggles it.
    var selectedPages: Set<Int> = []
    /// Live per-page status during a scan (keyed by page index).
    var pageStatus: [Int: PageStatus] = [:]
    /// Fraction of pages recognised so far (0…1), for the determinate bar.
    var scanProgress: Double = 0

    /// How many queued pages are currently selected for scanning.
    var selectedCount: Int { selectedPages.count }

    private let processor = OMRProcessor()
    /// Current page during a multi-page scan, for progress labelling.
    private var currentPage: (index: Int, total: Int) = (0, 0)

    /// Append photo-library picks to the scan queue (each photo = one page).
    func addPhotos(_ items: [PhotosPickerItem]) async {
        errorMessage = nil
        let firstNew = pages.count
        for item in items {
            guard let data = try? await item.loadTransferable(type: Data.self) else {
                errorMessage = "Could not load one of the selected photos."
                continue
            }
            do {
                pages.append(try SheetMusicImporter.loadImageData(data))
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        selectNewPages(from: firstNew)
    }

    /// Append document-picker files to the queue. A multi-page PDF contributes
    /// all of its pages (in order); an image contributes one page.
    func addFiles(_ urls: [URL]) async {
        errorMessage = nil
        let firstNew = pages.count
        for url in urls {
            // Document-picker URLs are security-scoped; access per file.
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            do {
                pages.append(contentsOf: try SheetMusicImporter.loadPages(from: url))
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        selectNewPages(from: firstNew)
    }

    /// Toggle whether a queued page is included in the next scan.
    func toggleSelection(_ index: Int) {
        guard pages.indices.contains(index) else { return }
        if selectedPages.contains(index) { selectedPages.remove(index) }
        else { selectedPages.insert(index) }
    }

    /// Select every queued page.
    func selectAll() { selectedPages = Set(pages.indices) }

    /// Deselect every queued page.
    func selectNone() { selectedPages.removeAll() }

    /// Remove a queued page (before scanning), re-indexing the selection.
    func removePage(at index: Int) {
        guard pages.indices.contains(index) else { return }
        pages.remove(at: index)
        // Indices after the removed one shift down by one.
        selectedPages = Set(selectedPages.compactMap { idx in
            idx == index ? nil : (idx > index ? idx - 1 : idx)
        })
        pageStatus.removeAll()
        refreshQueuePreview()
    }

    /// Empty the queue and clear any previous result.
    func clearQueue() {
        pages.removeAll()
        selectedPages.removeAll()
        pageStatus.removeAll()
        scanProgress = 0
        resetForNewImport()
        previewImage = nil
    }

    /// Mark freshly-added pages (indices `from..<count`) selected by default and
    /// refresh the standing preview image.
    private func selectNewPages(from firstNew: Int) {
        for index in firstNew..<pages.count { selectedPages.insert(index) }
        refreshQueuePreview()
    }

    /// Keep the preview showing the first queued page until a scan replaces it.
    private func refreshQueuePreview() {
        if savedRecord == nil, overlayImage == nil {
            previewImage = pages.first
        }
    }

    private func resetForNewImport() {
        errorMessage = nil
        resultMessage = nil
        musicXML = nil
        overlayImage = nil
        savedRecord = nil
        clearPlayback()
    }

    /// Stop any audio and drop the previous playback/export artefacts.
    private func clearPlayback() {
        player.stop()
        midiFileURL = nil
    }

    /// Persist MIDI bytes to a temp `.mid` so `ShareLink` can offer a real file.
    private func writeTempMIDI(_ data: Data) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("score.mid")
        do {
            try data.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }

    /// Recognise the SELECTED queued pages into ONE score. Triggered manually by
    /// the user (the scan never auto-starts); pages are chained in queue order.
    func scan(library: ScoreLibrary) async {
        // Scan the selected pages in queue order; `order[i]` maps a process-index
        // back to the original page index so we can update its thumbnail status.
        let order = selectedPages.sorted()
        guard !order.isEmpty else { return }

        isProcessing = true
        resetForNewImport()
        scanProgress = 0
        pageStatus = Dictionary(uniqueKeysWithValues: order.map { ($0, PageStatus.waiting) })
        previewImage = pages[order[0]]
        currentPage = (1, order.count)
        let images = order.map { pages[$0] }

        do {
            let result = try await processor.process(
                images: images,
                onPage: { index, total in
                    Task { @MainActor in
                        self.currentPage = (index + 1, total)
                        // Pages before this one are finished; this one is running.
                        for done in 0..<index { self.pageStatus[order[done]] = .done }
                        self.pageStatus[order[index]] = .scanning
                        self.scanProgress = total > 0 ? Double(index) / Double(total) : 0
                    }
                },
                onStage: { stage in
                    Task { @MainActor in
                        let prefix = self.currentPage.total > 1
                            ? "Page \(self.currentPage.index)/\(self.currentPage.total) · "
                            : ""
                        self.stageLabel = prefix + stage.rawValue
                    }
                }
            )
            // All selected pages recognised.
            for index in order { pageStatus[index] = .done }
            scanProgress = 1
            // Show the preprocessed (grayscale, CLAHE) image under the overlay so
            // the masks line up with exactly what the model saw.
            if let preprocessed = result.preprocessed {
                previewImage = preprocessed
            }
            overlayImage = result.overlay
            resultMessage = result.summary
            musicXML = result.musicXML
            // Load the score into the synth and stage the MIDI file for export.
            if let sequence = result.midiSequence {
                player.load(sequence)
            }
            if let midiData = result.midiData {
                midiFileURL = writeTempMIDI(midiData)
            }
            displayMode = .overlay

            // Persist the scan into the library (bookcase) when music was found.
            if let xml = result.musicXML {
                savedRecord = try? library.create(
                    title: "",
                    summary: result.summary,
                    previewImage: previewImage,
                    overlayImage: overlayImage,
                    parsedStaves: result.parsedStaves,
                    arrangement: result.arrangement,
                    musicXML: xml,
                    midiData: result.midiData
                )
                // Saved — empty the queue so the user doesn't accidentally
                // re-scan the same pages into a duplicate score.
                if savedRecord != nil {
                    pages.removeAll()
                    selectedPages.removeAll()
                    pageStatus.removeAll()
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }

        isProcessing = false
    }
}

#Preview {
    NavigationStack {
        ContentView()
            .environment(ModelStore())
            .environment(ScoreLibrary())
    }
}
