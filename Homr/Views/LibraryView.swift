import SwiftUI

// MARK: - Library (the bookcase)
//
// The app's home screen: a persistent grid of every score the user has scanned.
// Tapping a score opens it (notation, playback, editor); the "+" pushes the
// scanner. Records are restored from disk on launch by `ScoreLibrary`.

struct LibraryView: View {
    @Environment(ModelStore.self) private var modelStore
    @Environment(ScoreLibrary.self) private var library

    /// Programmatic navigation path: opening a score appends a record and the
    /// stack pushes its detail — driven from THIS stack so it always sits on top.
    @State private var path: [ScoreRecord] = []
    /// Whether the scanner sheet is presented.
    @State private var showScanner = false

    /// Two flexible columns for the bookcase grid.
    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 16)]

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                if library.records.isEmpty {
                    ContentUnavailableView {
                        Label("No scores yet", systemImage: "books.vertical")
                    } description: {
                        Text("Scan sheet music to build your library.")
                    } actions: {
                        Button {
                            showScanner = true
                        } label: {
                            Label("New Scan", systemImage: "plus")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 16) {
                            ForEach(library.records) { record in
                                NavigationLink(value: record) {
                                    ScoreTile(record: record, thumbnail: library.thumbnail(for: record))
                                }
                                .buttonStyle(.plain)
                                .contextMenu {
                                    Button(role: .destructive) {
                                        library.delete(record)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Library")
            .navigationDestination(for: ScoreRecord.self) { record in
                ScoreDetailView(record: record)
            }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showScanner = true
                    } label: {
                        Label("New Scan", systemImage: "plus")
                    }
                }
            }
            .onAppear { library.reload() }
        }
        // The scanner lives in its own modal stack. When it saves a score it
        // calls `onOpen`; we dismiss the sheet and push the detail in the
        // Library's stack so it appears on top (not hidden behind the scanner).
        .sheet(isPresented: $showScanner, onDismiss: { library.reload() }) {
            NavigationStack {
                ContentView(onOpen: { record in
                    showScanner = false
                    path = [record]
                })
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Done") { showScanner = false }
                    }
                }
            }
            .environment(modelStore)
            .environment(library)
        }
    }
}

/// One bookcase cell: thumbnail + title + subtitle.
private struct ScoreTile: View {
    let record: ScoreRecord
    let thumbnail: UIImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(.quaternary.opacity(0.4))
                if let thumbnail {
                    Image(uiImage: thumbnail)
                        .resizable()
                        .scaledToFill()
                } else {
                    Image(systemName: "music.note.list")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 160)
            .clipShape(RoundedRectangle(cornerRadius: 10))

            Text(record.title)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)

            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    /// "3 parts · Oboe, Cello" style subtitle.
    private var subtitle: String {
        var parts = "\(record.partCount) part\(record.partCount == 1 ? "" : "s")"
        if !record.instrumentNames.isEmpty {
            parts += " · " + record.instrumentNames.prefix(3).joined(separator: ", ")
        }
        return parts
    }
}
