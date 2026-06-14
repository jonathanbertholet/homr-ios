import Observation
import SwiftUI
import UIKit

// MARK: - Score library (the "bookcase")
//
// Persists every recognised score so scans survive app relaunches. Each score is
// a folder under Application Support containing a small JSON manifest plus the
// heavy artefacts (images, MusicXML, MIDI, the cached per-staff recognition and
// the editable arrangement). The manifest is what the bookcase grid lists; the
// heavy parts are loaded lazily only when a score is opened.

/// Lightweight, listable metadata for one saved score (the manifest).
struct ScoreRecord: Codable, Identifiable, Sendable, Hashable {
    let id: UUID
    var title: String
    let createdAt: Date
    var summary: String
    /// Names of the identified instruments (for subtitle display).
    var instrumentNames: [String]
    /// Number of parallel parts in the arrangement.
    var partCount: Int
}

/// The full, lazily-loaded contents of a saved score (used by the detail screen).
struct ScoreDetail: Sendable {
    var record: ScoreRecord
    var previewImage: UIImage?
    var overlayImage: UIImage?
    var parsedStaves: [ParsedStaff]
    var arrangement: Arrangement
    var musicXML: String
    var midiURL: URL?
}

/// On-disk store + observable list backing the library UI.
@Observable
@MainActor
final class ScoreLibrary {
    /// Manifests, newest first. Drives the bookcase grid.
    private(set) var records: [ScoreRecord] = []

    /// Root directory holding one sub-folder per score.
    private let baseURL: URL

    // File names used inside each score's folder.
    private enum File {
        static let manifest = "manifest.json"
        static let preview = "preview.png"
        static let overlay = "overlay.png"
        static let staves = "staves.json"
        static let arrangement = "arrangement.json"
        static let musicXML = "score.musicxml"
        static let midi = "score.mid"
    }

    init() {
        // Application Support is the right home for app-managed, user-invisible data.
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        baseURL = support.appendingPathComponent("ScoreLibrary", isDirectory: true)
        try? FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        reload()
    }

    // MARK: - Listing

    /// Re-scan the library folder and decode every manifest (newest first).
    func reload() {
        let fm = FileManager.default
        guard let folders = try? fm.contentsOfDirectory(
            at: baseURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else {
            records = []
            return
        }
        var loaded: [ScoreRecord] = []
        for folder in folders where folder.hasDirectoryPath {
            let manifestURL = folder.appendingPathComponent(File.manifest)
            guard let data = try? Data(contentsOf: manifestURL),
                  let record = try? JSONDecoder.iso().decode(ScoreRecord.self, from: data) else { continue }
            loaded.append(record)
        }
        records = loaded.sorted { $0.createdAt > $1.createdAt }
    }

    // MARK: - Saving

    /// Persist a freshly recognised score and return its manifest.
    @discardableResult
    func create(
        title: String,
        summary: String,
        previewImage: UIImage?,
        overlayImage: UIImage?,
        parsedStaves: [ParsedStaff],
        arrangement: Arrangement,
        musicXML: String,
        midiData: Data?
    ) throws -> ScoreRecord {
        let id = UUID()
        let folder = baseURL.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let record = ScoreRecord(
            id: id,
            title: title.isEmpty ? defaultTitle() : title,
            createdAt: Date(),
            summary: summary,
            instrumentNames: arrangement.partInstruments.sorted { $0.key < $1.key }.map { $0.value.name },
            partCount: arrangement.partCount
        )

        try write(record: record, folder: folder)
        try writeAssets(
            folder: folder, previewImage: previewImage, overlayImage: overlayImage,
            parsedStaves: parsedStaves, arrangement: arrangement,
            musicXML: musicXML, midiData: midiData
        )

        reload()
        return record
    }

    /// Persist an edited arrangement (and the regenerated XML/MIDI) for an
    /// existing score, refreshing its manifest fields.
    func update(
        id: UUID,
        arrangement: Arrangement,
        musicXML: String,
        midiData: Data?
    ) throws {
        let folder = baseURL.appendingPathComponent(id.uuidString, isDirectory: true)
        guard var record = records.first(where: { $0.id == id }) else { return }

        record.instrumentNames = arrangement.partInstruments.sorted { $0.key < $1.key }.map { $0.value.name }
        record.partCount = arrangement.partCount

        try write(record: record, folder: folder)
        try Data(arrangementJSON: arrangement).write(to: folder.appendingPathComponent(File.arrangement))
        try musicXML.data(using: .utf8)?.write(to: folder.appendingPathComponent(File.musicXML))
        let midiURL = folder.appendingPathComponent(File.midi)
        if let midiData {
            try midiData.write(to: midiURL)
        } else {
            try? FileManager.default.removeItem(at: midiURL)
        }
        reload()
    }

    /// Rename a saved score.
    func rename(id: UUID, to newTitle: String) throws {
        let folder = baseURL.appendingPathComponent(id.uuidString, isDirectory: true)
        guard var record = records.first(where: { $0.id == id }) else { return }
        record.title = newTitle.isEmpty ? record.title : newTitle
        try write(record: record, folder: folder)
        reload()
    }

    /// Delete a saved score and all of its files.
    func delete(_ record: ScoreRecord) {
        let folder = baseURL.appendingPathComponent(record.id.uuidString, isDirectory: true)
        try? FileManager.default.removeItem(at: folder)
        reload()
    }

    // MARK: - Loading detail

    /// Load the full contents of a saved score for display/editing.
    func detail(for record: ScoreRecord) throws -> ScoreDetail {
        let folder = baseURL.appendingPathComponent(record.id.uuidString, isDirectory: true)

        let staves: [ParsedStaff]
        if let data = try? Data(contentsOf: folder.appendingPathComponent(File.staves)) {
            staves = (try? JSONDecoder().decode([ParsedStaff].self, from: data)) ?? []
        } else {
            staves = []
        }

        let arrangement: Arrangement
        if let data = try? Data(contentsOf: folder.appendingPathComponent(File.arrangement)),
           let decoded = try? JSONDecoder().decode(Arrangement.self, from: data) {
            arrangement = decoded
        } else {
            arrangement = Arrangement.makeDefault(staves)
        }

        let xml = (try? String(contentsOf: folder.appendingPathComponent(File.musicXML), encoding: .utf8)) ?? ""
        let midiURL = folder.appendingPathComponent(File.midi)
        let hasMIDI = FileManager.default.fileExists(atPath: midiURL.path)

        return ScoreDetail(
            record: record,
            previewImage: UIImage(contentsOfFile: folder.appendingPathComponent(File.preview).path),
            overlayImage: UIImage(contentsOfFile: folder.appendingPathComponent(File.overlay).path),
            parsedStaves: staves,
            arrangement: arrangement,
            musicXML: xml,
            midiURL: hasMIDI ? midiURL : nil
        )
    }

    /// Bundle-relative URL of a score's exportable MIDI, if present.
    func midiURL(for record: ScoreRecord) -> URL? {
        let url = baseURL
            .appendingPathComponent(record.id.uuidString, isDirectory: true)
            .appendingPathComponent(File.midi)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Thumbnail (preview image) for the bookcase grid.
    func thumbnail(for record: ScoreRecord) -> UIImage? {
        let url = baseURL
            .appendingPathComponent(record.id.uuidString, isDirectory: true)
            .appendingPathComponent(File.preview)
        return UIImage(contentsOfFile: url.path)
    }

    // MARK: - Private helpers

    private func write(record: ScoreRecord, folder: URL) throws {
        let data = try JSONEncoder.iso().encode(record)
        try data.write(to: folder.appendingPathComponent(File.manifest))
    }

    private func writeAssets(
        folder: URL,
        previewImage: UIImage?,
        overlayImage: UIImage?,
        parsedStaves: [ParsedStaff],
        arrangement: Arrangement,
        musicXML: String,
        midiData: Data?
    ) throws {
        if let png = previewImage?.pngData() {
            try png.write(to: folder.appendingPathComponent(File.preview))
        }
        if let png = overlayImage?.pngData() {
            try png.write(to: folder.appendingPathComponent(File.overlay))
        }
        try JSONEncoder().encode(parsedStaves).write(to: folder.appendingPathComponent(File.staves))
        try Data(arrangementJSON: arrangement).write(to: folder.appendingPathComponent(File.arrangement))
        try musicXML.data(using: .utf8)?.write(to: folder.appendingPathComponent(File.musicXML))
        if let midiData {
            try midiData.write(to: folder.appendingPathComponent(File.midi))
        }
    }

    /// Default title derived from the current date (e.g. "Score · Jun 14, 14:30").
    private func defaultTitle() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, HH:mm"
        return "Score · \(formatter.string(from: Date()))"
    }
}

// MARK: - JSON helpers

private extension JSONEncoder {
    /// Encoder using ISO-8601 dates (so manifests are human-readable & stable).
    static func iso() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static func iso() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

private extension Data {
    /// Convenience: encode an `Arrangement` to JSON `Data`.
    init(arrangementJSON arrangement: Arrangement) throws {
        self = try JSONEncoder().encode(arrangement)
    }
}
