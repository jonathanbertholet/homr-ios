import Foundation
import Observation

/// Tracks ONNX model download state and exposes paths once ready.
@Observable
@MainActor
final class ModelStore {
    enum State: Equatable {
        case idle
        case downloading(progress: Double, label: String)
        case ready
        case failed(String)
    }

    private(set) var state: State = .idle
    private let downloader = ModelDownloader()

    var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    /// Ensures all ONNX models are present on disk; downloads missing zips from GitHub releases.
    func prepareModelsIfNeeded() async {
        guard !isReady else { return }

        let missing = HomrModelPaths.requiredModels.filter {
            !FileManager.default.fileExists(atPath: $0.localURL.path)
        }

        if missing.isEmpty {
            state = .ready
            return
        }

        state = .downloading(progress: 0, label: "Preparing models…")

        do {
            try await downloader.downloadAll(missing: missing) { [weak self] progress, label in
                Task { @MainActor in
                    self?.state = .downloading(progress: progress, label: label)
                }
            }
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}
