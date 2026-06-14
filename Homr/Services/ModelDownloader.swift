import Foundation
import ZIPFoundation

/// Downloads and extracts homr ONNX model archives from GitHub releases.
struct ModelDownloader {
    enum DownloadError: LocalizedError {
        case badHTTPStatus(Int)
        case unzipFailed(String)

        var errorDescription: String? {
            switch self {
            case .badHTTPStatus(let code):
                return "Model download failed (HTTP \(code))."
            case .unzipFailed(let name):
                return "Failed to extract \(name)."
            }
        }
    }

    func downloadAll(
        missing: [ModelDescriptor],
        onProgress: @escaping (Double, String) -> Void
    ) async throws {
        let modelsDir = HomrModelPaths.modelsDirectory()
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        for (index, model) in missing.enumerated() {
            let fractionStart = Double(index) / Double(missing.count)
            let fractionEnd = Double(index + 1) / Double(missing.count)

            onProgress(fractionStart, "Downloading \(model.fileName)…")

            let zipURL = modelsDir.appendingPathComponent(model.zipName)
            try await downloadFile(from: model.downloadURL, to: zipURL) { fileProgress in
                let overall = fractionStart + (fractionEnd - fractionStart) * fileProgress
                onProgress(overall, "Downloading \(model.fileName)…")
            }

            onProgress(fractionStart + (fractionEnd - fractionStart) * 0.9, "Extracting \(model.fileName)…")
            try unzipArchive(at: zipURL, to: modelsDir)

            // Remove zip after successful extraction to save space.
            try? FileManager.default.removeItem(at: zipURL)

            if !FileManager.default.fileExists(atPath: model.localURL.path) {
                throw DownloadError.unzipFailed(model.fileName)
            }
        }

        onProgress(1.0, "Models ready")
    }

    private func downloadFile(
        from url: URL,
        to destination: URL,
        onProgress: @escaping (Double) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        request.timeoutInterval = 300

        let (tempURL, response) = try await URLSession.shared.download(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw DownloadError.badHTTPStatus(-1)
        }
        guard (200...299).contains(http.statusCode) else {
            throw DownloadError.badHTTPStatus(http.statusCode)
        }

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
        onProgress(1.0)
    }

    private func unzipArchive(at zipURL: URL, to destination: URL) throws {
        try FileManager.default.unzipItem(at: zipURL, to: destination)
    }
}
