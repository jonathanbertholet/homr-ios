import Foundation
import UniformTypeIdentifiers
import UIKit

/// Normalises photo-library picks and document-picker files into a `UIImage`
/// the OMR pipeline can consume (images pass through; PDFs are rasterized).
enum SheetMusicImporter {
    enum ImportError: LocalizedError {
        case unsupportedType
        case unreadableImage

        var errorDescription: String? {
            switch self {
            case .unsupportedType:
                return "Unsupported file type. Choose a PDF, PNG, JPEG, HEIC, or TIFF."
            case .unreadableImage:
                return "Could not load the image."
            }
        }
    }

    /// UTTypes offered by the document picker.
    static let supportedTypes: [UTType] = [
        .pdf,
        .png,
        .jpeg,
        .heic,
        .tiff,
    ]

    /// Loads an on-disk file (from `.fileImporter`) into a preview/process image.
    ///
    /// - Parameters:
    ///   - url: Security-scoped URL returned by the document picker.
    ///   - pdfPageIndex: Zero-based page when `url` is a PDF.
    /// - Returns: The rasterized or decoded image plus PDF metadata (page count is
    ///   1 for non-PDF imports).
    static func load(
        from url: URL,
        pdfPageIndex: Int = 0
    ) throws -> (image: UIImage, pageCount: Int, pageIndex: Int) {
        let ext = url.pathExtension.lowercased()
        if ext == "pdf" {
            let pageCount = PDFRasterizer.pageCount(at: url)
            let image = try PDFRasterizer.rasterize(url: url, pageIndex: pdfPageIndex)
            return (image, pageCount, pdfPageIndex)
        }

        guard let data = try? Data(contentsOf: url),
              let image = UIImage(data: data) else {
            throw ImportError.unreadableImage
        }
        return (image, 1, 0)
    }

    /// Loads raw image bytes (from `PhotosPicker`) — unchanged behaviour.
    static func loadImageData(_ data: Data) throws -> UIImage {
        guard let image = UIImage(data: data) else {
            throw ImportError.unreadableImage
        }
        return image
    }

    /// Loads a file as an ordered list of page images: every page of a PDF (so a
    /// multi-page PDF becomes one multi-page scan), or a single-element list for a
    /// plain image. Used by the multi-page import queue.
    static func loadPages(from url: URL) throws -> [UIImage] {
        let ext = url.pathExtension.lowercased()
        if ext == "pdf" {
            let count = PDFRasterizer.pageCount(at: url)
            guard count > 0 else { throw ImportError.unreadableImage }
            return try (0..<count).map { try PDFRasterizer.rasterize(url: url, pageIndex: $0) }
        }
        guard let data = try? Data(contentsOf: url), let image = UIImage(data: data) else {
            throw ImportError.unreadableImage
        }
        return [image]
    }
}
