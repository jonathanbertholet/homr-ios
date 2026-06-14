import CoreGraphics
import Foundation
import PDFKit
import UIKit

/// Renders PDF pages to bitmaps for the OMR pipeline.
///
/// Sheet-music PDFs are vector; homr expects a raster image. We render at 300 DPI
/// (the same resolution used during empirical validation) so staff lines and
/// noteheads stay sharp enough for segnet.
enum PDFRasterizer {
    /// Default render resolution — 300 DPI matches our validated Bach fixture.
    static let defaultDPI: CGFloat = 300

    enum RasterizeError: LocalizedError {
        case unreadablePDF
        case pageOutOfRange(requested: Int, pageCount: Int)
        case renderFailed

        var errorDescription: String? {
            switch self {
            case .unreadablePDF:
                return "Could not open the PDF."
            case .pageOutOfRange(let requested, let pageCount):
                return "Page \(requested + 1) is out of range (PDF has \(pageCount) page\(pageCount == 1 ? "" : "s"))."
            case .renderFailed:
                return "Could not render the PDF page."
            }
        }
    }

    /// Returns how many pages the document contains (0 if unreadable).
    static func pageCount(at url: URL) -> Int {
        PDFDocument(url: url)?.pageCount ?? 0
    }

    /// Rasterizes a single PDF page to a UIImage at the given DPI.
    ///
    /// PDFKit uses a bottom-left origin; we flip the context so the bitmap matches
    /// the usual top-left image layout that `GrayscaleImage` / UIKit expect.
    static func rasterize(
        url: URL,
        pageIndex: Int = 0,
        dpi: CGFloat = defaultDPI
    ) throws -> UIImage {
        guard let document = PDFDocument(url: url) else {
            throw RasterizeError.unreadablePDF
        }
        let count = document.pageCount
        guard pageIndex >= 0, pageIndex < count else {
            throw RasterizeError.pageOutOfRange(requested: pageIndex, pageCount: count)
        }
        guard let page = document.page(at: pageIndex) else {
            throw RasterizeError.renderFailed
        }

        // Scale from PDF points (72 DPI) to target DPI.
        let pageRect = page.bounds(for: .mediaBox)
        let scale = dpi / 72.0
        let pixelSize = CGSize(
            width: max(1, (pageRect.width * scale).rounded()),
            height: max(1, (pageRect.height * scale).rounded())
        )

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true

        let renderer = UIGraphicsImageRenderer(size: pixelSize, format: format)
        let image = renderer.image { context in
            // White paper background (transparent PDF regions become white).
            UIColor.white.setFill()
            context.fill(CGRect(origin: .zero, size: pixelSize))

            let cg = context.cgContext
            cg.saveGState()
            // Flip from PDF's bottom-left origin to UIKit's top-left origin.
            cg.translateBy(x: 0, y: pixelSize.height)
            cg.scaleBy(x: scale, y: -scale)
            page.draw(with: .mediaBox, to: cg)
            cg.restoreGState()
        }

        return image
    }
}
