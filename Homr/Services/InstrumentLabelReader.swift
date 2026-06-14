import CoreGraphics
import Foundation
import Vision

// MARK: - Instrument label OCR
//
// Reads the printed instrument names that sit in the left margin of the first
// system (e.g. "Oboe d'amore I", "Basso", "Continuo"). On engraved scores these
// appear only on the first system, so we OCR the margin of each staff there and
// rely on the pipeline's stable staff order (voice i == i-th staff, top→bottom)
// to propagate identity to every later system.
//
// We OCR the preprocessed grayscale image because staff coordinates live in that
// space, so no coordinate transform is needed.
enum InstrumentLabelReader {
    /// OCR the left-margin label for each staff (already in top→bottom order).
    /// Returns one entry per staff; nil where nothing legible was found.
    static func readLabels(image: GrayscaleImage, staves: [Staff]) -> [String?] {
        staves.map { readLabel(image: image, staff: $0) }
    }

    /// OCR the left-margin label for a single staff (nil when nothing legible).
    static func readLabel(image: GrayscaleImage, staff: Staff) -> String? {
        // The label lives left of where the staff lines begin. Pad vertically by
        // ~1.5 unit sizes so tall names aren't clipped.
        let pad = max(staff.averageUnitSize * 1.5, 6)
        let x1 = staff.minX
        let y0 = staff.minY - pad
        let y1 = staff.maxY + pad
        let rect = CGRect(x: 0, y: y0, width: x1, height: y1 - y0)

        // Need a reasonably sized margin to contain any text.
        guard rect.width > 12, rect.height > 12 else { return nil }

        let crop = image.cropped(to: rect)
        guard let cgImage = crop.makeCGImage() else { return nil }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        // Instrument names are proper nouns/abbreviations — don't "correct" them.
        request.usesLanguageCorrection = false
        // Common engraving languages for instrument names.
        request.recognitionLanguages = ["it-IT", "de-DE", "en-US", "fr-FR"]

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        // `VNRecognizeTextRequest.results` is already `[VNRecognizedTextObservation]?`.
        guard let observations = request.results, !observations.isEmpty else {
            return nil
        }
        let text = observations
            .compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }
}
