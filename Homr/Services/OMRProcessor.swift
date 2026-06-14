import CoreGraphics
import Foundation
import UIKit

/// Orchestrates the full homr OMR pipeline on iOS.
///
/// Mirrors `homr/main.py` end-to-end:
///   1. preprocessing (CLAHE) + segmentation (segnet ONNX),
///   2. symbol detection (noteheads, staff fragments, clefs, stems, bar lines),
///   3. staff detection + grand-staff grouping,
///   4. transformer recognition (`parseStaffs`) and MusicXML export.
actor OMRProcessor {
    enum Stage: String, CaseIterable {
        case preprocessing = "Preprocessing image"
        case segmentation = "Running segmentation model"
        case staffDetection = "Detecting staffs"
        case symbolRecognition = "Recognizing symbols"
        case musicXML = "Generating MusicXML"
    }

    enum ProcessingError: LocalizedError {
        case preprocessingFailed
        case pipeline(String)
        case underlying(Error)

        var errorDescription: String? {
            switch self {
            case .preprocessingFailed:
                return "Could not preprocess the image."
            case .pipeline(let message):
                return message
            case .underlying(let error):
                return error.localizedDescription
            }
        }
    }

    struct Result {
        /// Coloured overlay of the segmentation classes (for display).
        let overlay: UIImage?
        /// The preprocessed grayscale image the model actually saw.
        let preprocessed: UIImage?
        /// Human-readable summary of what was recognised.
        let summary: String
        /// The generated MusicXML document, if recognition produced one.
        let musicXML: String?
        /// Playable note sequence converted from the MusicXML, if any.
        let midiSequence: MIDISequence?
        /// Standard MIDI File (`.mid`) bytes for export, if any.
        let midiData: Data?
        /// Per-physical-staff recognition results (cached for editing/persistence).
        let parsedStaves: [ParsedStaff]
        /// How staves are grouped into instrument parts (editable downstream).
        let arrangement: Arrangement
    }

    /// Per-page recognition output (stages 1–3): the cached staves plus how many
    /// systems the page held (used to offset indices when merging pages) and the
    /// page's preview/overlay images.
    private struct PageRecognition {
        var parsedStaves: [ParsedStaff]
        var systemCount: Int
        let overlay: UIImage?
        let preprocessed: UIImage?
        let segSummary: String
    }

    /// Convenience: run the pipeline on a single image (camera / single import).
    func process(image: UIImage, onStage: @escaping @Sendable (Stage) -> Void) async throws -> Result {
        try await process(images: [image], onStage: onStage)
    }

    /// Runs the OMR pipeline across one or more page images and assembles them
    /// into a SINGLE score. Pages are recognised independently (stages 1–3), then
    /// their staves are concatenated — each page's `systemIndex`/`id` offset by
    /// the running totals — so the pages stack vertically in reading order and
    /// the parts (instruments) carry through every page. Stage 4 (arrangement +
    /// MusicXML/MIDI) then runs once over the combined staves.
    ///
    /// - Parameters:
    ///   - images: page bitmaps in reading order (e.g. all pages of a PDF).
    ///   - onPage: reports `(pageIndex, totalPages)` as each page starts.
    ///   - onStage: reports the current pipeline stage (per page for 1–3).
    func process(
        images: [UIImage],
        onPage: @escaping @Sendable (_ index: Int, _ total: Int) -> Void = { _, _ in },
        onStage: @escaping @Sendable (Stage) -> Void
    ) async throws -> Result {
        guard !images.isEmpty else { throw ProcessingError.pipeline("No pages to scan.") }

        var combined: [ParsedStaff] = []
        var systemOffset = 0
        var idOffset = 0
        var firstOverlay: UIImage?
        var firstPreprocessed: UIImage?
        let total = images.count

        // --- Recognise each page, offsetting indices so they merge cleanly ---
        for (pageIndex, image) in images.enumerated() {
            onPage(pageIndex, total)
            let page = try recognizePage(image: image, onStage: onStage)
            if firstPreprocessed == nil {
                firstPreprocessed = page.preprocessed
                firstOverlay = page.overlay
            }
            for staff in page.parsedStaves {
                combined.append(ParsedStaff(
                    id: staff.id + idOffset,
                    systemIndex: staff.systemIndex + systemOffset,
                    orderInSystem: staff.orderInSystem,
                    // Stamp the real source page so the score can be re-paginated.
                    pageIndex: pageIndex,
                    minX: staff.minX, minY: staff.minY, maxX: staff.maxX, maxY: staff.maxY,
                    label: staff.label,
                    symbols: staff.symbols
                ))
            }
            systemOffset += page.systemCount
            idOffset += page.parsedStaves.count
        }

        // No music found on any page: surface a friendly summary.
        guard !combined.isEmpty else {
            return Result(
                overlay: firstOverlay,
                preprocessed: firstPreprocessed,
                summary: "Segmentation succeeded but no music was found.",
                musicXML: nil,
                midiSequence: nil,
                midiData: nil,
                parsedStaves: [],
                arrangement: Arrangement(partCount: 1, staffToPart: [:], partInstruments: [:], mutedParts: [])
            )
        }

        // --- Stage 4: arrangement + MusicXML/MIDI export (over all pages) ---
        onStage(.musicXML)
        let arrangement = Arrangement.makeDefault(combined)
        let score = buildScore(from: combined, arrangement: arrangement, title: "")

        let systemCount = (combined.map { $0.systemIndex }.max() ?? -1) + 1
        let symbolCount = combined.reduce(0) { $0 + $1.symbols.count }
        let identifiedNames = arrangement.partInstruments
            .sorted { $0.key < $1.key }
            .map { $0.value.name }
        let instrumentLine = identifiedNames.isEmpty
            ? ""
            : "\nInstruments: \(identifiedNames.joined(separator: ", "))"
        let pageLine = total > 1 ? "Pages: \(total) · " : ""
        let summary = """
        Recognition complete.
        \(pageLine)Systems: \(systemCount) · Staves: \(combined.count) · Parts: \(arrangement.partCount) · Symbols: \(symbolCount)\(instrumentLine)
        """

        return Result(
            overlay: firstOverlay,
            preprocessed: firstPreprocessed,
            summary: summary,
            musicXML: score.musicXML,
            midiSequence: score.midiSequence,
            midiData: score.midiData,
            parsedStaves: combined,
            arrangement: arrangement
        )
    }

    /// Stages 1–3 for a single page. Returns empty staves (rather than throwing)
    /// when a page has no detectable music, so a multi-page scan can skip blank /
    /// title pages and keep going.
    private func recognizePage(image: UIImage, onStage: @escaping @Sendable (Stage) -> Void) throws -> PageRecognition {
        // --- Stage 1: preprocessing -----------------------------------------
        onStage(.preprocessing)
        guard let preprocessed = ImagePreprocessor.preprocess(image) else {
            throw ProcessingError.preprocessingFailed
        }

        // --- Stage 1b: segmentation (segnet) --------------------------------
        onStage(.segmentation)
        let segnetSession = try ONNXSessionFactory.makeSegnetSession()
        let segnet = SegnetInference(session: segnetSession)
        let maps = try segnet.run(on: preprocessed)

        func mask(_ pixels: [UInt8]) -> GrayscaleImage {
            GrayscaleImage(pixels: pixels, width: maps.width, height: maps.height)
        }
        var predictions = InputPredictions(
            original: preprocessed,
            preprocessed: preprocessed,
            notehead: mask(maps.noteheads),
            symbols: mask(maps.symbols),
            staff: mask(maps.staff),
            clefsKeys: mask(maps.clefsKeys),
            stemsRest: mask(maps.stemsRests)
        )
        predictions = filterPredictions(predictions)
        predictions = withStrongerStaffLines(predictions)

        let overlay = maps.makeOverlay().map { UIImage(cgImage: $0) }
        let preview = preprocessed.makeCGImage().map { UIImage(cgImage: $0) }

        // --- Stage 2: symbol + staff detection ------------------------------
        onStage(.staffDetection)
        let multiStaffs: [MultiStaff]
        do {
            multiStaffs = try detectStaffsInImage(predictions)
        } catch is ProcessingError {
            // No noteheads / staffs on this page — return empty so the caller can
            // continue with the next page (or report "no music" if all are empty).
            return PageRecognition(
                parsedStaves: [], systemCount: 0,
                overlay: overlay, preprocessed: preview, segSummary: maps.summary
            )
        }

        // --- Stage 3: transformer recognition (one pass per physical staff) --
        onStage(.symbolRecognition)
        let transformerConfig = TransformerConfig()
        let parsedStaves = parseIndividualStaves(
            staffs: multiStaffs,
            image: predictions.preprocessed,
            labelImage: predictions.preprocessed,
            config: transformerConfig
        )
        let systemCount = (parsedStaves.map { $0.systemIndex }.max() ?? -1) + 1
        return PageRecognition(
            parsedStaves: parsedStaves, systemCount: systemCount,
            overlay: overlay, preprocessed: preview, segSummary: maps.summary
        )
    }

    // MARK: - Pipeline helpers

    /// Rebuilds the predictions bundle with thickened/binarised staff lines.
    /// `InputPredictions` is immutable, so we construct a fresh instance.
    private func withStrongerStaffLines(_ predictions: InputPredictions) -> InputPredictions {
        let strongerStaff = makeLinesStronger(predictions.staff, kernelSize: (width: 1, height: 2))
        return InputPredictions(
            original: predictions.original,
            preprocessed: predictions.preprocessed,
            notehead: predictions.notehead,
            symbols: predictions.symbols,
            staff: strongerStaff,
            clefsKeys: predictions.clefsKeys,
            stemsRest: predictions.stemsRest
        )
    }

    /// Symbol + staff detection — port of `detect_staffs_in_image` (debug dropped).
    /// Returns the grand-staff-grouped `MultiStaff`s ready for the transformer.
    private func detectStaffsInImage(_ predictions: InputPredictions) throws -> [MultiStaff] {
        // predict_symbols: derive bounding boxes from each mask.
        let noteheads = createBoundingEllipses(predictions.notehead, minSize: (4, 4))
        var staffFragments = createRotatedBoundingBoxes(
            predictions.staff, skipMerging: true, minSize: (5, 1), maxSize: (10000, 100)
        )
        let clefsKeys = createRotatedBoundingBoxes(
            predictions.clefsKeys, minSize: (20, 40), maxSize: (1000, 1000)
        )
        let stemsRest = createRotatedBoundingBoxes(predictions.stemsRest)
        let barLineImg = prepareBarLineImage(predictions.stemsRest)
        let barLines = createRotatedBoundingBoxes(barLineImg, skipMerging: true, minSize: (1, 5))

        // Break overly wide (curved) staff-line fragments into smaller parts.
        staffFragments = breakWideFragments(staffFragments)

        // Pair noteheads with their stems.
        let noteheadsWithStems = combineNoteheadsWithStems(noteheads: noteheads, stems: stemsRest)
        if noteheadsWithStems.isEmpty {
            throw ProcessingError.pipeline("No noteheads found")
        }

        // Median notehead height drives the bar-line size gate.
        let averageNoteHeadHeight = medianOf(noteheadsWithStems.map { $0.notehead.size.height })

        // Bar-line candidates are stem/rest boxes that aren't part of a note.
        let allNoteheads = noteheadsWithStems.map { $0.notehead }
        let allStems = noteheadsWithStems.compactMap { $0.stem }
        let barLinesOrRests = barLines.filter {
            !$0.isOverlappingWithAny(allNoteheads) && !$0.isOverlappingWithAny(allStems)
        }
        let barLineBoxes = detectBarLines(barLines: barLinesOrRests, unitSize: averageNoteHeadHeight)

        // Detect staffs from the strengthened staff mask + anchor symbols.
        let staffs = detectStaff(
            image: predictions.staff,
            staffFragments: staffFragments,
            clefsKeys: clefsKeys,
            likelyBarOrRestsLines: barLineBoxes
        )
        if staffs.isEmpty {
            throw ProcessingError.pipeline("No staffs found")
        }

        // Brace/bracket/grand-staff detection.
        let braceDotImg = prepareBraceDotImage(symbols: predictions.symbols, staff: predictions.staff)
        let braceDot = createRotatedBoundingBoxes(braceDotImg, skipMerging: true, maxSize: (100, -1))

        // Assign notes to staffs in place (return value not needed downstream).
        _ = addNotesToStaffs(
            staffs: staffs,
            noteheads: noteheadsWithStems,
            symbols: predictions.symbols,
            noteheadPred: predictions.notehead
        )

        return findBracesBracketsAndGrandStaffLines(staffs: staffs, braceDot: braceDot)
    }

    /// Median of a list of doubles (`np.median`), 0 for an empty input.
    private func medianOf(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let n = sorted.count
        if n % 2 == 1 { return sorted[n / 2] }
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }
}
