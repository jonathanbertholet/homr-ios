import CoreGraphics
import Foundation
import ImageIO
import OnnxRuntimeBindings

// Headless validation entry point. Mirrors `ImagePreprocessor.preprocess` and
// `OMRProcessor.process`/`detectStaffsInImage` (which can't be reused directly
// because they import UIKit) so the ported pipeline runs unchanged on macOS.

// MARK: - Image loading (CoreGraphics / ImageIO, no UIKit)

/// Loads any image file into a grayscale buffer via ImageIO + CoreGraphics.
func loadGrayscale(path: String) -> GrayscaleImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }
    return GrayscaleImage(cgImage: cgImage)
}

// MARK: - Preprocessing (port of ImagePreprocessor)

private let targetWidth = 1920

/// Verbatim port of `ImagePreprocessor.autocrop` (UIKit-free).
func autocrop(_ image: GrayscaleImage) -> GrayscaleImage {
    let width = image.width
    let height = image.height
    guard width > 0, height > 0 else { return image }

    // Dominant gray value (the paper/background peak in the histogram).
    var histogram = [Int](repeating: 0, count: 256)
    for value in image.pixels { histogram[Int(value)] += 1 }
    let dominant = histogram.firstIndex(of: histogram.max() ?? 0) ?? 0
    let threshold = UInt8(max(0, dominant - 30))

    // Bounding box of all foreground (brighter-than-threshold) pixels.
    var minX = width, minY = height, maxX = 0, maxY = 0
    var found = false
    image.pixels.withUnsafeBufferPointer { src in
        for y in 0..<height {
            let row = y * width
            for x in 0..<width where src[row + x] > threshold {
                found = true
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }
        }
    }

    guard found, maxX > minX, maxY > minY else { return image }

    // Skip the crop for full-page views (box hugs the top-left corner).
    let isFullPageView = minX < Int(Double(width) * 0.25) || minY < Int(Double(height) * 0.25)
    if isFullPageView { return image }

    let rect = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    return image.cropped(to: rect)
}

/// Verbatim port of `ImagePreprocessor.preprocess` operating on a buffer.
func preprocess(_ image: GrayscaleImage) -> GrayscaleImage {
    var gray = autocrop(image)
    gray = gray.resized(toWidth: targetWidth)
    CLAHE.apply(to: &gray)
    return gray
}

// MARK: - Detection orchestration (port of OMRProcessor)

enum PipelineError: Error, CustomStringConvertible {
    case noNoteheads
    case noStaffs
    var description: String {
        switch self {
        case .noNoteheads: return "No noteheads found"
        case .noStaffs: return "No staffs found"
        }
    }
}

/// Count "on" pixels (>127) per horizontal band — used to see whether the staff
/// mask survives in the lower portion of the page.
func bandCounts(_ image: GrayscaleImage, bands: Int = 5) -> [Int] {
    var counts = [Int](repeating: 0, count: bands)
    let h = image.height, w = image.width
    guard h > 0, w > 0 else { return counts }
    image.pixels.withUnsafeBufferPointer { src in
        for y in 0..<h {
            let band = min(bands - 1, y * bands / h)
            let row = y * w
            for x in 0..<w where src[row + x] > 127 { counts[band] += 1 }
        }
    }
    return counts
}

/// Median of a list of doubles (`np.median`), 0 for an empty input.
func medianOf(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let n = sorted.count
    if n % 2 == 1 { return sorted[n / 2] }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}

/// Rebuilds the predictions bundle with thickened/binarised staff lines.
func withStrongerStaffLines(_ predictions: InputPredictions) -> InputPredictions {
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

/// Verbatim port of `OMRProcessor.detectStaffsInImage` (debug drawing dropped).
func detectStaffsInImage(_ predictions: InputPredictions) throws -> [MultiStaff] {
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

    staffFragments = breakWideFragments(staffFragments)

    let noteheadsWithStems = combineNoteheadsWithStems(noteheads: noteheads, stems: stemsRest)
    if noteheadsWithStems.isEmpty { throw PipelineError.noNoteheads }

    let averageNoteHeadHeight = medianOf(noteheadsWithStems.map { $0.notehead.size.height })

    let allNoteheads = noteheadsWithStems.map { $0.notehead }
    let allStems = noteheadsWithStems.compactMap { $0.stem }
    let barLinesOrRests = barLines.filter {
        !$0.isOverlappingWithAny(allNoteheads) && !$0.isOverlappingWithAny(allStems)
    }
    let barLineBoxes = detectBarLines(barLines: barLinesOrRests, unitSize: averageNoteHeadHeight)

    let staffs = detectStaff(
        image: predictions.staff,
        staffFragments: staffFragments,
        clefsKeys: clefsKeys,
        likelyBarOrRestsLines: barLineBoxes
    )
    if staffs.isEmpty { throw PipelineError.noStaffs }
    // Diagnostic: raw staff count + y-centers BEFORE brace/grand-staff grouping,
    // to distinguish "detection missed staves" from "grouping merged them".
    let rawCenters = staffs.map { Int(($0.minY + $0.maxY) / 2) }.sorted()
    log("raw staffs (pre-group): \(staffs.count)  y-centers=\(rawCenters)")

    let braceDotImg = prepareBraceDotImage(symbols: predictions.symbols, staff: predictions.staff)
    let braceDot = createRotatedBoundingBoxes(braceDotImg, skipMerging: true, maxSize: (100, -1))

    _ = addNotesToStaffs(
        staffs: staffs,
        noteheads: noteheadsWithStems,
        symbols: predictions.symbols,
        noteheadPred: predictions.notehead
    )

    return findBracesBracketsAndGrandStaffLines(staffs: staffs, braceDot: braceDot)
}

// MARK: - Benchmarking

/// Monotonic wall time in seconds (matches Python `perf_counter`).
func benchNow() -> Double {
    Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1_000_000_000
}

func benchMedian(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let n = sorted.count
    if n % 2 == 1 { return sorted[n / 2] }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}

/// Runs timed segnet / detection / transformer passes when `HOMR_BENCHMARK=1`.
func runBenchmark(imagePath: String, outPath: String) throws {
    let warmup = Int(ProcessInfo.processInfo.environment["HOMR_BENCH_WARMUP"] ?? "1") ?? 1
    let runs = Int(ProcessInfo.processInfo.environment["HOMR_BENCH_RUNS"] ?? "3") ?? 3
    let full = ProcessInfo.processInfo.environment["HOMR_FULL"] == "1"

    guard let raw = loadGrayscale(path: imagePath) else {
        throw PipelineError.noNoteheads // reuse exit path; message overwritten below
    }
    let pre = preprocess(raw)
    log("Swift HomrValidate benchmark — \(imagePath)")
    log("  preprocessed: \(pre.width)x\(pre.height)  segnet CoreML=on  encoder CoreML=on")

    let segnetSession = try ONNXSessionFactory.makeSegnetSession()
    let segnet = SegnetInference(session: segnetSession)

    func runSegnetAndDetection() throws -> [MultiStaff] {
        let maps = try segnet.run(on: pre)
        func mask(_ pixels: [UInt8]) -> GrayscaleImage {
            GrayscaleImage(pixels: pixels, width: maps.width, height: maps.height)
        }
        var predictions = InputPredictions(
            original: pre,
            preprocessed: pre,
            notehead: mask(maps.noteheads),
            symbols: mask(maps.symbols),
            staff: mask(maps.staff),
            clefsKeys: mask(maps.clefsKeys),
            stemsRest: mask(maps.stemsRests)
        )
        predictions = filterPredictions(predictions)
        predictions = withStrongerStaffLines(predictions)
        return try detectStaffsInImage(predictions)
    }

    for i in 0..<warmup {
        _ = try runSegnetAndDetection()
        if full {
            let config = TransformerConfig()
            _ = parseStaffs(
                staffs: try runSegnetAndDetection(),
                image: pre,
                config: config
            )
        }
        log("  warmup \(i + 1)/\(warmup) done")
    }

    var segnetTimes: [Double] = []
    var detectionTimes: [Double] = []
    var transformerTimes: [Double] = []
    var totalTimes: [Double] = []

    for i in 0..<runs {
        let tTotal = benchNow()
        let tSeg = benchNow()
        let maps = try segnet.run(on: pre)
        let segnetS = benchNow() - tSeg

        let tDet = benchNow()
        func mask(_ pixels: [UInt8]) -> GrayscaleImage {
            GrayscaleImage(pixels: pixels, width: maps.width, height: maps.height)
        }
        var predictions = InputPredictions(
            original: pre,
            preprocessed: pre,
            notehead: mask(maps.noteheads),
            symbols: mask(maps.symbols),
            staff: mask(maps.staff),
            clefsKeys: mask(maps.clefsKeys),
            stemsRest: mask(maps.stemsRests)
        )
        predictions = filterPredictions(predictions)
        predictions = withStrongerStaffLines(predictions)
        let multiStaffs = try detectStaffsInImage(predictions)
        let detectionS = benchNow() - tDet

        var transformerS = 0.0
        if full {
            let tTr = benchNow()
            let config = TransformerConfig()
            let resultStaffs = parseStaffs(
                staffs: multiStaffs,
                image: predictions.preprocessed,
                config: config
            )
            transformerS = benchNow() - tTr
            if i == runs - 1 {
                let xmlDocument = generateXml(XmlGeneratorArguments(), staffs: resultStaffs, title: "")
                try xmlDocument.xmlString().write(toFile: outPath, atomically: true, encoding: .utf8)
            }
        }

        let totalS = benchNow() - tTotal
        segnetTimes.append(segnetS)
        detectionTimes.append(detectionS)
        transformerTimes.append(transformerS)
        totalTimes.append(totalS)

        var line = String(format: "  run %d: segnet=%.3fs  detection=%.3fs", i + 1, segnetS, detectionS)
        if full { line += String(format: "  transformer=%.3fs", transformerS) }
        line += String(format: "  total=%.3fs", totalS)
        log(line)
    }

    log("--- summary (median of timed runs) ---")
    log(String(format: "  segnet:      %.3fs", benchMedian(segnetTimes)))
    log(String(format: "  detection:   %.3fs", benchMedian(detectionTimes)))
    if full {
        log(String(format: "  transformer: %.3fs", benchMedian(transformerTimes)))
    }
    log(String(format: "  timed total: %.3fs", benchMedian(totalTimes)))
}

// MARK: - Run

func log(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    log("usage: HomrValidate <image-path> [out.musicxml]")
    exit(2)
}
let imagePath = arguments[1]
let outPath = arguments.count >= 3 ? arguments[2] : "/tmp/homr_validate/bach_swift.musicxml"

if ProcessInfo.processInfo.environment["HOMR_BENCHMARK"] == "1" {
    do {
        try runBenchmark(imagePath: imagePath, outPath: outPath)
    } catch {
        log("benchmark error: \(error)")
        exit(1)
    }
    exit(0)
}

do {
    guard let raw = loadGrayscale(path: imagePath) else {
        log("error: could not load image at \(imagePath)")
        exit(1)
    }
    log("loaded image: \(raw.width)x\(raw.height)")

    let pre = preprocess(raw)
    log("preprocessed: \(pre.width)x\(pre.height)")

    // --- segmentation ---
    let segnetSession = try ONNXSessionFactory.makeSegnetSession()
    let segnet = SegnetInference(session: segnetSession)
    let maps = try segnet.run(on: pre)
    log("segmentation:\n\(maps.summary)")

    // Multi-page crash repro: reuse the CACHED segnet session over additional,
    // differently-sized images (colon-separated paths in HOMR_REPEAT_IMAGES).
    // Before the fixed-batch-size fix this recompiled the CoreML MLProgram for
    // each new input shape and aborted on the 2nd page; it must now run clean.
    if let repeatList = ProcessInfo.processInfo.environment["HOMR_REPEAT_IMAGES"], !repeatList.isEmpty {
        for (i, extraPath) in repeatList.split(separator: ":").enumerated() {
            guard let extraRaw = loadGrayscale(path: String(extraPath)) else {
                log("repeat[\(i)]: could not load \(extraPath)"); continue
            }
            let extraPre = preprocess(extraRaw)
            let extraMaps = try segnet.run(on: extraPre)
            log("repeat[\(i)] \(extraPre.width)x\(extraPre.height) OK:\n\(extraMaps.summary)")
        }
        log("multi-page segnet reuse: completed without crash")
    }

    func mask(_ pixels: [UInt8]) -> GrayscaleImage {
        GrayscaleImage(pixels: pixels, width: maps.width, height: maps.height)
    }
    var predictions = InputPredictions(
        original: pre,
        preprocessed: pre,
        notehead: mask(maps.noteheads),
        symbols: mask(maps.symbols),
        staff: mask(maps.staff),
        clefsKeys: mask(maps.clefsKeys),
        stemsRest: mask(maps.stemsRests)
    )
    // Diagnostic: staff/clef mask distribution top→bottom BEFORE noise filtering.
    log("staff band counts (raw):   \(bandCounts(predictions.staff))")
    log("clef band counts (raw):    \(bandCounts(predictions.clefsKeys))")
    predictions = filterPredictions(predictions)
    log("staff band counts (filt):  \(bandCounts(predictions.staff))")
    log("clef band counts (filt):   \(bandCounts(predictions.clefsKeys))")
    predictions = withStrongerStaffLines(predictions)

    // --- staff detection ---
    let multiStaffs = try detectStaffsInImage(predictions)
    let allStaffs = multiStaffs.flatMap { $0.staffs }
    log("connected systems: \(multiStaffs.count) · physical staves: \(allStaffs.count)")
    for (i, ms) in multiStaffs.enumerated() {
        let ys = ms.staffs.map { "(\(Int($0.minY))–\(Int($0.maxY)))" }.joined(separator: " ")
        log("  system \(i + 1): \(ms.staffs.count) staves  y=\(ys)")
    }
    // Sorted staff y-centers, to spot vertical gaps where staves were dropped.
    let centers = allStaffs.map { Int(($0.minY + $0.maxY) / 2) }.sorted()
    log("staff y-centers: \(centers)  (image h=\(predictions.staff.height))")

    // Skip the (slow) transformer unless HOMR_FULL=1: detection benchmarking only
    // needs the staff counts above.
    guard ProcessInfo.processInfo.environment["HOMR_FULL"] == "1" else {
        log("skipping transformer (set HOMR_FULL=1 to run it)")
        exit(0)
    }

    // --- transformer recognition ---
    let config = TransformerConfig()
    let resultStaffs = parseStaffs(staffs: multiStaffs, image: predictions.preprocessed, config: config)

    // --- MusicXML ---
    let xmlDocument = generateXml(XmlGeneratorArguments(), staffs: resultStaffs, title: "")
    let xmlString = xmlDocument.xmlString()
    try xmlString.write(toFile: outPath, atomically: true, encoding: .utf8)

    let symbolCount = resultStaffs.reduce(0) { $0 + $1.count }
    log("voices: \(resultStaffs.count) · symbols: \(symbolCount)")
    log("wrote MusicXML → \(outPath)")
} catch {
    log("pipeline error: \(error)")
    exit(1)
}
