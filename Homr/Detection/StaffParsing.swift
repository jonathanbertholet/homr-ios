import Foundation

// MARK: - StaffParsing
//
// Port of `homr/staff_parsing.py` (all `Debug` parameters and `debug.write_*` /
// `cv2.circle` / `cv2.putText` blocks dropped).
//
// This is the top of the staff → symbols stage. For each detected staff it:
//   1. carves out a region around the staff (clamped so it doesn't overlap
//      neighbouring staffs — see `StaffRegions`),
//   2. resizes / crops / (identity-)dewarps it into the transformer's canvas,
//   3. runs the transformer (`parseStaffTromr`),
// and finally de-duplicates symbols per voice. The geometry math (canvas sizing,
// scaling, region offsets, rounding) is reproduced exactly to match the Python
// pipeline; only the dewarp step is an identity fallback (see `StaffDewarping`).

// MARK: - Voice / system bookkeeping

/// Port of `_have_all_the_same_number_of_staffs`: do all voices contain the same
/// number of staffs as the first one?
private func haveAllTheSameNumberOfStaffs(_ staffs: [MultiStaff]) -> Bool {
    for staff in staffs where staff.staffs.count != staffs[0].staffs.count {
        return false
    }
    return true
}

/// Port of `_is_close_to_image_top_or_bottom`.
///
/// Bug-for-bug: the Python compares each staff's `min_x` and `image.shape[0]`
/// (the image HEIGHT) minus `max_x` — i.e. it mixes the x extent with the image
/// height. Reproduced exactly.
private func isCloseToImageTopOrBottom(_ staff: MultiStaff, _ image: GrayscaleImage) -> Bool {
    let tolerance = 50.0
    let closestDistanceToTopOrBottom = staff.staffs.map { min($0.minX, Double(image.height) - $0.maxX) }
    return closestDistanceToTopOrBottom.min()! < tolerance
}

/// Port of `_ensure_same_number_of_staffs`.
///
/// If the voices don't all have the same number of staffs, try dropping the
/// first or last system (when it touches an image edge and the remainder is
/// consistent); otherwise break every multi-staff into single staffs and sort
/// them top-to-bottom by `staffs[0].min_y`.
private func ensureSameNumberOfStaffs(_ staffs: [MultiStaff], image: GrayscaleImage) -> [MultiStaff] {
    if haveAllTheSameNumberOfStaffs(staffs) {
        return staffs
    }
    if staffs.count > 2 {
        if isCloseToImageTopOrBottom(staffs[0], image)
            && haveAllTheSameNumberOfStaffs(Array(staffs[1...])) {
            print("Removing first system from all voices, as it has a different number of staffs")
            return Array(staffs[1...])
        }
        if isCloseToImageTopOrBottom(staffs[staffs.count - 1], image)
            && haveAllTheSameNumberOfStaffs(Array(staffs[..<(staffs.count - 1)])) {
            print("Removing last system from all voices, as it has a different number of staffs")
            return Array(staffs[..<(staffs.count - 1)])
        }
    }
    var result: [MultiStaff] = []
    for staff in staffs {
        result.append(contentsOf: staff.breakApart())
    }
    // sorted(result, key=lambda s: s.staffs[0].min_y). Python's sort is stable;
    // decorate with the original index to break ties the same way.
    return result.enumerated()
        .sorted { a, b in
            let ya = a.element.staffs[0].minY
            let yb = b.element.staffs[0].minY
            if ya != yb { return ya < yb }
            return a.offset < b.offset
        }
        .map { $0.element }
}

/// Port of `_get_number_of_voices`: the staff count of the first voice.
private func getNumberOfVoices(_ staffs: [MultiStaff]) -> Int {
    staffs[0].staffs.count
}

// MARK: - Canvas sizing

/// Port of `get_tr_omr_canvas_size`.
///
/// Returns the size the staff image should be resized to so it fits exactly into
/// the transformer canvas (`maxHeight` x `maxWidth`, minus the top/bottom
/// margins) while preserving aspect ratio. Element 0 is the WIDTH, element 1 the
/// HEIGHT of the resized image (matching the Python `np.array` indexing).
///
/// `imageShape` is `(height, width)` to mirror numpy's `image.shape[:2]`.
func getTrOmrCanvasSize(
    imageShape: (height: Int, width: Int),
    marginTop: Int = 0,
    marginBottom: Int = 0,
    config: TransformerConfig
) -> (width: Int, height: Int) {
    let trOmrMaxHeight = config.maxHeight
    let trOmrMaxWidth = config.maxWidth
    let trOmrMaxHeightWithMargin = trOmrMaxHeight - marginTop - marginBottom
    let trOmrRatio = Double(trOmrMaxHeightWithMargin) / Double(trOmrMaxWidth)
    let height = imageShape.height
    let width = imageShape.width

    // Defensive guard (Python would divide by zero on a degenerate shape); a
    // zero-sized input cannot be meaningfully fitted, so fall back to the canvas.
    guard height > 0, width > 0 else {
        return (width: trOmrMaxWidth, height: trOmrMaxHeightWithMargin)
    }

    if Double(height) / Double(width) > trOmrRatio {
        // The height is the limiting factor.
        let newWidth = homrInt(Double(width) / Double(height) * Double(trOmrMaxHeightWithMargin))
        return (width: newWidth, height: trOmrMaxHeightWithMargin)
    } else {
        // The width is the limiting factor.
        let newHeight = homrInt(Double(height) / Double(width) * Double(trOmrMaxWidth))
        return (width: trOmrMaxWidth, height: newHeight)
    }
}

/// Port of `center_image_on_canvas` (grayscale path only — our images are 1-channel).
///
/// Builds a white (255) canvas of size `maxHeight` x `maxWidth`, resizes the
/// input to `canvasSize` (a `(width, height)` from `getTrOmrCanvasSize`), and
/// pastes it at `x_offset = 0`, `y_offset = (maxHeightWithMargin - resizedH) // 2 + marginTop`.
func centerImageOnCanvas(
    _ image: GrayscaleImage,
    canvasSize: (width: Int, height: Int),
    marginTop: Int = 0,
    marginBottom: Int = 0,
    config: TransformerConfig
) -> GrayscaleImage {
    let trOmrMaxHeight = config.maxHeight
    let trOmrMaxWidth = config.maxWidth

    // cv2.resize(image, canvas_size) → dsize is (width, height).
    let resized = image.resized(toWidth: canvasSize.width, height: canvasSize.height)

    var newPixels = [UInt8](repeating: 255, count: trOmrMaxWidth * trOmrMaxHeight)
    let xOffset = 0
    let trOmrMaxHeightWithMargin = trOmrMaxHeight - marginTop - marginBottom
    let yOffset = homrFloorDiv(trOmrMaxHeightWithMargin - resized.height, 2) + marginTop

    // Paste the resized image. Python assumes it fits exactly; we clip
    // defensively to stay memory-safe if a rounding edge case overflows.
    let rh = resized.height
    let rw = resized.width
    resized.pixels.withUnsafeBufferPointer { src in
        newPixels.withUnsafeMutableBufferPointer { dst in
            for row in 0..<rh {
                let dy = yOffset + row
                if dy < 0 || dy >= trOmrMaxHeight { continue }
                let dstRow = dy * trOmrMaxWidth
                let srcRow = row * rw
                for col in 0..<rw {
                    let dx = xOffset + col
                    if dx < 0 || dx >= trOmrMaxWidth { continue }
                    dst[dstRow + dx] = src[srcRow + col]
                }
            }
        }
    }

    return GrayscaleImage(pixels: newPixels, width: trOmrMaxWidth, height: trOmrMaxHeight)
}

/// Port of `add_image_into_tr_omr_canvas`: size then center an image onto the
/// transformer canvas.
func addImageIntoTrOmrCanvas(_ image: GrayscaleImage, config: TransformerConfig) -> GrayscaleImage {
    let newShape = getTrOmrCanvasSize(
        imageShape: (height: image.height, width: image.width), config: config
    )
    return centerImageOnCanvas(image, canvasSize: newShape, config: config)
}

// MARK: - Edge cleanup

/// Port of `remove_black_contours_at_edges_of_image`.
///
/// Thresholds the gray image at 97, inverts it, finds contours, and for every
/// large contour that touches an image edge but is NOT mostly dark, paints that
/// region white in the gray image (removing scanner shadows along page edges).
/// Returns the modified gray image.
func removeBlackContoursAtEdgesOfImage(_ gray: GrayscaleImage, unitSize: Double) -> GrayscaleImage {
    // cv2.threshold(gray, 97, 255, THRESH_BINARY); thresh = 255 - thresh.
    let binary = CV.threshold(gray, thresh: 97, maxValue: 255, type: .binary)
    var invertedPixels = binary.pixels
    for i in invertedPixels.indices {
        invertedPixels[i] = 255 - invertedPixels[i]
    }
    let thresh = GrayscaleImage(pixels: invertedPixels, width: binary.width, height: binary.height)

    let contourResult = CV.findContours(thresh, mode: .tree)
    let threshold = Constants.blackSpotRemovalThreshold(unitSize)

    let w = gray.width
    let h = gray.height
    var grayPixels = gray.pixels

    for cnt in contourResult.contours {
        let rect = CV.boundingRect(cnt)
        let x = rect.x
        let y = rect.y
        let rw = rect.width
        let rh = rect.height

        // Skip small contours.
        if Double(rw) < threshold || Double(rh) < threshold {
            continue
        }
        // Only act on contours that touch an image edge.
        let isAtEdgeOfImage = x == 0 || y == 0 || x + rw == w || y + rh == h
        if !isAtEdgeOfImage {
            continue
        }
        // np.mean(thresh[y:y+h, x:x+w]) < 127 → mostly dark, leave it alone.
        var sum = 0.0
        var count = 0
        for row in y..<(y + rh) {
            let base = row * w
            for col in x..<(x + rw) {
                sum += Double(thresh.pixels[base + col])
                count += 1
            }
        }
        let averageGrayIntensity = 127.0
        let isMostlyDark = (count > 0 ? sum / Double(count) : 0) < averageGrayIntensity
        if isMostlyDark {
            continue
        }
        // gray[y:y+h, x:x+w] = 255.
        for row in y..<(y + rh) {
            let base = row * w
            for col in x..<(x + rw) {
                grayPixels[base + col] = 255
            }
        }
    }

    return GrayscaleImage(pixels: grayPixels, width: w, height: h)
}

// MARK: - Region / staff preparation

/// Port of `_calculate_region`.
///
/// Region around a staff: ±2 unit sizes horizontally and ±4 vertically, with the
/// vertical extent clamped to its nearest neighbouring staffs. Each value is
/// truncated with `int(...)` (toward zero); kept as `Double` since it is later
/// scaled and rounded again. Returns `[xMin, yMin, xMax, yMax]`.
private func calculateRegion(staff: Staff, regions: StaffRegions) -> [Double] {
    let avg = staff.averageUnitSize
    let xMin = staff.minX - 2 * avg
    let xMax = staff.maxX + 2 * avg
    let yMin = max(staff.minY - 4 * avg, regions.getStartOfClosestStaffAbove(staff.minY))
    let yMax = min(staff.maxY + 4 * avg, regions.getStartOfClosestStaffBelow(staff.maxY))
    return [Double(homrInt(xMin)), Double(homrInt(yMin)), Double(homrInt(xMax)), Double(homrInt(yMax))]
}

/// Port of `_dewarp_staff`.
///
/// Applies the same transformation to the staff's coordinates as was applied to
/// the image: shift by the crop's top-left, optionally dewarp the point (identity
/// here), then scale. Returns a transformed copy of the staff.
private func dewarpStaff(
    staff: Staff, dewarp: StaffDewarping?, region: (Double, Double), scaling: Double
) -> Staff {
    let transform: (CV.PointF) -> CV.PointF = { point in
        var x = point.x - region.0
        var y = point.y - region.1
        if let dewarp = dewarp {
            let p = dewarp.dewarpPoint(CV.PointF(x, y))
            x = p.x
            y = p.y
        }
        x *= scaling
        y *= scaling
        return CV.PointF(x, y)
    }
    return staff.transformCoordinates(transform)
}

/// Port of `prepare_staff_image` (debug blocks dropped).
///
/// Crops/resizes/(identity-)dewarps the full page image down to the transformer
/// canvas for a single staff, and returns the canvas image together with the
/// staff whose coordinates have been transformed into that canvas space.
///
/// Because the dewarp is an identity fallback (see `StaffDewarping`),
/// `dewarp.dewarp(staffImage)` returns the image unchanged — but every crop /
/// scale / region-offset step is reproduced exactly.
func prepareStaffImage(
    index: Int, staff: Staff, staffImage: GrayscaleImage, regions: StaffRegions, config: TransformerConfig
) -> (GrayscaleImage, Staff) {
    var staff = staff
    var staffImage = staffImage

    // Original (un-scaled) region as truncated ints, used for canvas sizing.
    let region = calculateRegion(staff: staff, regions: regions)
    let imageDimensions = getTrOmrCanvasSize(
        imageShape: (height: homrInt(region[3] - region[1]), width: homrInt(region[2] - region[0])),
        config: config
    )
    let scalingFactor = Double(imageDimensions.height) / (region[3] - region[1])

    // Scale the whole page image by the same factor.
    staffImage = staffImage.resized(
        toWidth: homrInt(Double(staffImage.width) * scalingFactor),
        height: homrInt(Double(staffImage.height) * scalingFactor)
    )

    // region = np.round(region * scaling_factor).
    let scaledRegion = region.map { homrRoundToDouble($0 * scalingFactor) }

    // Step 1 crop: region grown by [-10, -50, +10, +50].
    let regionStep1 = [
        scaledRegion[0] - 10, scaledRegion[1] - 50, scaledRegion[2] + 10, scaledRegion[3] + 50,
    ]
    let crop1 = cropImageAndReturnNewTop(
        staffImage, x1: regionStep1[0], y1: regionStep1[1], x2: regionStep1[2], y2: regionStep1[3]
    )
    staffImage = crop1.0
    let topLeft = crop1.1

    // Step 2 region is expressed relative to the step-1 crop's top-left.
    let regionStep2 = [
        scaledRegion[0] - Double(topLeft.x), scaledRegion[1] - Double(topLeft.y),
        scaledRegion[2] - Double(topLeft.x), scaledRegion[3] - Double(topLeft.y),
    ]

    // top_left / scaling_factor → original-image-space offset for the staff coords.
    let topLeftScaled = (Double(topLeft.x) / scalingFactor, Double(topLeft.y) / scalingFactor)
    staff = dewarpStaff(staff: staff, dewarp: nil, region: topLeftScaled, scaling: scalingFactor)

    // Build (identity) dewarp, apply to the image, then crop to step-2 region.
    let dewarp = dewarpStaffImage(image: staffImage, staff: staff, index: index)
    staffImage = dewarp.dewarp(staffImage)
    let crop2 = cropImageAndReturnNewTop(
        staffImage, x1: regionStep2[0], y1: regionStep2[1], x2: regionStep2[2], y2: regionStep2[3]
    )
    staffImage = crop2.0
    // scaling_factor = 1 from here on (it is not used again).

    staffImage = removeBlackContoursAtEdgesOfImage(staffImage, unitSize: staff.averageUnitSize)
    staffImage = centerImageOnCanvas(staffImage, canvasSize: imageDimensions, config: config)

    return (staffImage, staff)
}

/// Port of `parse_staff_image` (debug blocks dropped).
///
/// Prepares the staff image and runs the transformer over it.
func parseStaffImage(
    index: Int, staff: Staff, image: GrayscaleImage, regions: StaffRegions, config: TransformerConfig
) -> [EncodedSymbol] {
    let (staffImage, transformedStaff) = prepareStaffImage(
        index: index, staff: staff, staffImage: image, regions: regions, config: config
    )
    return parseStaffTromr(staff: transformedStaff, staffImage: staffImage, config: config)
}

// MARK: - Top-level entry point

/// Port of `parse_staffs` (the `Debug` parameter dropped).
///
/// Dewarps each staff and runs it through the transformer to extract rhythm and
/// pitch information. Voices are processed independently; staffs within a voice
/// are separated by a `newline` symbol, and each voice is de-duplicated before
/// being returned. `selectedStaff >= 0` restricts processing to that single
/// staff index (used for debugging / partial runs).
func parseStaffs(
    staffs: [MultiStaff], image: GrayscaleImage, config: TransformerConfig, selectedStaff: Int = -1
) -> [[EncodedSymbol]] {
    let staffs = ensureSameNumberOfStaffs(staffs, image: image)
    // For simplicity every staff in a multi staff is called a voice, even if it
    // is part of a grand staff.
    let numberOfVoices = getNumberOfVoices(staffs)
    var i = 0
    var voices: [[EncodedSymbol]] = []
    let regions = StaffRegions(staffs)

    for voice in 0..<numberOfVoices {
        let staffsForVoice = staffs.map { $0.staffs[voice] }
        var resultForVoice: [EncodedSymbol] = []
        for (staffIndex, staff) in staffsForVoice.enumerated() {
            if selectedStaff >= 0 && staffIndex != selectedStaff {
                print("Ignoring staff due to selectedStaff argument", i)
                i += 1
                continue
            }
            var resultStaff = parseStaffImage(
                index: i, staff: staff, image: image, regions: regions, config: config
            )
            if resultStaff.isEmpty {
                print("Skipping empty staff", i)
                i += 1
                continue
            }
            resultStaff.append(EncodedSymbol("newline"))
            resultForVoice.append(contentsOf: resultStaff)
            i += 1
        }
        voices.append(remove_duplicated_symbols(resultForVoice))
    }
    return voices
}
