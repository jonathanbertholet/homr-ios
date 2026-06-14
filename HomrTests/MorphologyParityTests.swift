import XCTest

/// Parity tests for `CV` morphology (structuring elements + erode/dilate/
/// open/close) against cv2 4.13 golden output.
final class MorphologyParityTests: XCTestCase {

    private let shapes: [(String, CV.MorphShape)] = [
        ("MORPH_RECT", .rect),
        ("MORPH_ELLIPSE", .ellipse),
    ]
    // (width, height) — matches the generator's KERNEL_SIZES order.
    private let sizes: [(Int, Int)] = [(3, 3), (5, 5), (3, 5)]

    func testStructuringElements() {
        let kernels = Fix.dict(Fix.json("morphology")["kernels"])
        for (shapeName, shape) in shapes {
            for (kw, kh) in sizes {
                let key = "\(shapeName)_\(kw)x\(kh)"
                let el = CV.getStructuringElement(shape, (width: kw, height: kh))
                let expected = Fix.dict(kernels[key])
                XCTAssertEqual(el.width, Fix.int(expected["width"]), "\(key): width")
                XCTAssertEqual(el.height, Fix.int(expected["height"]), "\(key): height")
                let producedMask = el.mask.map { $0 ? 1 : 0 }
                XCTAssertEqual(producedMask, Fix.ints(expected["data"]), "\(key): mask")
            }
        }
    }

    func testErodeDilateOpenClose() {
        let morph = Fix.json("morphology")
        let ops = Fix.dict(morph["operations"])
        // Inputs aren't duplicated in morphology.json; they live in contours.json.
        let images = Fix.dict(Fix.json("contours")["images"])
        let opKernel = CV.getStructuringElement(.rect, (width: 3, height: 3))

        for name in ["rectangle", "circle", "two_blobs", "blob_with_hole", "hline", "vline"] {
            let input = Fix.image(images[name])
            let per = Fix.dict(ops[name])
            for it in [1, 2] {
                assertImageEqual(CV.erode(input, opKernel, iterations: it),
                                 Fix.data(per["erode_it\(it)"]), "\(name) erode it\(it)")
                assertImageEqual(CV.dilate(input, opKernel, iterations: it),
                                 Fix.data(per["dilate_it\(it)"]), "\(name) dilate it\(it)")
                assertImageEqual(CV.morphologyEx(input, .open, opKernel, iterations: it),
                                 Fix.data(per["open_it\(it)"]), "\(name) open it\(it)")
                assertImageEqual(CV.morphologyEx(input, .close, opKernel, iterations: it),
                                 Fix.data(per["close_it\(it)"]), "\(name) close it\(it)")
            }
        }
    }
}
