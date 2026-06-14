import XCTest

/// Parity tests for `CV` BasicOps against cv2 4.13 golden output.
final class BasicOpsParityTests: XCTestCase {

    func testThreshold() {
        let t = Fix.dict(Fix.json("basicops")["threshold"])
        let input = Fix.image(t["input"])
        let thr = Fix.dbl(t["thresh"])
        let maxv = Fix.dbl(t["maxval"])

        let bin = CV.threshold(input, thresh: thr, maxValue: maxv, type: .binary)
        assertImageEqual(bin, Fix.data(Fix.dict(t["BINARY"])["image"]), "threshold BINARY")

        let inv = CV.threshold(input, thresh: thr, maxValue: maxv, type: .binaryInv)
        assertImageEqual(inv, Fix.data(Fix.dict(t["BINARY_INV"])["image"]), "threshold BINARY_INV")
    }

    func testAdaptiveThresholdGaussian() {
        let at = Fix.dict(Fix.json("basicops")["adaptiveThreshold"])
        let input = Fix.image(at["input"])
        for r in Fix.arr(at["results"]) {
            let rd = Fix.dict(r)
            let bs = Fix.int(rd["blockSize"])
            let c = Fix.dbl(rd["C"])
            // Generator used THRESH_BINARY (invert == false), maxValue 255.
            let out = CV.adaptiveThresholdGaussian(input, maxValue: 255, blockSize: bs, c: c, invert: false)
            assertImageEqual(out, Fix.data(rd["image"]), "adaptiveThreshold bs=\(bs) C=\(c)")
        }
    }

    func testSubtract() {
        let s = Fix.dict(Fix.json("basicops")["subtract"])
        let out = CV.subtract(Fix.image(s["a"]), Fix.image(s["b"]))
        assertImageEqual(out, Fix.data(s["result"]), "subtract")
    }

    func testBitwiseAnd() {
        let b = Fix.dict(Fix.json("basicops")["bitwise_and"])
        // Generator called cv2.bitwise_and(src, src, mask=mask).
        let src = Fix.image(b["src"])
        let out = CV.bitwiseAnd(src, src, mask: Fix.image(b["mask"]))
        assertImageEqual(out, Fix.data(b["result"]), "bitwise_and")
    }

    func testCalcHist() {
        let h = Fix.dict(Fix.json("basicops")["calcHist"])
        let hist = CV.calcHist(Fix.image(h["input"]))
        XCTAssertEqual(hist, Fix.ints(h["hist"]), "calcHist 256-bin")
    }
}
