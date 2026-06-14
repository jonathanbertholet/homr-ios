import Foundation
import XCTest
// CV sources + GrayscaleImage are compiled directly into this test target
// (see project.yml), so no module import is needed.

// Anchor class so `Bundle(for:)` resolves the unit-test bundle that the
// CVParityFixtures JSON files are copied into.
final class FixtureBundleToken {}

/// Tiny JSON-fixture reader built on `JSONSerialization`. The fixtures are
/// heterogeneous (images, contours, rotated rects, matrices) so a dynamic
/// reader is simpler and less brittle than bespoke Codable types.
enum Fix {
    /// Loads and parses one `<name>.json` fixture from the test bundle.
    static func json(_ name: String) -> [String: Any] {
        let bundle = Bundle(for: FixtureBundleToken.self)
        guard let url = bundle.url(forResource: name, withExtension: "json") else {
            fatalError("fixture \(name).json not found in test bundle")
        }
        let data = try! Data(contentsOf: url)
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    // MARK: scalar / container coercion (JSONSerialization yields NSNumber)
    static func int(_ v: Any?) -> Int { (v as! NSNumber).intValue }
    static func dbl(_ v: Any?) -> Double { (v as! NSNumber).doubleValue }
    static func arr(_ v: Any?) -> [Any] { v as! [Any] }
    static func dict(_ v: Any?) -> [String: Any] { v as! [String: Any] }
    static func ints(_ v: Any?) -> [Int] { arr(v).map { ($0 as! NSNumber).intValue } }

    // MARK: domain objects
    /// `{width,height,data}` → `GrayscaleImage`.
    static func image(_ v: Any?) -> GrayscaleImage {
        let d = dict(v)
        return GrayscaleImage(pixels: ints(d["data"]).map { UInt8($0) },
                              width: int(d["width"]), height: int(d["height"]))
    }
    /// Raw row-major pixel data of an image object.
    static func data(_ v: Any?) -> [Int] { ints(dict(v)["data"]) }

    static func points(_ v: Any?) -> [CV.Point] {
        arr(v).map { p in let xy = arr(p); return CV.Point(int(xy[0]), int(xy[1])) }
    }
    static func pointsF(_ v: Any?) -> [CV.PointF] {
        arr(v).map { p in let xy = arr(p); return CV.PointF(dbl(xy[0]), dbl(xy[1])) }
    }
    static func rrect(_ v: Any?) -> CV.RotatedRect {
        let d = dict(v); let c = arr(d["center"]); let s = arr(d["size"])
        return CV.RotatedRect(center: CV.PointF(dbl(c[0]), dbl(c[1])),
                              size: CV.Size(width: dbl(s[0]), height: dbl(s[1])),
                              angle: dbl(d["angle"]))
    }
}

// MARK: - Shared assertions

/// Asserts a produced image matches the recorded cv2 pixels exactly, reporting
/// the mismatch count and first differing index on failure.
func assertImageEqual(_ produced: GrayscaleImage, _ expected: [Int], _ label: String,
                      file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(produced.pixels.count, expected.count, "\(label): pixel-count", file: file, line: line)
    guard produced.pixels.count == expected.count else { return }
    var mismatches = 0
    var firstIdx = -1
    for i in 0..<expected.count where Int(produced.pixels[i]) != expected[i] {
        mismatches += 1
        if firstIdx < 0 { firstIdx = i }
    }
    XCTAssertEqual(mismatches, 0, "\(label): \(mismatches) px differ (first idx \(firstIdx))",
                   file: file, line: line)
}

/// Asserts two sets of 4 box corners match as an unordered set within `tol`
/// (OpenCV and the port can emit the same rectangle starting at a different
/// corner / winding, so order is not significant).
func assertCornersMatch(_ produced: [CV.PointF], _ expected: [CV.PointF], _ label: String,
                        tol: Double = 0.05, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(produced.count, expected.count, "\(label): corner count", file: file, line: line)
    guard produced.count == expected.count else { return }
    var remaining = produced
    for e in expected {
        if let idx = remaining.firstIndex(where: { abs($0.x - e.x) <= tol && abs($0.y - e.y) <= tol }) {
            remaining.remove(at: idx)
        } else {
            XCTFail("\(label): no produced corner near (\(e.x), \(e.y))", file: file, line: line)
            return
        }
    }
}
