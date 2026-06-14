import XCTest

/// Parity tests for `CV.findContours` (Suzuki–Abe) against cv2 4.13.
/// Contours and hierarchy are compared exactly (point order, start vertex,
/// sibling/child links must all match OpenCV).
final class ContoursParityTests: XCTestCase {

    private let modes: [(String, CV.RetrievalMode)] = [
        ("RETR_TREE", .tree),
        ("RETR_EXTERNAL", .externalOnly),
    ]

    func testFindContours() {
        let c = Fix.json("contours")
        let images = Fix.dict(c["images"])
        let results = Fix.dict(c["results"])

        for (name, perModeAny) in results {
            let input = Fix.image(images[name])
            let perMode = Fix.dict(perModeAny)
            for (modeName, mode) in modes {
                let expected = Fix.dict(perMode[modeName])
                let produced = CV.findContours(input, mode: mode)

                // Contours: count, then point-by-point equality per contour.
                let expContours = Fix.arr(expected["contours"]).map { Fix.points($0) }
                XCTAssertEqual(produced.contours.count, expContours.count,
                               "\(name)/\(modeName): contour count")
                if produced.contours.count == expContours.count {
                    for i in 0..<expContours.count {
                        XCTAssertEqual(produced.contours[i], expContours[i],
                                       "\(name)/\(modeName): contour[\(i)] points")
                    }
                }

                // Hierarchy: [next, previous, first_child, parent] per node.
                let expHier = Fix.arr(expected["hierarchy"]).map { Fix.ints($0) }
                XCTAssertEqual(produced.hierarchy.count, expHier.count,
                               "\(name)/\(modeName): hierarchy count")
                if produced.hierarchy.count == expHier.count {
                    for i in 0..<expHier.count {
                        let n = produced.hierarchy[i]
                        XCTAssertEqual([n.next, n.previous, n.firstChild, n.parent], expHier[i],
                                       "\(name)/\(modeName): hierarchy[\(i)]")
                    }
                }
            }
        }
    }
}
