import XCTest

/// Parity tests for `CV` geometry primitives against cv2 4.13.
///
/// Integer outputs (boundingRect, convexHull, intersection flags) are compared
/// exactly; floating outputs (contourArea, minAreaRect, boxPoints, affine,
/// fitEllipse) use tolerances that absorb float32 round-off.
final class GeometryParityTests: XCTestCase {

    /// Validates every field of one geometry entry (shared by from_contours and
    /// explicit point sets).
    private func checkEntry(_ e: [String: Any], _ label: String) {
        let pts = Fix.points(e["points"])

        // boundingRect — exact integers.
        let br = CV.boundingRect(pts)
        let ebr = Fix.dict(e["boundingRect"])
        XCTAssertEqual(br.x, Fix.int(ebr["x"]), "\(label) boundingRect.x")
        XCTAssertEqual(br.y, Fix.int(ebr["y"]), "\(label) boundingRect.y")
        XCTAssertEqual(br.width, Fix.int(ebr["width"]), "\(label) boundingRect.width")
        XCTAssertEqual(br.height, Fix.int(ebr["height"]), "\(label) boundingRect.height")

        // contourArea — float.
        XCTAssertEqual(CV.contourArea(pts), Fix.dbl(e["contourArea"]),
                       accuracy: 1e-3, "\(label) contourArea")

        // convexHull — exact (Sklansky + cv2 cyclic-shift / winding).
        XCTAssertEqual(CV.convexHull(pts), Fix.points(e["convexHull"]), "\(label) convexHull")

        // minAreaRect — compare via box corners (order/angle-ambiguity proof),
        // plus center and area from the recorded RotatedRect.
        let mar = CV.minAreaRect(pts)
        assertCornersMatch(CV.boxPoints(mar), Fix.pointsF(e["boxPoints"]), "\(label) boxPoints", tol: 0.05)
        let emar = Fix.rrect(e["minAreaRect"])
        XCTAssertEqual(mar.center.x, emar.center.x, accuracy: 0.05, "\(label) minAreaRect.center.x")
        XCTAssertEqual(mar.center.y, emar.center.y, accuracy: 0.05, "\(label) minAreaRect.center.y")
        let producedArea = mar.size.width * mar.size.height
        let expectedArea = emar.size.width * emar.size.height
        XCTAssertEqual(producedArea, expectedArea,
                       accuracy: max(1e-2, expectedArea * 1e-3), "\(label) minAreaRect.area")

        // fitEllipse — only present for >=5 points. cv2 switches to
        // fitEllipseDirect at exactly 5 points (a path homr never hits, since
        // noteheads have many contour points), so we only assert the
        // fitEllipseNoDirect path (count > 5) that the port implements.
        if let feAny = e["fitEllipse"], pts.count > 5 {
            let fe = Fix.dict(feAny)
            let ec = Fix.arr(fe["center"])
            let ea = Fix.arr(fe["axes"])
            let res = CV.fitEllipse(pts)
            XCTAssertEqual(res.center.x, Fix.dbl(ec[0]), accuracy: 0.1, "\(label) fitEllipse.center.x")
            XCTAssertEqual(res.center.y, Fix.dbl(ec[1]), accuracy: 0.1, "\(label) fitEllipse.center.y")
            let producedAxes = [res.size.width, res.size.height].sorted()
            let expectedAxes = [Fix.dbl(ea[0]), Fix.dbl(ea[1])].sorted()
            XCTAssertEqual(producedAxes[0], expectedAxes[0], accuracy: 0.2, "\(label) fitEllipse.minorAxis")
            XCTAssertEqual(producedAxes[1], expectedAxes[1], accuracy: 0.2, "\(label) fitEllipse.majorAxis")
        }
    }

    func testFromContours() {
        let g = Fix.json("geometry")
        let fromContours = Fix.dict(g["from_contours"])
        for (name, entriesAny) in fromContours {
            for (i, entry) in Fix.arr(entriesAny).enumerated() {
                checkEntry(Fix.dict(entry), "from_contours/\(name)[\(i)]")
            }
        }
    }

    func testExplicitPointSets() {
        let g = Fix.json("geometry")
        let explicit = Fix.dict(g["explicit"])
        for (name, entryAny) in explicit {
            checkEntry(Fix.dict(entryAny), "explicit/\(name)")
        }
    }

    func testGetAffineTransform() {
        let af = Fix.dict(Fix.json("geometry")["affine"])
        let src = Fix.pointsF(af["src"])
        let dst = Fix.pointsF(af["dst"])
        let m = CV.getAffineTransform(src: src, dst: dst)  // [m00,m01,m02,m10,m11,m12]
        let mat = Fix.arr(af["matrix"])
        let row0 = Fix.arr(mat[0])
        let row1 = Fix.arr(mat[1])
        let expected = [Fix.dbl(row0[0]), Fix.dbl(row0[1]), Fix.dbl(row0[2]),
                        Fix.dbl(row1[0]), Fix.dbl(row1[1]), Fix.dbl(row1[2])]
        for i in 0..<6 {
            XCTAssertEqual(m[i], expected[i], accuracy: 1e-6, "affine matrix[\(i)]")
        }
    }

    func testRotatedRectanglesIntersect() {
        let pairs = Fix.arr(Fix.json("geometry")["rrect_intersection"])
        for item in pairs {
            let d = Fix.dict(item)
            let label = d["label"] as? String ?? "?"
            let r1 = Fix.rrect(d["rect1"])
            let r2 = Fix.rrect(d["rect2"])
            let expected = Fix.int(d["result"]) != 0   // 0 == INTERSECT_NONE
            XCTAssertEqual(CV.rotatedRectanglesIntersect(r1, r2), expected,
                           "rrect_intersection/\(label)")
        }
    }
}
