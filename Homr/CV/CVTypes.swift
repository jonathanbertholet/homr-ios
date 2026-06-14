import Foundation

/// Shared value types for the native OpenCV re-implementation.
///
/// Everything lives under the `CV` namespace. Each algorithm group is
/// implemented in its own file via `extension CV { ... }`, so the modules can be
/// developed independently without redeclaration conflicts.
///
/// Conventions follow OpenCV 4.x so results match the Python `homr` pipeline
/// (which pins `opencv-python-headless` 4.13). The golden-fixture test harness
/// validates parity against the real library.
enum CV {

    /// Integer pixel coordinate (OpenCV `Point`).
    struct Point: Equatable, Hashable {
        var x: Int
        var y: Int
        init(_ x: Int, _ y: Int) { self.x = x; self.y = y }
    }

    /// Sub-pixel coordinate (OpenCV `Point2f`).
    struct PointF: Equatable {
        var x: Double
        var y: Double
        init(_ x: Double, _ y: Double) { self.x = x; self.y = y }
    }

    /// Size in floating point (OpenCV `Size2f`).
    struct Size: Equatable {
        var width: Double
        var height: Double
        init(width: Double, height: Double) { self.width = width; self.height = height }
    }

    /// Axis-aligned integer rectangle (OpenCV `Rect`): origin + size.
    struct Rect: Equatable {
        var x: Int
        var y: Int
        var width: Int
        var height: Int
        init(x: Int, y: Int, width: Int, height: Int) {
            self.x = x; self.y = y; self.width = width; self.height = height
        }
    }

    /// Rotated rectangle (OpenCV `RotatedRect`), also used for fitted ellipses.
    ///
    /// `angle` is in degrees, following OpenCV 4.x semantics: the rotation of
    /// the rectangle around `center`. For `minAreaRect` the real 4.13 library
    /// returns the angle in the range [-90, 0) (its own `CV_DbgCheck` asserts
    /// this), and the native port matches that, not [0, 90).
    /// For `fitEllipse`, `size` holds the full axis lengths (width = minor,
    /// height = major as returned by OpenCV) and `angle` the major-axis rotation.
    struct RotatedRect: Equatable {
        var center: PointF
        var size: Size
        var angle: Double
        init(center: PointF, size: Size, angle: Double) {
            self.center = center; self.size = size; self.angle = angle
        }
    }

    /// One node of the OpenCV contour hierarchy (the `[next, prev, child, parent]`
    /// quadruple from `findContours`). `-1` means "none".
    struct HierarchyNode: Equatable {
        var next: Int
        var previous: Int
        var firstChild: Int
        var parent: Int
        init(next: Int = -1, previous: Int = -1, firstChild: Int = -1, parent: Int = -1) {
            self.next = next; self.previous = previous
            self.firstChild = firstChild; self.parent = parent
        }
    }

    /// Result of `findContours`: parallel arrays of contours and hierarchy nodes.
    struct ContourResult: Equatable {
        var contours: [[Point]]
        var hierarchy: [HierarchyNode]
        init(contours: [[Point]], hierarchy: [HierarchyNode]) {
            self.contours = contours; self.hierarchy = hierarchy
        }
    }

    /// Kernel shape for `getStructuringElement`.
    enum MorphShape { case rect, ellipse, cross }

    /// Morphological operation for `morphologyEx`.
    enum MorphOp { case erode, dilate, open, close }

    /// Contour retrieval mode (subset of OpenCV's modes used by homr).
    enum RetrievalMode { case externalOnly, tree }

    /// A structuring element / kernel for morphology.
    ///
    /// `mask[row * width + col]` is `true` where the kernel is active. `anchorX`
    /// / `anchorY` default to the centre (OpenCV's `(-1, -1)` anchor).
    struct StructuringElement {
        var width: Int
        var height: Int
        var mask: [Bool]
        var anchorX: Int
        var anchorY: Int
        init(width: Int, height: Int, mask: [Bool], anchorX: Int, anchorY: Int) {
            self.width = width; self.height = height
            self.mask = mask; self.anchorX = anchorX; self.anchorY = anchorY
        }
    }
}
