#!/usr/bin/env python3
# Golden-fixture generator for OpenCV -> native Swift parity testing.
# Produces deterministic input -> output JSON fixtures for the subset of
# OpenCV functions being re-implemented in Swift. No randomness is used:
# every input image and parameter set is hard-coded so the outputs are
# reproducible across runs and machines.

import os
import json
import cv2
import numpy as np

# Directory where all fixtures + README live (this script's own folder).
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenCV version is embedded into every fixture file + the manifest so the
# Swift side can assert which reference implementation it is matching.
CV2_VERSION = cv2.__version__


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def img_to_dict(img):
    """Serialize a 2D uint8 image as {width, height, data:[row-major ints]}."""
    # Force a contiguous 2D uint8 array, then flatten in row-major (C) order.
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    return {"width": int(w), "height": int(h), "data": arr.reshape(-1).astype(int).tolist()}


def contour_to_list(contour):
    """Serialize an OpenCV contour (Nx1x2 int32) to a list of [x, y] pairs."""
    # findContours returns shape (N,1,2); reshape to (N,2) for a clean list.
    pts = contour.reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in pts]


def hierarchy_to_list(hierarchy):
    """Serialize the findContours hierarchy (1xNx4 int32) to list of 4-int rows.

    Each row is [next, previous, first_child, parent] in OpenCV's convention.
    Returns an empty list when no contours were found.
    """
    if hierarchy is None:
        return []
    h = np.asarray(hierarchy).reshape(-1, 4)
    return [[int(v) for v in row] for row in h]


def rotated_rect_to_dict(rect):
    """Serialize a cv2 RotatedRect ((cx,cy),(w,h),angle) into a dict."""
    (cx, cy), (rw, rh), angle = rect
    return {
        "center": [float(cx), float(cy)],
        "size": [float(rw), float(rh)],
        "angle": float(angle),
    }


def points_to_list(pts):
    """Serialize an array of float/int points to list of [x, y] (floats)."""
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    return [[float(p[0]), float(p[1])] for p in arr]


def write_json(name, payload):
    """Write a fixture file, stamping the cv2 version into every file.

    Uses compact separators (no indentation) because the image payloads are
    large flat pixel arrays; indentation would multiply the file size several
    fold. sort_keys keeps the byte output stable run-to-run for diffing.
    """
    payload = {"cv2_version": CV2_VERSION, **payload}
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Deterministic test images (all binary uint8: values are exactly 0 or 255)
# ---------------------------------------------------------------------------

def make_images():
    """Build the full dictionary of named binary test images."""
    images = {}

    # 1) Single axis-aligned rectangle on a 60x80 canvas.
    rect = np.zeros((60, 80), np.uint8)
    cv2.rectangle(rect, (10, 12), (55, 40), 255, thickness=-1)  # filled
    images["rectangle"] = rect

    # 2) Rotated rectangle: build a known RotatedRect, fill its boxPoints.
    rot = np.zeros((80, 80), np.uint8)
    rot_rect = ((40.0, 40.0), (40.0, 20.0), 30.0)  # center, size, angle(deg)
    box = cv2.boxPoints(rot_rect)                   # 4 corner points (float)
    cv2.fillConvexPoly(rot, np.intp(np.round(box)), 255)
    images["rotated_rect"] = rot

    # 3) Filled circle with known center/radius.
    circle = np.zeros((60, 60), np.uint8)
    cv2.circle(circle, (30, 30), 18, 255, thickness=-1)
    images["circle"] = circle

    # 4) Filled ellipse with known axes/angle.
    ellipse = np.zeros((80, 100), np.uint8)
    cv2.ellipse(ellipse, (50, 40), (30, 18), 25.0, 0, 360, 255, thickness=-1)
    images["ellipse"] = ellipse

    # 5) Two separate blobs (disjoint rectangles) to test multi-contour output.
    two = np.zeros((60, 100), np.uint8)
    cv2.rectangle(two, (8, 10), (30, 45), 255, -1)    # left blob
    cv2.rectangle(two, (60, 12), (90, 50), 255, -1)   # right blob
    images["two_blobs"] = two

    # 6) Blob with a hole -> nested contours, exercises the hierarchy tree.
    hole = np.zeros((80, 80), np.uint8)
    cv2.rectangle(hole, (12, 12), (68, 68), 255, -1)  # solid outer block
    cv2.rectangle(hole, (30, 30), (50, 50), 0, -1)    # cut interior hole
    images["blob_with_hole"] = hole

    # 7) Thin horizontal line (1px tall).
    hline = np.zeros((40, 60), np.uint8)
    hline[20:21, 5:55] = 255
    images["hline"] = hline

    # 8) Thin vertical line (1px wide).
    vline = np.zeros((60, 40), np.uint8)
    vline[5:55, 20:21] = 255
    images["vline"] = vline

    # A small gradient/“gray” image used by threshold + histogram tests so the
    # ops have non-trivial (non-binary) input to operate on deterministically.
    gray = np.zeros((40, 40), np.uint8)
    for y in range(40):
        for x in range(40):
            # Smooth deterministic ramp in [0,255].
            gray[y, x] = (x * 6 + y * 3) % 256
    images["gray_ramp"] = gray

    return images


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

# Images that are meaningful inputs to contour/shape analysis.
CONTOUR_IMAGE_NAMES = [
    "rectangle", "rotated_rect", "circle", "ellipse",
    "two_blobs", "blob_with_hole", "hline", "vline",
]

# findContours retrieval modes we record.
RETRIEVAL_MODES = {
    "RETR_TREE": cv2.RETR_TREE,
    "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
}


def build_contours(images):
    """findContours outputs (contours + hierarchy) for each image and mode."""
    out = {}
    for name in CONTOUR_IMAGE_NAMES:
        img = images[name]
        per_mode = {}
        for mode_name, mode in RETRIEVAL_MODES.items():
            # CHAIN_APPROX_SIMPLE collapses straight runs to endpoints.
            contours, hierarchy = cv2.findContours(
                img.copy(), mode, cv2.CHAIN_APPROX_SIMPLE
            )
            per_mode[mode_name] = {
                "contours": [contour_to_list(c) for c in contours],
                "hierarchy": hierarchy_to_list(hierarchy),
            }
        out[name] = per_mode
    return {"images": {n: img_to_dict(images[n]) for n in CONTOUR_IMAGE_NAMES},
            "results": out}


def build_geometry(images):
    """Per-contour geometry + explicit known point sets.

    Covers boundingRect, contourArea, minAreaRect, boxPoints, fitEllipse,
    convexHull, getAffineTransform, and rotatedRectangleIntersection.
    """
    out = {"from_contours": {}, "explicit": {}, "affine": {}, "rrect_intersection": []}

    # --- Geometry derived from contours of each image (RETR_EXTERNAL) -------
    for name in CONTOUR_IMAGE_NAMES:
        img = images[name]
        contours, _ = cv2.findContours(
            img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        entries = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            entry = {
                "points": contour_to_list(c),
                "boundingRect": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "contourArea": float(cv2.contourArea(c)),
                "minAreaRect": rotated_rect_to_dict(cv2.minAreaRect(c)),
                "boxPoints": points_to_list(cv2.boxPoints(cv2.minAreaRect(c))),
                "convexHull": contour_to_list(cv2.convexHull(c)),
            }
            # fitEllipse requires at least 5 points.
            if len(c) >= 5:
                (ecx, ecy), (eaw, eah), eang = cv2.fitEllipse(c)
                entry["fitEllipse"] = {
                    "center": [float(ecx), float(ecy)],
                    "axes": [float(eaw), float(eah)],
                    "angle": float(eang),
                }
            entries.append(entry)
        out["from_contours"][name] = entries

    # --- Explicit, hard-coded point sets (independent of contour extraction) -
    explicit_sets = {
        # A simple axis-aligned quad.
        "square_quad": [[10, 10], [40, 10], [40, 30], [10, 30]],
        # A triangle.
        "triangle": [[5, 5], [50, 8], [25, 45]],
        # A convex pentagon-ish set with one interior (concave) point to test hull.
        "concave_poly": [[0, 0], [40, 0], [40, 40], [20, 15], [0, 40]],
    }
    for set_name, pts in explicit_sets.items():
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(arr)
        entry = {
            "points": [[int(p[0]), int(p[1])] for p in pts],
            "boundingRect": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "contourArea": float(cv2.contourArea(arr)),
            "minAreaRect": rotated_rect_to_dict(cv2.minAreaRect(arr)),
            "boxPoints": points_to_list(cv2.boxPoints(cv2.minAreaRect(arr))),
            "convexHull": [[int(p[0]), int(p[1])] for p in cv2.convexHull(arr).reshape(-1, 2)],
        }
        if len(pts) >= 5:
            (ecx, ecy), (eaw, eah), eang = cv2.fitEllipse(arr)
            entry["fitEllipse"] = {
                "center": [float(ecx), float(ecy)],
                "axes": [float(eaw), float(eah)],
                "angle": float(eang),
            }
        out["explicit"][set_name] = entry

    # --- getAffineTransform for a known src/dst triangle --------------------
    src_tri = np.float32([[0, 0], [10, 0], [0, 10]])
    dst_tri = np.float32([[2, 3], [12, 5], [1, 14]])
    affine = cv2.getAffineTransform(src_tri, dst_tri)  # 2x3 matrix
    out["affine"] = {
        "src": points_to_list(src_tri),
        "dst": points_to_list(dst_tri),
        "matrix": [[float(affine[r][c]) for c in range(3)] for r in range(2)],
    }

    # --- rotatedRectangleIntersection for overlapping/disjoint/contained ----
    rrect_pairs = [
        # Partial overlap.
        ("partial", ((20.0, 20.0), (20.0, 20.0), 0.0), ((30.0, 30.0), (20.0, 20.0), 0.0)),
        # Disjoint (far apart).
        ("disjoint", ((10.0, 10.0), (8.0, 8.0), 0.0), ((50.0, 50.0), (8.0, 8.0), 0.0)),
        # Fully contained (small rect inside big rect, same center).
        ("contained", ((30.0, 30.0), (40.0, 40.0), 0.0), ((30.0, 30.0), (10.0, 10.0), 0.0)),
        # Overlap with rotation.
        ("rotated_overlap", ((30.0, 30.0), (24.0, 24.0), 0.0), ((30.0, 30.0), (24.0, 24.0), 45.0)),
    ]
    for label, r1, r2 in rrect_pairs:
        # Returns (retval, intersectionRegion). retval: 0 NONE / 1 PARTIAL / 2 FULL.
        retval, region = cv2.rotatedRectangleIntersection(r1, r2)
        out["rrect_intersection"].append({
            "label": label,
            "rect1": rotated_rect_to_dict(r1),
            "rect2": rotated_rect_to_dict(r2),
            "result": int(retval),
            "region": points_to_list(region) if region is not None else [],
        })

    return out


# Structuring-element shapes + sizes used for morphology.
KERNEL_SHAPES = {
    "MORPH_RECT": cv2.MORPH_RECT,
    "MORPH_ELLIPSE": cv2.MORPH_ELLIPSE,
}
KERNEL_SIZES = [(3, 3), (5, 5), (3, 5)]
MORPH_IMAGE_NAMES = ["rectangle", "circle", "two_blobs", "blob_with_hole", "hline", "vline"]


def build_morphology(images):
    """getStructuringElement kernels + erode/dilate/open/close outputs."""
    out = {"kernels": {}, "operations": {}}

    # --- Record the structuring-element masks as 0/1 grids ------------------
    for shape_name, shape in KERNEL_SHAPES.items():
        for (kw, kh) in KERNEL_SIZES:
            k = cv2.getStructuringElement(shape, (kw, kh))
            key = f"{shape_name}_{kw}x{kh}"
            # Kernel values are 0/1 already; store as the row-major image dict.
            out["kernels"][key] = img_to_dict(k)

    # --- erode / dilate / morphologyEx(OPEN/CLOSE) on each image ------------
    # Use one representative 3x3 rectangular kernel for the op outputs.
    op_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for name in MORPH_IMAGE_NAMES:
        img = images[name]
        per_img = {}
        for iters in (1, 2):
            per_img[f"erode_it{iters}"] = img_to_dict(
                cv2.erode(img, op_kernel, iterations=iters))
            per_img[f"dilate_it{iters}"] = img_to_dict(
                cv2.dilate(img, op_kernel, iterations=iters))
            per_img[f"open_it{iters}"] = img_to_dict(
                cv2.morphologyEx(img, cv2.MORPH_OPEN, op_kernel, iterations=iters))
            per_img[f"close_it{iters}"] = img_to_dict(
                cv2.morphologyEx(img, cv2.MORPH_CLOSE, op_kernel, iterations=iters))
        out["operations"][name] = per_img

    # Record which kernel was used for the op outputs so Swift can match it.
    out["op_kernel"] = img_to_dict(op_kernel)
    return out


def build_basicops(images):
    """threshold, adaptiveThreshold, subtract, bitwise_and, calcHist."""
    out = {}
    gray = images["gray_ramp"]

    # --- threshold BINARY / BINARY_INV at a fixed thresh --------------------
    thresh_val, dst_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    _, dst_bininv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    out["threshold"] = {
        "input": img_to_dict(gray),
        "thresh": 128,
        "maxval": 255,
        "BINARY": {"retval": float(thresh_val), "image": img_to_dict(dst_bin)},
        "BINARY_INV": {"image": img_to_dict(dst_bininv)},
    }

    # --- adaptiveThreshold (Gaussian) at a couple blockSize/C combos --------
    adaptive = []
    for block_size, c in [(3, 2), (5, 1)]:
        a = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        adaptive.append({
            "blockSize": block_size,
            "C": c,
            "method": "ADAPTIVE_THRESH_GAUSSIAN_C",
            "image": img_to_dict(a),
        })
    out["adaptiveThreshold"] = {"input": img_to_dict(gray), "results": adaptive}

    # --- subtract: gray - rectangle-shaped second operand -------------------
    # Build a deterministic second operand of equal size (constant block).
    second = np.zeros_like(gray)
    second[5:25, 5:25] = 50  # constant region to subtract
    sub = cv2.subtract(gray, second)  # saturating subtraction
    out["subtract"] = {
        "a": img_to_dict(gray),
        "b": img_to_dict(second),
        "result": img_to_dict(sub),
    }

    # --- bitwise_and of a binary image with a mask --------------------------
    base = images["rectangle"]
    mask = np.zeros_like(base)
    cv2.circle(mask, (40, 26), 18, 255, -1)  # circular mask overlapping the rect
    band = cv2.bitwise_and(base, base, mask=mask)
    out["bitwise_and"] = {
        "src": img_to_dict(base),
        "mask": img_to_dict(mask),
        "result": img_to_dict(band),
    }

    # --- calcHist with 256 bins on the gray ramp ----------------------------
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    out["calcHist"] = {
        "input": img_to_dict(gray),
        "bins": 256,
        "range": [0, 256],
        "hist": [int(round(float(v[0]))) for v in hist],
    }

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"OpenCV (cv2) version: {CV2_VERSION}")
    images = make_images()

    # Build each fixture category.
    fixtures = {
        "contours.json": build_contours(images),
        "geometry.json": build_geometry(images),
        "morphology.json": build_morphology(images),
        "basicops.json": build_basicops(images),
    }

    written = []
    for fname, payload in fixtures.items():
        path = write_json(fname, payload)
        size = os.path.getsize(path)
        written.append({"file": fname, "bytes": size})
        print(f"  wrote {fname} ({size} bytes)")

    # Top-level manifest enumerating files + version.
    manifest = {
        "cv2_version": CV2_VERSION,
        "files": list(fixtures.keys()),
        "file_sizes": written,
    }
    mpath = os.path.join(OUT_DIR, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  wrote manifest.json ({os.path.getsize(mpath)} bytes)")


if __name__ == "__main__":
    main()
