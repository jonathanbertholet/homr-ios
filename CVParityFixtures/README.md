# CV Parity Fixtures

Golden-fixture dataset capturing **real OpenCV outputs** so the native Swift
re-implementation of the OpenCV subset can be validated for parity.

All fixtures are produced by [`generate_fixtures.py`](./generate_fixtures.py)
using `opencv-python-headless` + `numpy`. Inputs are fully deterministic
(hard-coded images and parameters; no randomness), so re-running the generator
reproduces byte-identical results on any machine running the same cv2 version.

- **cv2 version used to generate these fixtures:** `4.13.0`
- **Total size:** ~0.8 MB across the JSON files.
- JSON is written compact (no whitespace) and with sorted keys for stable diffs.

## Regenerating

```bash
python3 -m venv /tmp/cvfix
source /tmp/cvfix/bin/activate
pip install "opencv-python-headless>=4.11,<4.14" numpy
python generate_fixtures.py
```

The venv lives in `/tmp` and is intentionally **not** committed.

---

## Common conventions

Every fixture file includes a top-level `"cv2_version"` string.

### Image object

Binary/grayscale 2D `uint8` images are serialized as:

```json
{ "width": W, "height": H, "data": [ /* W*H ints, row-major (C order) */ ] }
```

`data[y*width + x]` is the pixel at column `x`, row `y`. Binary images use the
values `0` and `255`; the `gray_ramp` test image uses the full `0..255` range.

### Contour

A contour is a list of integer `[x, y]` points:

```json
[ [x0, y0], [x1, y1], ... ]
```

(OpenCV's native `Nx1x2` shape is flattened to `Nx2`.)

### Hierarchy

`findContours` hierarchy is a list of 4-int rows, one per contour, in OpenCV's
order `[next, previous, first_child, parent]`. `-1` means "none". Empty list
when no contours were found.

### RotatedRect

```json
{ "center": [cx, cy], "size": [w, h], "angle": degrees }
```

### BoundingRect

```json
{ "x": x, "y": y, "width": w, "height": h }
```

---

## Test images

Built in `make_images()`. All are binary `uint8` (0/255) except `gray_ramp`.

| name             | size (WxH) | description                                            |
|------------------|------------|--------------------------------------------------------|
| `rectangle`      | 80x60      | single filled axis-aligned rectangle                   |
| `rotated_rect`   | 80x80      | filled RotatedRect `((40,40),(40,20),30°)` via boxPoints |
| `circle`         | 60x60      | filled circle, center (30,30) r=18                     |
| `ellipse`        | 100x80     | filled ellipse, center (50,40) axes (30,18) angle 25°  |
| `two_blobs`      | 100x60     | two disjoint filled rectangles                         |
| `blob_with_hole` | 80x80      | solid block with a rectangular hole (nested contours)  |
| `hline`          | 60x40      | 1px-tall horizontal line                               |
| `vline`          | 40x60      | 1px-wide vertical line                                 |
| `gray_ramp`      | 40x40      | deterministic gradient `(x*6 + y*3) % 256`             |

---

## `manifest.json`

```json
{
  "cv2_version": "4.13.0",
  "files": ["contours.json", "geometry.json", "morphology.json", "basicops.json"],
  "file_sizes": [ { "file": "contours.json", "bytes": 12345 }, ... ]
}
```

---

## `contours.json`

`cv2.findContours(img, mode, CHAIN_APPROX_SIMPLE)` for modes `RETR_TREE` and
`RETR_EXTERNAL`, over the 8 binary test images.

```json
{
  "cv2_version": "4.13.0",
  "images":  { "<imageName>": <Image object>, ... },
  "results": {
    "<imageName>": {
      "RETR_TREE":     { "contours": [ <Contour>, ... ], "hierarchy": [ [n,p,c,par], ... ] },
      "RETR_EXTERNAL": { "contours": [ <Contour>, ... ], "hierarchy": [ ... ] }
    },
    ...
  }
}
```

> Note: `blob_with_hole` under `RETR_TREE` yields 2 contours (outer + inner
> hole) with hierarchy `[[-1,-1,1,-1],[-1,-1,-1,0]]`, while `RETR_EXTERNAL`
> yields only the outer contour.

---

## `geometry.json`

Per-shape geometry primitives.

```json
{
  "cv2_version": "4.13.0",

  "from_contours": {
    "<imageName>": [
      {
        "points":       <Contour>,                // the source contour (RETR_EXTERNAL)
        "boundingRect": <BoundingRect>,            // cv2.boundingRect
        "contourArea":  float,                     // cv2.contourArea
        "minAreaRect":  <RotatedRect>,             // cv2.minAreaRect
        "boxPoints":    [ [x,y] x4 (floats) ],     // cv2.boxPoints(minAreaRect)
        "convexHull":   <Contour>,                 // cv2.convexHull
        "fitEllipse":   { "center":[cx,cy], "axes":[w,h], "angle":deg }  // only when >=5 pts
      },
      ...
    ],
    ...
  },

  "explicit": {
    "<setName>": { /* same fields as above, on a hard-coded point set */ }
  },

  "affine": {
    "src":    [ [x,y] x3 ],     // source triangle
    "dst":    [ [x,y] x3 ],     // destination triangle
    "matrix": [ [a,b,c], [d,e,f] ]   // cv2.getAffineTransform -> 2x3
  },

  "rrect_intersection": [
    {
      "label":  "partial" | "disjoint" | "contained" | "rotated_overlap",
      "rect1":  <RotatedRect>,
      "rect2":  <RotatedRect>,
      "result": 0 | 1 | 2,           // 0=NONE, 1=PARTIAL, 2=FULL
      "region": [ [x,y], ... ]       // intersection polygon (may be empty)
    },
    ...
  ]
}
```

Explicit point sets: `square_quad`, `triangle`, `concave_poly`.

---

## `morphology.json`

```json
{
  "cv2_version": "4.13.0",

  "kernels": {
    "MORPH_RECT_3x3":    <Image object>,   // cv2.getStructuringElement, values 0/1
    "MORPH_RECT_5x5":    <Image object>,
    "MORPH_RECT_3x5":    <Image object>,
    "MORPH_ELLIPSE_3x3": <Image object>,
    "MORPH_ELLIPSE_5x5": <Image object>,
    "MORPH_ELLIPSE_3x5": <Image object>
  },

  "op_kernel": <Image object>,   // the 3x3 MORPH_RECT kernel used for the ops below

  "operations": {
    "<imageName>": {
      "erode_it1":  <Image>, "erode_it2":  <Image>,
      "dilate_it1": <Image>, "dilate_it2": <Image>,
      "open_it1":   <Image>, "open_it2":   <Image>,    // cv2.morphologyEx OPEN
      "close_it1":  <Image>, "close_it2":  <Image>     // cv2.morphologyEx CLOSE
    },
    ...
  }
}
```

Kernel masks store `0`/`1`. Operation outputs store `0`/`255`. Images covered:
`rectangle`, `circle`, `two_blobs`, `blob_with_hole`, `hline`, `vline`.

---

## `basicops.json`

```json
{
  "cv2_version": "4.13.0",

  "threshold": {
    "input": <Image>, "thresh": 128, "maxval": 255,
    "BINARY":     { "retval": float, "image": <Image> },   // cv2.THRESH_BINARY
    "BINARY_INV": { "image": <Image> }                     // cv2.THRESH_BINARY_INV
  },

  "adaptiveThreshold": {
    "input": <Image>,
    "results": [
      { "blockSize": 3, "C": 2, "method": "ADAPTIVE_THRESH_GAUSSIAN_C", "image": <Image> },
      { "blockSize": 5, "C": 1, "method": "ADAPTIVE_THRESH_GAUSSIAN_C", "image": <Image> }
    ]
  },

  "subtract":    { "a": <Image>, "b": <Image>, "result": <Image> },  // saturating cv2.subtract
  "bitwise_and": { "src": <Image>, "mask": <Image>, "result": <Image> },

  "calcHist": {
    "input": <Image>, "bins": 256, "range": [0, 256],
    "hist": [ /* 256 ints; sum == width*height */ ]
  }
}
```

---

## Writing Swift parity tests

1. Load each JSON file from the test bundle.
2. Rebuild input images from the `{width, height, data}` objects.
3. Run the native Swift function with the documented parameters.
4. Compare against the recorded output:
   - **Integer/image outputs** (morphology, threshold, bitwise, contours,
     boundingRect, histogram, intersection codes): assert **exact** equality.
   - **Floating-point outputs** (contourArea, minAreaRect, boxPoints,
     fitEllipse, affine matrix, intersection region): compare with a small
     tolerance (e.g. `abs(a-b) <= 1e-3`), and treat RotatedRect angle/size with
     OpenCV's known representation ambiguities in mind.
