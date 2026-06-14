# Homr iOS

Native iOS port of [homr](https://github.com/liebharc/homr) — Optical Music Recognition (OMR) that converts photos of sheet music into MusicXML.

The original Python project lives in `../homr/`. This folder contains the iOS app.

## Status

| Stage | Python module | iOS status |
|-------|---------------|------------|
| Grayscale + autocrop + resize | `autocrop.py`, `resize.py` | ✅ Ported (`Pipeline/`) |
| CLAHE contrast | `color_adjust.py` | ✅ Ported — real tile-based CLAHE (`Pipeline/CLAHE.swift`) |
| Segmentation (UNet) | `segmentation/inference_segnet.py` | ✅ Ported + segmentation overlay (`Services/ONNX/SegnetInference.swift`) |
| Native CV layer (OpenCV subset) | `cv2.*` calls across homr | ✅ Ported in pure Swift (`CV/`), parity-tested vs cv2 4.13 |
| Bounding boxes + data model | `bounding_boxes.py`, `model.py`, `constants.py` | ✅ Ported (`Detection/`) |
| Symbol detection | `bar_line_detection.py`, `note_detection.py`, `brace_dot_detection.py` | ✅ Ported (`Detection/`) |
| Staff detection & merging | `staff_detection.py`, `find_peaks.py`, `noise_filtering.py` | ✅ Ported (`Detection/`) |
| Staff parsing + dewarping | `staff_parsing*.py`, `staff_dewarping.py`, `staff_regions.py` | ✅ Ported (`Detection/`) |
| Transformer encoder/decoder | `transformer/*_inference.py`, `staff2score.py` | ✅ Ported (`Transformer/`), KV-cache decode on CPU EP |
| MusicXML export | `music_xml_generator.py`, `circle_of_fifths.py` | ✅ Ported (`MusicXML/`) with a native Swift XML builder |
| Playback + MIDI export | _(new — no Python equivalent)_ | ✅ Native (`Audio/`): MusicXML→MIDI + in-app synth |

The full pipeline now runs end-to-end: pick a photo and the app preprocesses,
segments, detects staffs/symbols, runs the transformer, and produces a
MusicXML document that can be exported via the **Export MusicXML** share sheet.
The detection overlay and per-class pixel counts are still shown for inspection.

### Audio playback & MIDI (`Audio/`)

The recognised score can be played back and exported as MIDI:

- `MusicXMLMIDIConverter` walks the `XMLNode` tree (divisions, `<backup>`/
  `<forward>`, `<chord>`, multi-part) into a tempo-tagged `MIDISequence`.
- `StandardMIDIFile` serialises that sequence to a real type-0 `.mid`, offered
  via the **Export MIDI** share sheet.
- `ScorePlayer` plays the sequence through `AVAudioEngine` + a hand-written
  polyphonic `AVAudioSourceNode` synth (sine + harmonics, AD/R envelope), so it
  sounds with no bundled SoundFont. The UI adds play/pause/stop, a scrubber, and
  a tempo stepper.

v1 playback simplifications: repeats/voltas are not expanded (plays through
once), grace notes are skipped, and ties re-articulate rather than sustain.

### Instrument identification (`Audio/Instrument.swift`, `Services/InstrumentLabelReader.swift`)

homr separates parts only geometrically (one part per staff position) and never
reads which instrument a staff is. We add identification on top:

- `InstrumentLabelReader` OCR-s the printed labels in the **first system's** left
  margin (Vision `VNRecognizeTextRequest`). Because `MultiStaff.staffs` is sorted
  top-to-bottom and `parseStaffs` keys voices by staff position, the labels
  propagate to every later system by voice index.
- `InstrumentCatalog` matches the label text (EN/IT/DE/FR terms) to a General
  MIDI voice, which is written into the MusicXML `<part-list>` (`<part-name>`,
  `<instrument-sound>`, `<midi-program>`).
- That `<midi-program>` is the single source of truth: `MusicXMLMIDIConverter`
  reads it back into `MIDISequence.channelPrograms`, so the `.mid` export gets
  per-channel program-change events and the synth picks a matching `Timbre`
  (additive harmonics + envelope, with decay for plucked/struck voices).

Unidentified parts fall back to the generic Voice/Piano default. Note this is
label-OCR only — figured-bass continuo is voiced as its bass line, not realised.

The Android app [Andromr](https://github.com/aicelen/Andromr) wraps the same homr backend using Java inference + Python UI. This iOS project takes a fully native Swift approach with ONNX Runtime.

> **xcodegen note:** when you add new Swift files, re-run `xcodegen generate`
> so they are added to the Xcode target.

## Requirements

- Xcode 16+
- iOS 17+
- ~500 MB free storage for ONNX models (downloaded on first launch)

## Open the project

```bash
cd HomrIOS
xcodegen generate
open Homr.xcodeproj
```

Set your **Development Team** in the Homr target signing settings, then build and run on a device or simulator.

> **Note:** ONNX Runtime via SPM may require patching `MinimumOSVersion` in framework Info.plists before App Store submission. See [onnxruntime-swift-package-manager#16](https://github.com/microsoft/onnxruntime-swift-package-manager/issues/16).

## Architecture

```
Photo → ImagePreprocessor → SegnetInference (ONNX + CoreML EP)
      → InputPredictions (noise-filtered masks)
      → predict_symbols (noteheads, staff fragments, clefs, stems, bar lines)
      → detectStaff → addNotesToStaffs → findBracesBracketsAndGrandStaffLines
      → parseStaffs → Staff2Score (encoder + KV-cache decoder)
      → generateXml → MusicXmlDocument (.musicxml via ShareLink)
```

The orchestration lives in `Services/OMRProcessor.swift`, mirroring
`homr/main.py`'s `detect_staffs_in_image` → `parse_staffs` → `generate_xml`.

Models are downloaded from the same GitHub releases as the Python CLI:

`https://github.com/liebharc/homr/releases/download/onnx_checkpoints/`

## Native CV layer (`Homr/CV/`)

Rather than embed `opencv2.xcframework`, the OpenCV subset homr needs is
re-implemented in pure Swift under `Homr/CV/`, faithfully ported from the
OpenCV 4.13 C++ sources:

| File | OpenCV functions |
|------|------------------|
| `CVTypes.swift` | shared `CV` namespace + `Point`/`Rect`/`RotatedRect`/… |
| `BasicOps.swift` | `threshold`, `adaptiveThreshold` (Gaussian), `subtract`, `bitwise_and`, `calcHist` |
| `Morphology.swift` | `getStructuringElement`, `erode`, `dilate`, `morphologyEx` |
| `Contours.swift` | `findContours` (Suzuki–Abe, RETR_TREE/EXTERNAL, CHAIN_APPROX_SIMPLE) |
| `Geometry.swift` | `boundingRect`, `contourArea`, `convexHull`, `minAreaRect`, `boxPoints`, `pointPolygonTest`, `getAffineTransform`, `rotatedRectangleIntersection`, `fillConvexPoly` |
| `EllipseFit.swift` | `fitEllipse` (NoDirect path), `HoughLinesP` |

### Parity tests

`HomrTests/` is a standalone **logic-test** bundle (no app host) that compiles
the CV sources directly and validates them against golden fixtures captured
from real cv2 4.13 (`CVParityFixtures/`, regenerable via `generate_fixtures.py`).
All 12 tests pass; integer outputs (morphology, contours, thresholds, bounding
rects, intersection flags, convex hulls) match cv2 **exactly**, floats within
fixture tolerance.

```bash
xcodegen generate
# derived data MUST live outside the iCloud/file-provider-synced home dir,
# otherwise injected com.apple.FinderInfo xattrs break code-signing:
xcodebuild -project Homr.xcodeproj -scheme HomrTests \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -derivedDataPath /tmp/homr_dd test
```

Known scoped gaps: `fitEllipse` ports only the `fitEllipseNoDirect` path (cv2's
>5-point case, which is all homr ever hits — noteheads have many contour
points); the exact-5-point `fitEllipseDirect` path is not ported. `HoughLinesP`
matches the algorithm but not cv2 bit-for-bit (OpenCV visits edge pixels in a
random order), so it has no golden fixture.

## Known gaps / future work

The whole pipeline is ported and compiles, but a few homr features were
deliberately simplified or skipped on iOS:

1. **Staff dewarping** uses an identity `PiecewiseAffineTransform` fallback
   instead of full Delaunay-based piecewise warping (`staff_dewarping.py`).
2. **Title detection** (the RapidOCR step) is not ported; the MusicXML
   `<work-title>` is left empty.
3. **`HoughLinesP`** matches the algorithm but not cv2 bit-for-bit (see the CV
   parity notes above).

## Empirical validation

The full chain has been numerically diffed against the Python reference on a
real score (Bach, English Suite No. 1, Bourrée I, rendered at 300 DPI). A
headless macOS harness (`HomrCLI/`) compiles the exact same Foundation-only
pipeline sources used by the app (minus the UIKit/SwiftUI files) plus a small
`main.swift` that replicates `ImagePreprocessor` + `OMRProcessor`, so the port
runs on the Mac against the same ONNX models.

**Result: the Swift port produces musically identical output to Python homr.**
Both detect 6 connected staffs, ~340 notes, 26 measures, and the extracted
musical streams match exactly:

| Stream | Count | Match |
|--------|-------|-------|
| Pitch steps (A–G) | 288 | identical |
| Octaves | 288 | identical |
| Durations | 433 | identical |
| Note types | 298 | identical |
| Alters (♯/♭) | 107 | identical |

The only `.musicxml` diffs are cosmetic serialization (self-closing-tag
whitespace, element order within `<note>`, a `<!DOCTYPE>` line) plus the OCR
title, which we deliberately don't port.

### Two real bugs found and fixed during validation

Both bugs exist in (or are shared with) the Python reference and only surface on
dense scores with articulations/ornaments — simple test images never trigger
them:

1. **Decoder slur/articulation input swap** (`Transformer/ScoreDecoder.swift`).
   The ONNX export (`training/onnx/convert.py`) traces the decoder with
   positional args `(…, articulations, slurs, …)` but declares `input_names` as
   `(…, "slurs", "articulations", …)`, so the input *named* `articulations`
   actually drives `slur_emb` (size 5). Binding straight (as Python does)
   overflows the size-5 slur embedding the instant a real articulation (e.g. a
   trill) is predicted, crashing the run. We compensate by swapping the two
   feedback bindings. (The reference crashes on this exact file without the same
   fix.)
2. **Autoregressive decode memory blow-up** (same file). ONNX Runtime hands back
   *autoreleased* `ORTValue`/`NSData` from `session.run`/`tensorData()`. The
   decode loop has no enclosing autorelease pool, so over hundreds of steps these
   accumulate and balloon RSS past 4.5 GB → OOM kill. Wrapping each step in an
   `autoreleasepool` (plus `memory.enable_memory_arena_shrinkage`) keeps peak
   memory flat (~1.7 GB for the whole 6-staff page).

### Reproducing

```bash
# Render a PDF page to PNG (any image works too):
python3 -c "import fitz; d=fitz.open('score.pdf'); \
  d[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72), colorspace=fitz.csGRAY).save('score.png')"

# fp32 segnet/encoder/decoder must live in ~/Library/Application Support/Models/
cd HomrIOS/HomrCLI && swift build -c release
./.build/release/HomrValidate /path/to/score.png out.musicxml
```

## License

homr is AGPL-3.0. This iOS port inherits the same license obligations if distributed.
