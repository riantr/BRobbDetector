# BRobbDetector

Detects hand-drawn red annotation boxes (rotated rectangles / OBB) in X-ray security images using a classical-CV pipeline. Built with .NET 10 + Avalonia MVVM, with a Python prototype/benchmark harness.

## Algorithm (pm_v5_1)

A 6-stage classical-CV pipeline operating on the HSV red mask:

```
red_mask → denoise → outer_contours → base_quads(hull ∪ edgefit ∪ mar) → admission_filter → dedup
```

| Stage | Operation | Key params |
|-------|-----------|-----------|
| 1. `red_mask` | HSV threshold (S≥80, V≥110) → binary mask | smin=80, vmin=110 |
| 2. `denoise` | Drop connected components < min_area | denoise_min_area=40 |
| 3. `outer_contours` | findContours RETR_CCOMP + area filter | min_area=1000 |
| 4. `base_quads` | 3 methods in parallel per contour (see below) | — |
| 5. `_pass` | Side + orthogonality admission filter | ortho_tol=15, boundary_band=12, max_aspect=12 |
| 6. `dedup` | IoU-based priority dedup | dedup_iou=0.30 |

### Stage 4 — three complementary quad extractors

Each contour is reduced to a quad by three independent methods, then unioned with priority dedup (hull > edgefit > mar):

- **hull** — `convexHull` → `approxPolyDP`: repairs concave dents, gives clean 4-vertex quads on noisy outlines.
- **edgefit** — sector partition + TLS line fit → line intersection: refines corners where approxPolyDP drifts.
- **mar** — `minAreaRect`: recovers non-convex fragmented contours where approxPolyDP collapses to 1-2 vertices (broken/fragmented red strokes). Contributes +138 TP (27.5% of total).

## Benchmark (500 real images, 746 GT boxes)

Point-match metric: a detection is TP only if all 4 corners, after optimal Hungarian 4-to-4 assignment, are each within D pixels of GT.

| D | TP | FN | rawFP | trueFP | recall | corrected_f1 |
|---|----|----|-------|--------|--------|--------------|
| 3 | 490 | 256 | 305 | 162 | 65.7% | 70.1% |
| 5 | 502 | 244 | 293 | 157 | 67.3% | 71.5% |
| 8 | 517 | 229 | 278 | 146 | 69.3% | 73.4% |
| 12 | 524 | 222 | 271 | 141 | 70.2% | 74.3% |

Frozen baseline: `benchmark/pm_v5_1_baseline.json`. C# port parity verified (804 boxes, byte-exact vs Python).

## Build & Run

```bash
# Build (Release)
dotnet build BRobbDetector/BRobbDetector.csproj -c Release

# Run GUI
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Release

# CLI detection
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Release -- --cli <inputDir> <outputDir> [--ocr]

# Checker
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Release -- --check <imageDir> <labelDir>
```

## Tech Stack

- .NET 10, Avalonia 11.2.8 (MVVM), OpenCvSharp4 4.9
- Python prototype + benchmark harness (`proto/`)
- RapidOCR sidecar for OCR (`proto/ocr_sidecar.py`)

## Project Structure

```
BRobbDetector/          C# project (main app)
  Services/             CvHybridDetector.cs (pm_v5_1 port), ColorBoxDetector, YoloObbDetector
  ViewModels/           MVVM view models
  Views/                Avalonia UI
proto/                  Python prototype + eval harness
  cv_hybrid.py          5 CV families + hybrids
  cv_scope4.py          _base_quads (hull/edgefit/mar)
  cv_scope7.py          pm5_params (pm_v5_1 config)
  eval_pointmatch.py    Point-match metric + GT-corrected eval
  eval_pm_benchmark.py  500-image real benchmark
  promote_pm_v5_1.py    Baseline promotion script
benchmark/             Frozen baselines + per-LOOP results
data/                  Test images and labels
```