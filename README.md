# BRobbDetector

Detects hand-drawn red annotation boxes (rotated rectangles / OBB) in X-ray security images using a classical-CV pipeline. Built with .NET 10 + Avalonia MVVM, with a Python prototype/benchmark harness.

## Algorithm (pm_v5_2)

A 6-stage classical-CV pipeline operating on the HSV red mask:

```
red_mask → denoise → outer_contours → base_quads(hull ∪ edgefit ∪ mar) → admission_filter → dedup
```

| Stage | Operation | Key params |
|-------|-----------|-----------|
| 1. `red_mask` | HSV threshold (S≥80, V≥110) → binary mask | smin=80, vmin=110 |
| 2. `denoise` | Drop connected components < min_area | denoise_min_area=40 |
| 3. `outer_contours` | findContours RETR_CCOMP + area filter | min_area=1500 |
| 4. `base_quads` | 3 methods in parallel per contour (see below) | — |
| 5. `_pass` | Side + orthogonality admission filter | ortho_tol=20, boundary_band=20, max_aspect=12 |
| 6. `dedup` | IoU-based priority dedup | dedup_iou=0.75 |

### Stage 4 — three complementary quad extractors

Each contour is reduced to a quad by three independent methods, then unioned with priority dedup (hull > edgefit > mar):

- **hull** — `convexHull` → `approxPolyDP`: repairs concave dents, gives clean 4-vertex quads on noisy outlines.
- **edgefit** — sector partition + TLS line fit → line intersection: refines corners where approxPolyDP drifts.
- **mar** — `minAreaRect`: recovers non-convex fragmented contours where approxPolyDP collapses to 1-2 vertices (broken/fragmented red strokes). Contributes +138 TP (27.5% of total).

## Benchmark (500 real images, 1121 GT boxes)

Point-match metric: a detection is TP only if all 4 corners, after optimal Hungarian 4-to-4 assignment, are each within D pixels of GT.

GT = `Label@CARS_202609_raw` — the hand-drawn annotation, whose geometry is the red
rectangle burned into the input image (`benchmark/gt_final.json`). The older
`Compare-V1@CARS_202609_annotated` baseline below was measured against a machine
artifact and is **not comparable**.

| D | TP | FN | recall | corrected_f1 |
|---|----|----|--------|--------------|
| 3 | 561 | 560 | 50.0% | 64.4 |
| 5 | 591 | 530 | 52.7% | 67.8 |
| 8 | 597 | 524 | 53.3% | 68.5 |
| 12 | 602 | 519 | 53.7% | 69.0 |

C# `cvhybrid` (hue<=5 red mask, edgefit family, min_area 900, dedup 0.30, ortho_tol 10):
`benchmark/pm_v6_csharp_rawgt.json`. Before the 2026-10-20 re-base the same detector
scored D=5 260 TP / 25.2 corr_f1 against the true GT.

### Historical baseline (invalid GT — Compare-V1, 746 boxes)

| D | TP | FN | rawFP | trueFP | recall | corrected_f1 |
|---|----|----|-------|--------|--------|--------------|
| 3 | 601 | 145 | 364 | 194 | 80.6% | 78.0% |
| 5 | 609 | 137 | 356 | 191 | 81.6% | 78.8% |
| 8 | 624 | 122 | 341 | 180 | 83.6% | 80.5% |
| 12 | 632 | 114 | 333 | 175 | 84.7% | 81.4% |

Frozen baseline: `benchmark/pm_v5_2_baseline.json`. C# port updated and Release build passes.

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

- .NET 10, Avalonia 11.2.8 (MVVM), OpenCvSharp5 5.0.0.20261018 (OpenCV 5 native)
- Python prototype + benchmark harness (`proto/`)
- RapidOCR sidecar for OCR (`proto/ocr_sidecar.py`)

## Project Structure

```
BRobbDetector/          C# project (main app)
  Services/             CvHybridDetector.cs (pm_v5_2 port), ColorBoxDetector, YoloObbDetector
  ViewModels/           MVVM view models
  Views/                Avalonia UI
proto/                  Python prototype + eval harness
  cv_hybrid.py          5 CV families + hybrids
  cv_scope4.py          _base_quads (hull/edgefit/mar)
  cv_scope7.py          pm5_params (pm_v5_1 config)
  eval_pointmatch.py    Point-match metric + GT-corrected eval
  eval_pm_benchmark.py  500-image real benchmark
  promote_pm_v5_1.py    Baseline promotion (pm_v5_1)
  promote_pm_v5_2.py    Baseline promotion (pm_v5_2, LOOP-31 optimized)
benchmark/             Frozen baselines + per-LOOP results
data/                  Test images and labels
```