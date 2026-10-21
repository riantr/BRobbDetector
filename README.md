# BRobbDetector

Detects hand-drawn red annotation boxes (rotated rectangles / OBB) in X-ray security images using a classical-CV pipeline. Built with .NET 10 + Avalonia MVVM, with a Python prototype/benchmark harness.

## Algorithm (cvhybrid, pm_v7)

A 6-stage classical-CV pipeline operating on the HSV red mask:

```
red_mask → denoise → outer_contours → edgefit_quads → admission_filter → dedup
```

| Stage | Operation | Key params |
|-------|-----------|-----------|
| 1. `red_mask` | HSV threshold (S≥100, V≥140, H≤4) → binary mask | smin=100, vmin=140, hmax=4 |
| 2. `denoise` | Drop connected components < min_area | denoise_min_area=40 |
| 3. `outer_contours` | findContours RETR_CCOMP + area filter | min_area=300 |
| 4. `edgefit` | sector partition + TLS line fit → line intersection | fit_band=8, fit_min_pts=12, fit_jump=20 |
| 5. `_pass` | Side + orthogonality admission filter | ortho_tol=10, boundary_band=6, min_side=12, max_aspect=12 |
| 6. `dedup` | IoU-based keep-order dedup | dedup_iou=0.30 |

### Stage 1 — the hue window is the critical parameter

The detector's input image has the human annotation burned in as red rectangles.
The annotation strokes are pure red (hue 0-4 for ~85% of their pixels), while the
orange/red X-ray object texture sits at hue 9-10. A wide hue window floods the mask
with object pixels — this was the single largest error source. Against the true GT,
hue≤10 scores D=5 335 TP / 42.6 corr_f1 and hue≤5 scores 592 / 67.9. Narrowing to
hue≤4 ties at D=5 but gives the best D=3 score (555 / 63.7 vs 547 / 62.8); combined
with the other re-tuned axes it reaches D=5 601 / 68.7.

### Stage 4 — edgefit only

`edgefit` (sector partition + total-least-squares line fit → adjacent-side line
intersection) is the only quad family emitted. The keep-order IoU dedup let the
cruder `hull` / `minAreaRect` quads displace the accurate edgefit ones: emitting the
union scored D=5 553 TP / 57.4 corr_f1 against 593 / 66.0 for edgefit alone.

## Benchmark (500 real images, 1121 GT boxes)

Point-match metric: a detection is TP only if all 4 corners, after optimal Hungarian 4-to-4 assignment, are each within D pixels of GT.

GT = `Label@CARS_202609_raw` — the hand-drawn annotation, whose geometry is the red
rectangle burned into the input image (`benchmark/gt_final.json`). The older
`Compare-V1@CARS_202609_annotated` baseline below was measured against a machine
artifact and is **not comparable**.

| D | TP | FN | recall | corrected_f1 |
|---|----|----|--------|--------------|
| 3 | 579 | 542 | 51.7% | 66.2 |
| 5 | 602 | 519 | 53.7% | 68.8 |
| 8 | 605 | 516 | 54.0% | 69.1 |
| 12 | 607 | 514 | 54.1% | 69.4 |

C# `cvhybrid` (hue≤4 red mask, edgefit family, min_area 300, min_side 12,
sat≥100, val≥140, dedup 0.30, ortho_tol 10, fit_band 8):
`benchmark/pm_v7_csharp_rawgt.json`. Before the 2026-10-20 re-base the same detector
scored D=5 260 TP / 25.2 corr_f1 against the true GT.

### Remaining failure modes (520 FN at D=5, `proto/scope41_small_targets.py`)

Measured on the Python prototype with the same production config (D=5 601 TP /
520 FN); the C# port scores 602 TP / 519 FN because it rounds corner coordinates
to the nearest pixel while Python truncates.

| reason | count | meaning |
|--------|-------|---------|
| `merged` | 257 | the GT box is swallowed by a larger red component (text inside the box merges with its outline) |
| `missed` | 253 | nothing close was detected at all |
| `imprecise` | 9 | a detection exists but its corners are 5-30 px away |
| `no_red` | 1 | the GT box is not drawn in the input image at all |

Corner precision is no longer the bottleneck — only 9 of 520 misses are imprecise
matches. The remaining errors are upstream, in the mask / connected-component stage.

### Prototype / production parity

`proto/cv_hybrid.py::PROD_DEFAULT` mirrors the constants in
`BRobbDetector/Services/CvHybridDetector.cs` one-for-one. Change one, change the
other.

### Historical baseline (invalid GT — Compare-V1, 746 boxes)

| D | TP | FN | rawFP | trueFP | recall | corrected_f1 |
|---|----|----|-------|--------|--------|--------------|
| 3 | 601 | 145 | 364 | 194 | 80.6% | 78.0 |
| 5 | 609 | 137 | 356 | 191 | 81.6% | 78.8 |
| 8 | 624 | 122 | 341 | 180 | 83.6% | 80.5 |
| 12 | 632 | 114 | 333 | 175 | 84.7% | 81.4 |

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
  Services/             CvHybridDetector.cs (cvhybrid / pm_v7), ColorBoxDetector, YoloObbDetector
  ViewModels/           MVVM view models
  Views/                Avalonia UI
proto/                  Python prototype + eval harness
  common.py             IMG / LAB (GT path) / red_mask / iou_obb
  cv_hybrid.py          5 CV families + hybrids
  eval_pointmatch.py    Point-match metric + GT-corrected eval
  eval_pm_benchmark.py  500-image real benchmark
  eval_det_dir.py       Evaluate any detection dir against the true GT
  gt_final.py           Ground-truth arbitration evidence bundle
  scope40_retune.py     One-axis-at-a-time re-tune on the TRUE GT
  scope41_small_targets.py  Small-target / FN-reason analysis
  scope42_combo.py      Combination check of the marginal winners
benchmark/             Frozen baselines + per-LOOP results
data/                  Test images and labels
```