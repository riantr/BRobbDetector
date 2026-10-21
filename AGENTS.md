# BRobbDetector

## Build & Run

```bash
# Build (Debug)
dotnet build BRobbDetector/BRobbDetector.csproj -c Debug

# Build (Release)
dotnet build BRobbDetector/BRobbDetector.csproj -c Release

# Run GUI
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug

# CLI detection
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug -- --cli <inputDir> <outputDir> [--ocr]

# Checker
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug -- --check <imageDir> <labelDir>
```

## Tech Stack
- .NET 10, Avalonia 11.2.8 (MVVM), OpenCvSharp5 5.0.0.20261018 (OpenCV 5 native)
- Python RapidOCR sidecar for OCR (`proto/ocr_sidecar.py`)

## Ground Truth
- **Benchmark GT = `E:\Xray\CARS\data\Label@CARS_202609_raw`** (`proto/common.py::LAB`).
  This is the hand-drawn annotation set: its geometry is literally the red
  rectangle burned into `Image@CARS_202609_annotated`, the detector's input
  (97.6% of raw boxes have >=0.8 red-line support in the input image; the
  machine-artifact candidates Compare-V1 / Compare / detector output score
  only ~49%). Evidence: `benchmark/gt_final.json`, `proto/gt_final.py`.
- The raw label files carry a **UTF-8 BOM** — read them with `encoding="utf-8-sig"`.
- `Label@CARS_202609_annotated` holds **detector output**, not GT — never read it as ground truth.
- `Compare-V1@CARS_202609_annotated` / `Compare@CARS_202609_annotated` are machine
  artifacts, not GT (the pre-2026-10-20 baselines were measured against them and
  are therefore invalid).
- Current C# baseline (`cvhybrid`, 500 imgs / 1121 GT boxes): D=3 579 TP / corr_f1 66.2,
  D=5 602 / 68.8, D=8 605 / 69.1, D=12 607 / 69.4 (`benchmark/pm_v7_csharp_rawgt.json`).
  Config: hue<=4, sat>=100, val>=140, edgefit-only, min_area 300, min_side 12,
  ortho_tol 10, boundary_band 6, dedup_iou 0.30, fit_band 8 (`proto/scope40_retune.py`,
  `proto/scope42_combo.py`).
- The detector's input has the human annotation burned in as red rectangles, so the
  red mask must be narrow: the annotation is hue 0-4, the orange X-ray object texture
  is hue 9-10. A wide hue window floods the mask with object pixels and was the
  single largest error source.

## Project Structure
- `BRobbDetector/` — C# project (main)
- `proto/` — Python prototype + OCR sidecar
- `data/` — test images and labels