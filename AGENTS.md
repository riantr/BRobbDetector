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
- Benchmark GT = `E:\Xray\CARS\data\Compare-V1@CARS_202609_annotated` (`proto/common.py::LAB`).
- `Label@CARS_202609_annotated` holds **detector output**, not GT — never read it as ground truth.

## Project Structure
- `BRobbDetector/` — C# project (main)
- `proto/` — Python prototype + OCR sidecar
- `data/` — test images and labels