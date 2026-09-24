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
- .NET 10, Avalonia 11.2.8 (MVVM), OpenCvSharp4 4.9
- Python RapidOCR sidecar for OCR (`proto/ocr_sidecar.py`)

## Project Structure
- `BRobbDetector/` — C# project (main)
- `proto/` — Python prototype + OCR sidecar
- `data/` — test images and labels