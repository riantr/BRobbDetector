using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using BRobbDetector.Models;
using BRobbDetector.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OpenCvSharp;

namespace BRobbDetector.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly ColorBoxDetector _detector;
    private readonly LabelJsonService _jsonService;
    private CancellationTokenSource? _cts;
    private DetectionParams? _activeParams;
    private OcrService? _ocrService;
    private CategoryNormalizer? _categoryNormalizer;

    [ObservableProperty]
    private string _inputDirectory = string.Empty;

    [ObservableProperty]
    private string _outputDirectory = string.Empty;

    [ObservableProperty]
    private string _typeName = "液体";

    public ObservableCollection<CategoryEntry> TypeCategories { get; } = new()
    {
        new CategoryEntry { Name = "液体" },
    };

    [ObservableProperty]
    private int _targetColorIndex = 3;

    [ObservableProperty]
    private int _satMin = 80;

    [ObservableProperty]
    private int _valMin = 110;

    [ObservableProperty]
    private int _holeMinArea = 300;

    [ObservableProperty]
    private int _holeMaxArea = 100000;

    [ObservableProperty]
    private int _holeMinSide = 12;

    [ObservableProperty]
    private int _holeMaxSide = 500;

    [ObservableProperty]
    private double _holeMinFillPercent = 50;

    [ObservableProperty]
    private double _holeMaxFillPercent = 100;

    [ObservableProperty]
    private double _holeMinSupportPercent = 50;

    [ObservableProperty]
    private double _holeMaxSupportPercent = 100;

    [ObservableProperty]
    private int _lineOpenLength = 15;

    [ObservableProperty]
    private int _lineOpenStep = 15;

    [ObservableProperty]
    private double _holeSplitSolidityPercent = 88;

    [ObservableProperty]
    private double _edgeMinSupportPercent = 25;

    [ObservableProperty]
    private int _minEdgesWithSupport = 3;

    [ObservableProperty]
    private bool _enableOcr;

    [ObservableProperty]
    private string _categoryList = "子弹,工具,枪支,液体,火种,烟火制品,管制器具,锂电池,锐器";

    [ObservableProperty]
    private int _ocrTextPositionIndex;

    [ObservableProperty]
    private int _ocrTextExtent = 35;

    [ObservableProperty]
    private int _displayMode;

    [ObservableProperty]
    private bool _isProcessing;

    [ObservableProperty]
    private string _progressText = string.Empty;

    [ObservableProperty]
    private ImageDetectionResult? _selectedResult;

    [ObservableProperty]
    private Bitmap? _previewImage;

    [ObservableProperty]
    private List<BoxOverlay> _overlays = new();

    [ObservableProperty]
    private List<TextOverlay> _textOverlays = new();

    [ObservableProperty]
    private string _previewInfo = string.Empty;

    [ObservableProperty]
    private int _previewWidth;

    [ObservableProperty]
    private int _previewHeight;

    [ObservableProperty]
    private int _canvasWidth = 800;

    [ObservableProperty]
    private int _canvasHeight = 600;

    [ObservableProperty]
    private int _selectedBoxIndex = -1;

    private double _previewScale = 1.0;
    private double _previewOffsetX;
    private double _previewOffsetY;

    public ObservableCollection<ImageDetectionResult> Results { get; } = new();

    public ObservableCollection<string> LogMessages { get; } = new();

    public ObservableCollection<ColorOption> ColorOptions { get; } = new()
    {
        new ColorOption { Name = "黑", Swatch = Brushes.Black, HueLow = 0, HueHigh = 180, SatMin = 0, SatMax = 255, ValMin = 0, ValMax = 46 },
        new ColorOption { Name = "灰", Swatch = Brushes.Gray, HueLow = 0, HueHigh = 180, SatMin = 0, SatMax = 43, ValMin = 46, ValMax = 220 },
        new ColorOption { Name = "白", Swatch = Brushes.White, HueLow = 0, HueHigh = 180, SatMin = 0, SatMax = 30, ValMin = 221, ValMax = 255 },
        new ColorOption { Name = "红", Swatch = Brushes.Red, HueLow = 0, HueHigh = 10, HueLow2 = 170, HueHigh2 = 180, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255, IsSelected = true },
        new ColorOption { Name = "橙", Swatch = Brushes.Orange, HueLow = 11, HueHigh = 25, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255 },
        new ColorOption { Name = "黄", Swatch = Brushes.Yellow, HueLow = 26, HueHigh = 34, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255 },
        new ColorOption { Name = "绿", Swatch = Brushes.Green, HueLow = 35, HueHigh = 77, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255 },
        new ColorOption { Name = "青", Swatch = Brushes.Cyan, HueLow = 78, HueHigh = 99, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255 },
        new ColorOption { Name = "蓝", Swatch = Brushes.Blue, HueLow = 100, HueHigh = 124, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255 },
        new ColorOption { Name = "紫", Swatch = Brushes.Purple, HueLow = 125, HueHigh = 155, SatMin = 43, SatMax = 255, ValMin = 46, ValMax = 255 },
    };

    public CheckerViewModel Checker { get; }

    public MainViewModel(ColorBoxDetector detector, LabelJsonService jsonService, CheckerViewModel checker)
    {
        _detector = detector;
        _jsonService = jsonService;
        Checker = checker;
        InitializeDefaultDirectories();
    }

    private void InitializeDefaultDirectories()
    {
        var baseDir = AppContext.BaseDirectory;
        var dir = new DirectoryInfo(baseDir);
        for (int i = 0; i < 5 && dir != null; i++)
        {
            var candidate = Path.Combine(dir.FullName, "data", "Image@test");
            if (Directory.Exists(candidate))
            {
                InputDirectory = candidate;
                OutputDirectory = Path.Combine(dir.FullName, "BRobbDetector", "Result");
                Log("[Init] Found default data directory.");
                return;
            }
            dir = dir.Parent;
        }
    }

    partial void OnInputDirectoryChanged(string value)
    {
        _ = LoadImagesFromDirectoryAsync();
    }

    private async Task LoadImagesFromDirectoryAsync()
    {
        Results.Clear();
        SelectedResult = null;
        PreviewImage = null;
        Overlays = new List<BoxOverlay>();
        TextOverlays = new List<TextOverlay>();
        PreviewInfo = string.Empty;

        if (string.IsNullOrWhiteSpace(InputDirectory) || !Directory.Exists(InputDirectory))
            return;

        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(InputDirectory)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToList();

        if (files.Count == 0) return;

        foreach (var file in files)
        {
            Results.Add(new ImageDetectionResult
            {
                ImagePath = file,
                ImageName = Path.GetFileName(file)
            });
        }

        await Task.Run(() =>
        {
            foreach (var result in Results)
            {
                try
                {
                    using var stream = File.OpenRead(result.ImagePath);
                    var thumb = Bitmap.DecodeToWidth(stream, 120);
                    Dispatcher.UIThread.Post(() => result.Thumbnail = thumb);
                }
                catch { }
            }
        });

        if (Results.Count > 0)
            SelectedResult = Results[0];
        Log($"[Load] {Results.Count} image(s) loaded from {InputDirectory}");
    }

    private DetectionParams BuildParams()
    {
        var rules = TypeCategories
            .Where(c => !string.IsNullOrWhiteSpace(c.Name))
            .Select(c => new ColorRule
            {
                Name = c.Name,
                Type = c.Name,
                HueLow = c.HueLow,
                HueHigh = c.HueHigh,
                UseSecondRange = c.HueLow2 >= 0,
                HueLow2 = c.HueLow2,
                HueHigh2 = c.HueHigh2,
                SatMin = c.SatMin,
                SatMax = c.SatMax,
                ValMin = c.ValMin,
                ValMax = c.ValMax
            })
            .ToList();

        if (rules.Count == 0)
        {
            rules.Add(new ColorRule { Type = TypeName });
        }

        return new DetectionParams
        {
            ColorRules = rules,
            HoleMinArea = HoleMinArea,
            HoleMaxArea = HoleMaxArea,
            HoleMinSide = HoleMinSide,
            HoleMaxSide = HoleMaxSide,
            HoleMinFill = HoleMinFillPercent / 100.0,
            HoleMaxFill = HoleMaxFillPercent / 100.0,
            HoleMinSupport = HoleMinSupportPercent / 100.0,
            HoleMaxSupport = HoleMaxSupportPercent / 100.0,
            HoleSplitSolidity = HoleSplitSolidityPercent / 100.0,
            EdgeMinSupport = EdgeMinSupportPercent / 100.0,
            MinEdgesWithSupport = MinEdgesWithSupport,
            LineOpenLength = LineOpenLength,
            LineOpenStep = LineOpenStep
        };
    }

    private bool CanDetect() => !IsProcessing;

    [RelayCommand(CanExecute = nameof(CanDetect))]
    private async Task DetectAsync()
    {
        if (string.IsNullOrWhiteSpace(InputDirectory) || !Directory.Exists(InputDirectory))
        {
            Log("[Error] Input directory does not exist.");
            return;
        }

        var outputDir = string.IsNullOrWhiteSpace(OutputDirectory) ? "Result" : OutputDirectory;
        Directory.CreateDirectory(outputDir);

        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(InputDirectory)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToList();

        if (files.Count == 0)
        {
            Log("[Error] No images found in input directory.");
            return;
        }

        IsProcessing = true;
        DetectCommand.NotifyCanExecuteChanged();
        _cts = new CancellationTokenSource();

        var existingByPath = Results.ToDictionary(r => r.ImagePath);
        if (existingByPath.Count == 0)
        {
            foreach (var file in files)
            {
                Results.Add(new ImageDetectionResult
                {
                    ImagePath = file,
                    ImageName = Path.GetFileName(file)
                });
            }
        }

        var parameters = BuildParams();
        _activeParams = parameters;
        int totalBoxes = 0;

        if (EnableOcr)
        {
            try
            {
                _ocrService = new OcrService();
                _ocrService.Start();
                _categoryNormalizer = new CategoryNormalizer(
                    CategoryList.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                Log("[OCR] Service started.");
            }
            catch (Exception ex)
            {
                Log($"[OCR] Failed to start: {ex.Message}");
                _ocrService = null;
                _categoryNormalizer = null;
            }
        }

        try
        {
            await Task.Run(() =>
            {
                int errors = 0;
                for (int i = 0; i < files.Count; i++)
                {
                    _cts.Token.ThrowIfCancellationRequested();
                    var file = files[i];
                    var name = Path.GetFileName(file);

                    if ((i + 1) % 50 == 0 || i == 0 || files.Count <= 50)
                    {
                        Dispatcher.UIThread.Post(() =>
                            ProgressText = $"[{i + 1}/{files.Count}] boxes={totalBoxes} errors={errors}");
                    }

                    List<DetectedBox> boxes;
                    try
                    {
                        boxes = _detector.DetectFile(file, parameters);
                    }
                    catch (Exception ex)
                    {
                        errors++;
                        Dispatcher.UIThread.Post(() =>
                            LogMessages.Insert(0, $"[Error] {name}: {ex.Message}"));
                        continue;
                    }

                    if (EnableOcr && boxes.Count > 0 && _ocrService != null && _categoryNormalizer != null)
                    {
                        try
                        {
                            boxes = RunOcrForBoxes(file, boxes);
                        }
                        catch (Exception ex)
                        {
                            Dispatcher.UIThread.Post(() =>
                                LogMessages.Insert(0, $"[OCR] {name}: {ex.Message}"));
                        }
                    }

                    var jsonPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(file) + ".json");
                    _jsonService.WriteResult(jsonPath, name, boxes);

                    totalBoxes += boxes.Count;

                    Dispatcher.UIThread.Post(() =>
                    {
                        if (existingByPath.TryGetValue(file, out var existing))
                        {
                            existing.JsonPath = jsonPath;
                            existing.Boxes = boxes;
                            if (ReferenceEquals(existing, SelectedResult))
                                _ = LoadPreviewAsync(existing);
                        }
                        else
                        {
                            Results.Add(new ImageDetectionResult
                            {
                                ImagePath = file,
                                ImageName = name,
                                JsonPath = jsonPath,
                                Boxes = boxes
                            });
                        }
                    });

                    if ((i + 1) % 500 == 0)
                        GC.Collect(2, GCCollectionMode.Optimized, false);
                }

                Dispatcher.UIThread.Post(() =>
                    ProgressText = $"Done: {files.Count} images, {totalBoxes} boxes, {errors} errors");
            }, _cts.Token);

            Log($"[Done] Processed {files.Count} image(s), {totalBoxes} box(es). Output: {outputDir}");
            if (SelectedResult != null)
                await LoadPreviewAsync(SelectedResult);
            else if (Results.Count > 0)
                SelectedResult = Results[0];
        }
        catch (OperationCanceledException)
        {
            Log("[Cancelled] Detection cancelled by user.");
        }
        catch (Exception ex)
        {
            Log($"[Error] {ex.Message}");
        }
        finally
        {
            _ocrService?.Dispose();
            _ocrService = null;
            IsProcessing = false;
            DetectCommand.NotifyCanExecuteChanged();
            ProgressText = string.Empty;
            _cts.Dispose();
            _cts = null;
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        _cts?.Cancel();
    }

    [RelayCommand]
    private void AddCategory()
    {
        TypeCategories.Add(new CategoryEntry { Name = "" });
    }

    [RelayCommand]
    private void RestoreDefaults()
    {
        HoleMinArea = 300;
        HoleMaxArea = 100000;
        HoleMinSide = 12;
        HoleMaxSide = 500;
        HoleMinFillPercent = 50;
        HoleMaxFillPercent = 100;
        HoleMinSupportPercent = 50;
        HoleMaxSupportPercent = 100;
        LineOpenLength = 15;
        LineOpenStep = 15;
        HoleSplitSolidityPercent = 88;
        EdgeMinSupportPercent = 25;
        MinEdgesWithSupport = 3;
        Log("[Restore] Hole detection parameters restored to defaults.");
    }

    private bool CanAutoColorThreshold() => SelectedResult != null && File.Exists(SelectedResult.ImagePath);

    [RelayCommand(CanExecute = nameof(CanAutoColorThreshold))]
    private void AutoColorThreshold()
    {
        var imagePath = SelectedResult!.ImagePath;
        try
        {
            using var image = Cv2.ImRead(imagePath);
            if (image.Empty()) { Log("[AutoColor] Failed to load image."); return; }
            using var hsv = new Mat();
            Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);

            foreach (var cat in TypeCategories)
            {
                using var mask = new Mat();
                Cv2.InRange(hsv,
                    new Scalar(cat.HueLow, cat.SatMin, cat.ValMin),
                    new Scalar(cat.HueHigh, cat.SatMax, cat.ValMax), mask);
                if (cat.HueLow2 >= 0)
                {
                    using var mask2 = new Mat();
                    Cv2.InRange(hsv,
                        new Scalar(cat.HueLow2, cat.SatMin, cat.ValMin),
                        new Scalar(cat.HueHigh2, cat.SatMax, cat.ValMax), mask2);
                    Cv2.BitwiseOr(mask, mask2, mask);
                }

                int count = Cv2.CountNonZero(mask);
                if (count < 10) continue;

                var hVals = new List<int>();
                var sVals = new List<int>();
                var vVals = new List<int>();
                for (int y = 0; y < hsv.Rows; y++)
                {
                    for (int x = 0; x < hsv.Cols; x++)
                    {
                        if (mask.At<byte>(y, x) == 0) continue;
                        var pix = hsv.At<Vec3b>(y, x);
                        hVals.Add(pix.Item0);
                        sVals.Add(pix.Item1);
                        vVals.Add(pix.Item2);
                    }
                }

                if (hVals.Count == 0) continue;
                hVals.Sort(); sVals.Sort(); vVals.Sort();
                int n = hVals.Count;
                int lo = (int)(n * 0.05), hi = (int)(n * 0.95);

                cat.HueLow = Math.Max(0, hVals[lo]);
                cat.HueHigh = Math.Min(180, hVals[hi]);
                cat.SatMin = Math.Max(0, sVals[lo]);
                cat.SatMax = Math.Min(255, sVals[hi]);
                cat.ValMin = Math.Max(0, vVals[lo]);
                cat.ValMax = Math.Min(255, vVals[hi]);

                Log($"[AutoColor] {cat.Name}: H[{cat.HueLow}-{cat.HueHigh}] S[{cat.SatMin}-{cat.SatMax}] V[{cat.ValMin}-{cat.ValMax}] pixels={n}");
            }
        }
        catch (Exception ex)
        {
            Log($"[AutoColor] Error: {ex.Message}");
        }
    }

    [RelayCommand]
    private void RestoreColorDefaults()
    {
        foreach (var cat in TypeCategories)
        {
            cat.HueLow = 0;
            cat.HueHigh = 10;
            cat.HueLow2 = 170;
            cat.HueHigh2 = 180;
            cat.SatMin = 43;
            cat.SatMax = 255;
            cat.ValMin = 46;
            cat.ValMax = 255;
        }
        Log("[Restore] Color rules restored to defaults (red).");
    }

    partial void OnTargetColorIndexChanged(int value)
    {
        if (value < 0 || value >= ColorOptions.Count) return;
        var opt = ColorOptions[value];
        foreach (var cat in TypeCategories)
        {
            cat.HueLow = opt.HueLow;
            cat.HueHigh = opt.HueHigh;
            cat.HueLow2 = opt.HueLow2;
            cat.HueHigh2 = opt.HueHigh2;
            cat.SatMin = opt.SatMin;
            cat.SatMax = opt.SatMax;
            cat.ValMin = opt.ValMin;
            cat.ValMax = opt.ValMax;
        }
        Log($"[TargetColor] Set to {opt.Name}: H[{opt.HueLow}-{opt.HueHigh}] S[{opt.SatMin}-{opt.SatMax}] V[{opt.ValMin}-{opt.ValMax}]");
    }

    [RelayCommand]
    private void RemoveCategory(CategoryEntry entry)
    {
        if (TypeCategories.Count > 1)
            TypeCategories.Remove(entry);
    }

    private bool CanPrev() => Results.Count > 0 && SelectedResult != null && Results.IndexOf(SelectedResult) > 0;

    [RelayCommand(CanExecute = nameof(CanPrev))]
    private void Prev()
    {
        var idx = Results.IndexOf(SelectedResult!);
        if (idx > 0)
            SelectedResult = Results[idx - 1];
    }

    private bool CanRefresh() => SelectedResult != null;

    [RelayCommand(CanExecute = nameof(CanRefresh))]
    private void Refresh()
    {
        _ = LoadPreviewAsync(SelectedResult);
    }

    private bool CanAutoThreshold() => !string.IsNullOrWhiteSpace(InputDirectory) && Directory.Exists(InputDirectory);

    [RelayCommand(CanExecute = nameof(CanAutoThreshold))]
    private void AutoThreshold()
    {
        var imagePath = SelectedResult?.ImagePath;
        if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
        {
            var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
            imagePath = Directory.GetFiles(InputDirectory)
                .FirstOrDefault(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()));
        }
        if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
        {
            Log("[AutoThreshold] No image available.");
            return;
        }

        var parameters = BuildParams();
        List<HoleStats> stats;
        try
        {
            stats = _detector.CollectHoleStats(imagePath, parameters);
        }
        catch (Exception ex)
        {
            Log($"[AutoThreshold] Error: {ex.Message}");
            return;
        }

        if (stats.Count == 0)
        {
            Log("[AutoThreshold] No candidate holes found. Try adjusting HSV rules.");
            return;
        }

        var areas = stats.Select(s => s.Area).OrderBy(v => v).ToList();
        var minSides = stats.Select(s => s.MinSide).OrderBy(v => v).ToList();
        var maxSides = stats.Select(s => s.MaxSide).OrderBy(v => v).ToList();
        var fills = stats.Select(s => s.FillRatio).OrderBy(v => v).ToList();
        var supports = stats.Select(s => s.Support).OrderBy(v => v).ToList();

        double pLow = 0.10, pHigh = 0.90;
        HoleMinArea = Math.Max(50, (int)Percentile(areas, pLow));
        HoleMaxArea = Math.Min(100000, (int)Percentile(areas, pHigh));
        HoleMinSide = Math.Max(5, (int)Percentile(minSides, pLow));
        HoleMaxSide = Math.Min(500, (int)Percentile(maxSides, pHigh));
        HoleMinFillPercent = Math.Max(10, Percentile(fills, pLow) * 100);
        HoleMaxFillPercent = Math.Min(100, Percentile(fills, pHigh) * 100);
        HoleMinSupportPercent = Math.Max(10, Percentile(supports, pLow) * 100);
        HoleMaxSupportPercent = Math.Min(100, Percentile(supports, pHigh) * 100);

        Log($"[AutoThreshold] {stats.Count} candidates from {Path.GetFileName(imagePath)}: " +
            $"area[{HoleMinArea}-{HoleMaxArea}] side[{HoleMinSide}-{HoleMaxSide}] " +
            $"fill[{HoleMinFillPercent:F0}-{HoleMaxFillPercent:F0}] support[{HoleMinSupportPercent:F0}-{HoleMaxSupportPercent:F0}]");
    }

    private static double Percentile(List<double> sorted, double p)
    {
        if (sorted.Count == 0) return 0;
        if (sorted.Count == 1) return sorted[0];
        double idx = p * (sorted.Count - 1);
        int lo = (int)Math.Floor(idx);
        int hi = Math.Min(lo + 1, sorted.Count - 1);
        double frac = idx - lo;
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
    }

    private bool CanNext() => Results.Count > 0 && SelectedResult != null && Results.IndexOf(SelectedResult) < Results.Count - 1;

    [RelayCommand(CanExecute = nameof(CanNext))]
    private void Next()
    {
        var idx = Results.IndexOf(SelectedResult!);
        if (idx >= 0 && idx < Results.Count - 1)
            SelectedResult = Results[idx + 1];
    }

    partial void OnSelectedResultChanged(ImageDetectionResult? value)
    {
        _ = LoadPreviewAsync(value);
        PrevCommand.NotifyCanExecuteChanged();
        RefreshCommand.NotifyCanExecuteChanged();
        NextCommand.NotifyCanExecuteChanged();
    }

    private async Task LoadPreviewAsync(ImageDetectionResult? result)
    {
        if (result == null || !File.Exists(result.ImagePath))
        {
            PreviewImage = null;
            Overlays = new List<BoxOverlay>();
            PreviewInfo = string.Empty;
            PreviewWidth = 0;
            PreviewHeight = 0;
            return;
        }

        try
        {
            var bitmap = await Task.Run(() =>
            {
                using var stream = File.OpenRead(result.ImagePath);
                return new Bitmap(stream);
            });

            var old = PreviewImage;
            PreviewImage = bitmap;
            old?.Dispose();

            PreviewWidth = bitmap.PixelSize.Width;
            PreviewHeight = bitmap.PixelSize.Height;

            double scale = Math.Min((double)CanvasWidth / PreviewWidth, (double)CanvasHeight / PreviewHeight);
            double offsetX = (CanvasWidth - PreviewWidth * scale) / 2.0;
            double offsetY = (CanvasHeight - PreviewHeight * scale) / 2.0;
            _previewScale = scale;
            _previewOffsetX = offsetX;
            _previewOffsetY = offsetY;

            Overlays = DisplayMode == 2
                ? new List<BoxOverlay>()
                : result.Boxes.Select((b, i) => new BoxOverlay
                {
                    Points = BoxOverlay.ToPoints(b.Corners.Select(c => (c.X * scale + offsetX, c.Y * scale + offsetY))),
                    Stroke = GetBrushForBox(b),
                    StrokeThickness = i == SelectedBoxIndex ? 6.0 : 3.0
                }).ToList();

            TextOverlays = DisplayMode == 2
                ? new List<TextOverlay>()
                : result.Boxes.Select(b => CreateTextOverlay(b, scale, offsetX, offsetY)).ToList();

            PreviewInfo = $"{result.ImageName} - {result.Boxes.Count} box(es)";
        }
        catch (Exception ex)
        {
            Log($"[Error] Failed to load preview: {ex.Message}");
        }
    }

    partial void OnDisplayModeChanged(int value)
    {
        _ = LoadPreviewAsync(SelectedResult);
    }

    partial void OnCanvasWidthChanged(int value)
    {
        _ = LoadPreviewAsync(SelectedResult);
    }

    partial void OnCanvasHeightChanged(int value)
    {
        _ = LoadPreviewAsync(SelectedResult);
    }

    partial void OnSelectedBoxIndexChanged(int value)
    {
        RebuildOverlays();
    }

    private void RebuildOverlays()
    {
        if (SelectedResult == null || DisplayMode == 2) return;
        var boxes = SelectedResult.Boxes;
        double scale = _previewScale;
        double offsetX = _previewOffsetX;
        double offsetY = _previewOffsetY;
        Overlays = boxes.Select((b, i) => new BoxOverlay
        {
            Points = BoxOverlay.ToPoints(b.Corners.Select(c => (c.X * scale + offsetX, c.Y * scale + offsetY))),
            Stroke = GetBrushForBox(b),
            StrokeThickness = i == SelectedBoxIndex ? 6.0 : 3.0
        }).ToList();
    }

    public int HitTestBox(double canvasX, double canvasY)
    {
        if (SelectedResult == null || SelectedResult.Boxes.Count == 0)
            return -1;

        double imgX = (canvasX - _previewOffsetX) / _previewScale;
        double imgY = (canvasY - _previewOffsetY) / _previewScale;

        for (int i = 0; i < SelectedResult.Boxes.Count; i++)
        {
            var corners = SelectedResult.Boxes[i].Corners;
            if (PointInPolygon(imgX, imgY, corners))
                return i;
        }
        return -1;
    }

    private static bool PointInPolygon(double x, double y, List<Point> polygon)
    {
        int n = polygon.Count;
        if (n < 3) return false;
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            double xi = polygon[i].X, yi = polygon[i].Y;
            double xj = polygon[j].X, yj = polygon[j].Y;
            if (((yi > y) != (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }

    public void GetBoxHsvValues(int boxIndex)
    {
        if (SelectedResult == null || boxIndex < 0 || boxIndex >= SelectedResult.Boxes.Count)
            return;

        var box = SelectedResult.Boxes[boxIndex];
        var imagePath = SelectedResult.ImagePath;
        if (!File.Exists(imagePath))
        {
            Log($"[HSV] Image not found: {imagePath}");
            return;
        }

        try
        {
            using var image = Cv2.ImRead(imagePath);
            if (image.Empty()) return;
            using var hsv = new Mat();
            Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);

            var corners = box.Corners;
            var hVals = new List<int>();
            var sVals = new List<int>();
            var vVals = new List<int>();

            int samples = 100;
            for (int edge = 0; edge < 4; edge++)
            {
                var p1 = corners[edge];
                var p2 = corners[(edge + 1) % 4];
                for (int t = 0; t <= samples; t++)
                {
                    double frac = (double)t / samples;
                    int x = (int)Math.Round(p1.X + (p2.X - p1.X) * frac);
                    int y = (int)Math.Round(p1.Y + (p2.Y - p1.Y) * frac);
                    if (x < 0 || x >= hsv.Cols || y < 0 || y >= hsv.Rows) continue;
                    var pix = hsv.At<Vec3b>(y, x);
                    hVals.Add(pix.Item0);
                    sVals.Add(pix.Item1);
                    vVals.Add(pix.Item2);
                }
            }

            if (hVals.Count == 0)
            {
                Log("[HSV] No pixels sampled along box border.");
                return;
            }

            Log($"[HSV] Box #{boxIndex} ({box.Type}): " +
                $"H min={hVals.Min()} max={hVals.Max()} mean={(int)hVals.Average()} | " +
                $"S min={sVals.Min()} max={sVals.Max()} mean={(int)sVals.Average()} | " +
                $"V min={vVals.Min()} max={vVals.Max()} mean={(int)vVals.Average()} | " +
                $"samples={hVals.Count}");
        }
        catch (Exception ex)
        {
            Log($"[HSV] Error: {ex.Message}");
        }
    }

    public void ComputeHoleArea(int boxIndex)
    {
        if (SelectedResult == null || boxIndex < 0 || boxIndex >= SelectedResult.Boxes.Count)
            return;

        var box = SelectedResult.Boxes[boxIndex];
        var imagePath = SelectedResult.ImagePath;
        if (!File.Exists(imagePath))
        {
            Log($"[Hole] Image not found: {imagePath}");
            return;
        }

        try
        {
            using var image = Cv2.ImRead(imagePath);
            if (image.Empty()) return;

            var corners = box.Corners.Select(c => new OpenCvSharp.Point2f(c.X, c.Y)).ToArray();
            var ptsInt = corners.Select(c => new OpenCvSharp.Point((int)Math.Round(c.X), (int)Math.Round(c.Y))).ToArray();

            double polyArea = Cv2.ContourArea(ptsInt);

            using var interiorMask = new Mat(image.Rows, image.Cols, MatType.CV_8UC1, Scalar.Black);
            Cv2.FillPoly(interiorMask, new[] { ptsInt }, new Scalar(255));

            using var hsv = new Mat();
            Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);
            using var darkMask = new Mat();
            Cv2.InRange(hsv, new Scalar(0, 0, 0), new Scalar(180, 255, 46), darkMask);

            using var holesInside = new Mat();
            Cv2.BitwiseAnd(darkMask, interiorMask, holesInside);
            int darkArea = Cv2.CountNonZero(holesInside);

            using var borderMask = new Mat();
            Cv2.Polylines(new Mat(image.Rows, image.Cols, MatType.CV_8UC1, Scalar.Black), new[] { ptsInt }, true, new Scalar(255), 3);
            var colorRule = _activeParams?.ColorRules.FirstOrDefault(r => r.Name == box.RuleName && !string.IsNullOrEmpty(box.RuleName))
                ?? _activeParams?.ColorRules.FirstOrDefault(r => r.Type == box.Type);
            int colorArea = 0;
            if (colorRule != null)
            {
                using var colorMask = BuildColorMask(hsv, colorRule);
                using var colorInside = new Mat();
                Cv2.BitwiseAnd(colorMask, interiorMask, colorInside);
                colorArea = Cv2.CountNonZero(colorInside);
            }

            Log($"[Hole] Box #{boxIndex} ({box.Type}): " +
                $"polygon_area={polyArea:F0} | dark_hole_area={darkArea} | color_border_area={colorArea} | " +
                $"fill_ratio={(polyArea > 0 ? darkArea / polyArea * 100 : 0):F1}%");
        }
        catch (Exception ex)
        {
            Log($"[Hole] Error: {ex.Message}");
        }
    }

    private static Mat BuildColorMask(Mat hsv, ColorRule rule)
    {
        var mask = new Mat();
        Cv2.InRange(hsv, new Scalar(rule.HueLow, rule.SatMin, rule.ValMin), new Scalar(rule.HueHigh, rule.SatMax, rule.ValMax), mask);
        if (rule.UseSecondRange)
        {
            using var mask2 = new Mat();
            Cv2.InRange(hsv, new Scalar(rule.HueLow2, rule.SatMin, rule.ValMin), new Scalar(rule.HueHigh2, rule.SatMax, rule.ValMax), mask2);
            Cv2.BitwiseOr(mask, mask2, mask);
        }
        return mask;
    }

    public void MeasureMinSide(int boxIndex)
    {
        if (SelectedResult == null || boxIndex < 0 || boxIndex >= SelectedResult.Boxes.Count)
            return;

        var box = SelectedResult.Boxes[boxIndex];
        var corners = box.Corners;
        if (corners.Count < 2) return;

        var sideLengths = new double[corners.Count];
        for (int i = 0; i < corners.Count; i++)
        {
            var p1 = corners[i];
            var p2 = corners[(i + 1) % corners.Count];
            sideLengths[i] = Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
        }

        double minSide = sideLengths.Min();
        double maxSide = sideLengths.Max();
        Log($"[Side] Box #{boxIndex} ({box.Type}): " +
            $"sides=[{string.Join(", ", sideLengths.Select(s => s.ToString("F1")))}] | " +
            $"min={minSide:F1} max={maxSide:F1} ratio={(minSide > 0 ? maxSide / minSide : 0):F2}");
    }

    private IBrush GetBrushForBox(DetectedBox box)
    {
        if (DisplayMode == 1)
        {
            var opt = ColorOptions.FirstOrDefault(c => c.Name == box.RuleName);
            if (opt != null) return opt.Swatch;
        }

        var rule = _activeParams?.ColorRules.FirstOrDefault(r => r.Name == box.RuleName && !string.IsNullOrEmpty(box.RuleName))
            ?? _activeParams?.ColorRules.FirstOrDefault(r => r.Type == box.Type);
        return rule == null ? Brushes.Cyan : BoxOverlay.CreateComplementBrush(rule);
    }

    private TextOverlay CreateTextOverlay(DetectedBox box, double scale, double offsetX, double offsetY)
    {
        int minX = box.Corners.Min(c => c.X);
        int minY = box.Corners.Min(c => c.Y);
        int maxX = box.Corners.Max(c => c.X);
        int maxY = box.Corners.Max(c => c.Y);
        double cx = (minX + maxX) / 2.0;
        double cy = (minY + maxY) / 2.0;
        int margin = 20;

        double tx, ty;
        int opposite = OcrTextPositionIndex switch
        {
            0 => 1,  // Above → Below
            1 => 0,  // Below → Above
            2 => 3,  // Left → Right
            3 => 2,  // Right → Left
            _ => 4   // Inside → Inside
        };

        switch (opposite)
        {
            case 0: tx = cx; ty = minY - margin; break;       // Above
            case 1: tx = cx; ty = maxY + margin; break;       // Below
            case 2: tx = minX - margin; ty = cy; break;       // Left
            case 3: tx = maxX + margin; ty = cy; break;       // Right
            default: tx = cx; ty = cy; break;                 // Inside
        }

        return new TextOverlay
        {
            Text = box.Type,
            X = tx * scale + offsetX,
            Y = ty * scale + offsetY,
            Foreground = GetBrushForBox(box)
        };
    }

    private List<DetectedBox> RunOcrForBoxes(string imagePath, List<DetectedBox> boxes)
    {
        using var image = Cv2.ImRead(imagePath);
        if (image.Empty())
            return boxes;

        var tempDir = Path.Combine(Path.GetTempPath(), "brobb_ocr");
        Directory.CreateDirectory(tempDir);
        var result = new List<DetectedBox>();

        foreach (var box in boxes)
        {
            int minX = box.Corners.Min(c => c.X);
            int minY = box.Corners.Min(c => c.Y);
            int maxX = box.Corners.Max(c => c.X);
            int maxY = box.Corners.Max(c => c.Y);

            var pos = (TextPosition)OcrTextPositionIndex;
            var rect = OcrService.CropForOcr(image.Width, image.Height, minX, minY, maxX, maxY,
                pos, OcrTextExtent);

            if (rect.Width < 10 || rect.Height < 10)
            {
                result.Add(box);
                continue;
            }

            using var crop = new Mat(image, rect);
            var tempPath = Path.Combine(tempDir, $"crop_{Guid.NewGuid():N}.jpg");
            Cv2.ImWrite(tempPath, crop);

            try
            {
                var ocrResult = _ocrService!.Recognize(tempPath);
                var catResult = _categoryNormalizer!.Normalize(ocrResult.Text);
                var newBox = new DetectedBox
                {
                    Corners = box.Corners,
                    Type = string.IsNullOrEmpty(catResult.Category) ? box.Type : catResult.Category,
                    OcrConfidence = catResult.Confidence
                };
                result.Add(newBox);
            }
            finally
            {
                try { File.Delete(tempPath); } catch { }
            }
        }

        return result;
    }

    private void Log(string message)
    {
        Dispatcher.UIThread.Post(() => LogMessages.Insert(0, message));
    }
}