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

    [ObservableProperty]
    private int _satMin = 80;

    [ObservableProperty]
    private int _valMin = 110;

    [ObservableProperty]
    private int _holeMinArea = 300;

    [ObservableProperty]
    private int _holeMinSide = 12;

    [ObservableProperty]
    private double _holeMinFillPercent = 50;

    [ObservableProperty]
    private double _holeMinSupportPercent = 50;

    [ObservableProperty]
    private int _lineOpenLength = 15;

    [ObservableProperty]
    private bool _enableOcr;

    [ObservableProperty]
    private string _categoryList = "子弹,工具,枪支,液体,火种,烟火制品,管制器具,锂电池,锐器";

    [ObservableProperty]
    private int _ocrTextPositionIndex;

    [ObservableProperty]
    private int _ocrTextExtent = 35;

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
    private string _previewInfo = string.Empty;

    [ObservableProperty]
    private int _previewWidth;

    [ObservableProperty]
    private int _previewHeight;

    public ObservableCollection<ImageDetectionResult> Results { get; } = new();

    public ObservableCollection<string> LogMessages { get; } = new();

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

    private DetectionParams BuildParams()
    {
        return new DetectionParams
        {
            ColorRules = new List<ColorRule>
            {
                new ColorRule
                {
                    Type = TypeName,
                    HueLow = 0,
                    HueHigh = 10,
                    HueLow2 = 170,
                    HueHigh2 = 180,
                    SatMin = SatMin,
                    ValMin = ValMin
                }
            },
            HoleMinArea = HoleMinArea,
            HoleMinSide = HoleMinSide,
            HoleMinFill = HoleMinFillPercent / 100.0,
            HoleMinSupport = HoleMinSupportPercent / 100.0,
            LineOpenLength = LineOpenLength
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
        Results.Clear();
        PreviewImage = null;
        Overlays = new List<BoxOverlay>();
        PreviewInfo = string.Empty;

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

                    if ((i + 1) % 50 == 0 || i == 0)
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

                    var result = new ImageDetectionResult
                    {
                        ImagePath = file,
                        ImageName = name,
                        JsonPath = jsonPath,
                        Boxes = boxes
                    };
                    Dispatcher.UIThread.Post(() => Results.Add(result));

                    if ((i + 1) % 500 == 0)
                        GC.Collect(2, GCCollectionMode.Optimized, false);
                }

                Dispatcher.UIThread.Post(() =>
                    ProgressText = $"Done: {files.Count} images, {totalBoxes} boxes, {errors} errors");
            }, _cts.Token);

            Log($"[Done] Processed {files.Count} image(s), {totalBoxes} box(es). Output: {outputDir}");
            if (Results.Count > 0)
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

    private bool CanPrev() => Results.Count > 0 && SelectedResult != null && Results.IndexOf(SelectedResult) > 0;

    [RelayCommand(CanExecute = nameof(CanPrev))]
    private void Prev()
    {
        var idx = Results.IndexOf(SelectedResult!);
        if (idx > 0)
            SelectedResult = Results[idx - 1];
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

            Overlays = result.Boxes.Select(b => new BoxOverlay
            {
                Points = BoxOverlay.ToPointsString(b.Corners.Select(c => ((double)c.X, (double)c.Y))),
                Stroke = GetBrushForType(b.Type)
            }).ToList();

            PreviewInfo = $"{result.ImageName} - {result.Boxes.Count} box(es)";
        }
        catch (Exception ex)
        {
            Log($"[Error] Failed to load preview: {ex.Message}");
        }
    }

    private IBrush GetBrushForType(string type)
    {
        var rule = _activeParams?.ColorRules.FirstOrDefault(r => r.Type == type);
        return rule == null ? Brushes.Cyan : BoxOverlay.CreateComplementBrush(rule);
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