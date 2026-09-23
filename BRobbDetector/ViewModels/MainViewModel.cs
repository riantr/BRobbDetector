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

namespace BRobbDetector.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly ColorBoxDetector _detector;
    private readonly LabelJsonService _jsonService;
    private CancellationTokenSource? _cts;
    private DetectionParams? _activeParams;

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

                    var jsonPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(file) + ".json");
                    _jsonService.WriteResult(jsonPath, name, boxes);

                    totalBoxes += boxes.Count;

                    if (boxes.Count > 0)
                    {
                        var result = new ImageDetectionResult
                        {
                            ImagePath = file,
                            ImageName = name,
                            JsonPath = jsonPath,
                            Boxes = boxes
                        };
                        Dispatcher.UIThread.Post(() => Results.Add(result));
                    }

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

    partial void OnSelectedResultChanged(ImageDetectionResult? value)
    {
        _ = LoadPreviewAsync(value);
    }

    private async Task LoadPreviewAsync(ImageDetectionResult? result)
    {
        if (result == null || !File.Exists(result.ImagePath))
        {
            PreviewImage = null;
            Overlays = new List<BoxOverlay>();
            PreviewInfo = string.Empty;
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

    private void Log(string message)
    {
        Dispatcher.UIThread.Post(() => LogMessages.Insert(0, message));
    }
}