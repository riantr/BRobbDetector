using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using BRobbDetector.Models;
using BRobbDetector.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace BRobbDetector.ViewModels;

public partial class CheckerViewModel : ViewModelBase
{
    private readonly LabelJsonService _jsonService;
    private readonly CsvService _csvService;
    private readonly CheckerService _checkerService;

    private List<BBoxData> _data = new();
    private string? _csvFile;

    [ObservableProperty]
    private string _imageDirectory = string.Empty;

    [ObservableProperty]
    private string _labelDirectory = string.Empty;

    [ObservableProperty]
    private string _infoText = string.Empty;

    [ObservableProperty]
    private bool _isBusy;

    [ObservableProperty]
    private Bitmap? _previewImage;

    [ObservableProperty]
    private List<BoxOverlay> _overlays = new();

    [ObservableProperty]
    private BBoxData? _currentItem;

    [ObservableProperty]
    private int _currentIndex;

    [ObservableProperty]
    private string _positionText = string.Empty;

    public List<BBoxData> Data => _data;

    public CheckerViewModel(LabelJsonService jsonService, CsvService csvService, CheckerService checkerService)
    {
        _jsonService = jsonService;
        _csvService = csvService;
        _checkerService = checkerService;
    }

    private bool CanRun() => !IsBusy;

    [RelayCommand(CanExecute = nameof(CanRun))]
    private async Task LoadDataAsync()
    {
        if (string.IsNullOrWhiteSpace(LabelDirectory) || !Directory.Exists(LabelDirectory))
        {
            InfoText = "[Error] Label directory does not exist.";
            return;
        }

        IsBusy = true;
        LoadDataCommand.NotifyCanExecuteChanged();

        try
        {
            var imageFolderName = string.IsNullOrWhiteSpace(ImageDirectory)
                ? "LabelData"
                : Path.GetFileName(ImageDirectory.TrimEnd(Path.DirectorySeparatorChar));

            _csvFile = Path.Combine(AppContext.BaseDirectory, $"{imageFolderName}.csv");

            if (File.Exists(_csvFile))
            {
                _data = await _csvService.LoadFromCsvAsync(_csvFile);
                InfoText = $"Loaded from CSV. {_data.Count} box(es).";
            }
            else
            {
                _data = await Task.Run(() => _jsonService.LoadBoxesFromDir(LabelDirectory, ImageDirectory));
                InfoText = $"Loaded from JSON. {_data.Count} box(es) (gray mean not computed).";
            }

            OnPropertyChanged(nameof(Data));
            CurrentIndex = 0;
            await ShowCurrentImageAsync();
        }
        catch (Exception ex)
        {
            InfoText = $"[Error] Load failed: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
            LoadDataCommand.NotifyCanExecuteChanged();
        }
    }

    [RelayCommand(CanExecute = nameof(CanRun))]
    private async Task CalculateAsync()
    {
        if (_data.Count == 0)
            return;

        IsBusy = true;
        CalculateCommand.NotifyCanExecuteChanged();

        try
        {
            InfoText = "Calculating gray mean...";
            int computed = 0;

            foreach (var item in _data.Where(x => x.GrayMean == null))
            {
                var imagePath = Path.Combine(ImageDirectory, item.ImageName);
                if (File.Exists(imagePath))
                {
                    item.GrayMean = await _checkerService.CalculateGrayMeanAsync(imagePath, item);
                    if (item.GrayMean.HasValue)
                        computed++;
                }
            }

            if (_csvFile != null)
                await _csvService.SaveToCsvAsync(_csvFile, _data);

            InfoText = $"Calculated {computed} box(es). Sorted view not changed.";
        }
        catch (Exception ex)
        {
            InfoText = $"[Error] Calculate failed: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
            CalculateCommand.NotifyCanExecuteChanged();
        }
    }

    [RelayCommand]
    private void SortAscending()
    {
        _data = _data.OrderBy(d => d.GrayMean ?? double.MaxValue).ToList();
        OnPropertyChanged(nameof(Data));
        CurrentIndex = 0;
        _ = ShowCurrentImageAsync();
    }

    [RelayCommand]
    private void SortDescending()
    {
        _data = _data.OrderByDescending(d => d.GrayMean ?? double.MinValue).ToList();
        OnPropertyChanged(nameof(Data));
        CurrentIndex = 0;
        _ = ShowCurrentImageAsync();
    }

    [RelayCommand]
    private async Task PreviousAsync()
    {
        if (_data.Count == 0 || CurrentIndex <= 0)
            return;
        CurrentIndex--;
        await ShowCurrentImageAsync();
    }

    [RelayCommand]
    private async Task NextAsync()
    {
        if (_data.Count == 0 || CurrentIndex >= _data.Count - 1)
            return;
        CurrentIndex++;
        await ShowCurrentImageAsync();
    }

    [RelayCommand]
    private void ToggleMarkForDelete()
    {
        if (CurrentItem == null)
            return;
        CurrentItem.MarkedForDelete = !CurrentItem.MarkedForDelete;
        _ = ShowCurrentImageAsync();
    }

    [RelayCommand(CanExecute = nameof(CanRun))]
    private async Task ApplyDeletionsAsync()
    {
        if (_data.Count == 0 || string.IsNullOrWhiteSpace(LabelDirectory))
            return;

        int marked = _data.Count(d => d.MarkedForDelete);
        if (marked == 0)
        {
            InfoText = "No boxes marked for delete.";
            return;
        }

        IsBusy = true;
        ApplyDeletionsCommand.NotifyCanExecuteChanged();

        try
        {
            await Task.Run(() => _jsonService.ApplyDeletions(LabelDirectory, _data));

            var kept = _data.Where(d => !d.MarkedForDelete).ToList();
            int removed = _data.Count - kept.Count;
            _data = kept;
            OnPropertyChanged(nameof(Data));

            if (_csvFile != null)
                await _csvService.SaveToCsvAsync(_csvFile, _data);

            if (CurrentIndex >= _data.Count)
                CurrentIndex = Math.Max(0, _data.Count - 1);

            InfoText = $"Deleted {removed} box(es) from JSON labels.";
            await ShowCurrentImageAsync();
        }
        catch (Exception ex)
        {
            InfoText = $"[Error] Delete failed: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
            ApplyDeletionsCommand.NotifyCanExecuteChanged();
        }
    }

    private async Task ShowCurrentImageAsync()
    {
        if (_data.Count == 0)
        {
            CurrentItem = null;
            PreviewImage = null;
            Overlays = new List<BoxOverlay>();
            PositionText = string.Empty;
            return;
        }

        CurrentIndex = Math.Clamp(CurrentIndex, 0, _data.Count - 1);
        var item = _data[CurrentIndex];
        CurrentItem = item;
        PositionText = $"{CurrentIndex + 1} / {_data.Count}";

        var imagePath = Path.Combine(ImageDirectory, item.ImageName);
        if (!File.Exists(imagePath))
        {
            PreviewImage = null;
            Overlays = new List<BoxOverlay>();
            InfoText = $"[Warn] Image not found: {item.ImageName}";
            return;
        }

        try
        {
            var bitmap = await Task.Run(() =>
            {
                using var stream = File.OpenRead(imagePath);
                return new Bitmap(stream);
            });

            var old = PreviewImage;
            PreviewImage = bitmap;
            old?.Dispose();

            var overlays = new List<BoxOverlay>
            {
                new BoxOverlay
                {
                    Points = BoxOverlay.ToPointsString(new List<(double, double)>
                    {
                        (item.X1, item.Y1), (item.X2, item.Y1), (item.X2, item.Y2), (item.X1, item.Y2)
                    }),
                    Stroke = Brushes.Red
                },
                new BoxOverlay
                {
                    Points = BoxOverlay.ToPointsString(new List<(double, double)>
                    {
                        (item.X1Inner, item.Y1Inner), (item.X2Inner, item.Y1Inner),
                        (item.X2Inner, item.Y2Inner), (item.X1Inner, item.Y2Inner)
                    }),
                    Stroke = Brushes.Lime
                }
            };

            Overlays = overlays;
        }
        catch (Exception ex)
        {
            InfoText = $"[Error] Preview failed: {ex.Message}";
        }
    }
}