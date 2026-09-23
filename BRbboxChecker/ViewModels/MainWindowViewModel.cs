using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using BRbboxChecker.Models;
using BRbboxChecker.Services;
using ReactiveUI;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;

namespace BRbboxChecker.ViewModels;

public class MainWindowViewModel : ViewModelBase
{
    private readonly IImageProcessingService _imageProcessingService;
    private readonly ICsvService _csvService;
    private readonly IJsonService _jsonService;

    private string _imageDir = @"D:\src\opencode\BRbboxChecker\Image@Xray_SY_Bboxed";
    private string _labelDir = @"D:\src\opencode\BRbboxChecker\Label@Xray_SY_Bboxed";
    private string? _csvFile;
    
    private List<BBoxData> _data = new();
    private List<BBoxData> _originalData = new();
    private List<DeleteItem> _deleteList = new();
    
    private int _currentIndex = 0;
    private bool _sortReverse = true;
    private bool _showTarget = true;
    private bool _keepAspectRatio = true;
    private bool _isCalculating = false;
    
    private Bitmap? _currentImage;
    private string _infoText = "就绪";

    public MainWindowViewModel()
    {
        _imageProcessingService = new ImageProcessingService();
        _csvService = new CsvService();
        _jsonService = new JsonService();

        // 命令
        SelectImageFolderCommand = ReactiveCommand.CreateFromTask(SelectImageFolderAsync);
        SelectLabelFolderCommand = ReactiveCommand.CreateFromTask(SelectLabelFolderAsync);
        PreviousImageCommand = ReactiveCommand.Create(PreviousImage);
        NextImageCommand = ReactiveCommand.Create(NextImage);
        SortAscendingCommand = ReactiveCommand.Create(SortAscending);
        SortDescendingCommand = ReactiveCommand.Create(SortDescending);
        CalculateGrayMeanCommand = ReactiveCommand.CreateFromTask(CalculateGrayMeanAsync);
        DeleteAllCommand = ReactiveCommand.CreateFromTask(DeleteAllAsync);

        // 初始化
        _ = LoadDataAsync();
    }

    // 属性
    public string ImageDir
    {
        get => _imageDir;
        set => this.RaiseAndSetIfChanged(ref _imageDir, value);
    }

    public string LabelDir
    {
        get => _labelDir;
        set => this.RaiseAndSetIfChanged(ref _labelDir, value);
    }

    public List<string> FileList => _data.Select((item, index) => 
        $"{index + 1}. {item.ImageName} - {item.GrayMean?.ToString("F2") ?? "N/A"}").ToList();

    public List<string> DeleteList => _deleteList.Select(item => item.ImageName).ToList();

    public int CurrentIndex
    {
        get => _currentIndex;
        set
        {
            this.RaiseAndSetIfChanged(ref _currentIndex, value);
            this.RaisePropertyChanged(nameof(FileList));
            _ = ShowCurrentImageAsync();
        }
    }

    public bool SortReverse
    {
        get => _sortReverse;
        set => this.RaiseAndSetIfChanged(ref _sortReverse, value);
    }

    public bool ShowTarget
    {
        get => _showTarget;
        set
        {
            this.RaiseAndSetIfChanged(ref _showTarget, value);
            _ = ShowCurrentImageAsync();
        }
    }

    public bool KeepAspectRatio
    {
        get => _keepAspectRatio;
        set
        {
            this.RaiseAndSetIfChanged(ref _keepAspectRatio, value);
            _ = ShowCurrentImageAsync();
        }
    }

    public bool IsCalculating
    {
        get => _isCalculating;
        set => this.RaiseAndSetIfChanged(ref _isCalculating, value);
    }

    public Bitmap? CurrentImage
    {
        get => _currentImage;
        set => this.RaiseAndSetIfChanged(ref _currentImage, value);
    }

    public string InfoText
    {
        get => _infoText;
        set => this.RaiseAndSetIfChanged(ref _infoText, value);
    }

    // 命令
    public ReactiveCommand<Unit, Unit> SelectImageFolderCommand { get; }
    public ReactiveCommand<Unit, Unit> SelectLabelFolderCommand { get; }
    public ReactiveCommand<Unit, Unit> PreviousImageCommand { get; }
    public ReactiveCommand<Unit, Unit> NextImageCommand { get; }
    public ReactiveCommand<Unit, Unit> SortAscendingCommand { get; }
    public ReactiveCommand<Unit, Unit> SortDescendingCommand { get; }
    public ReactiveCommand<Unit, Unit> CalculateGrayMeanCommand { get; }
    public ReactiveCommand<Unit, Unit> DeleteAllCommand { get; }

    // 方法
    private async Task LoadDataAsync()
    {
        // 设置CSV文件路径
        var imageFolderName = Path.GetFileName(_imageDir.TrimEnd(Path.DirectorySeparatorChar));
        _csvFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"{imageFolderName}.csv");

        // 检查CSV文件是否存在
        if (File.Exists(_csvFile))
        {
            InfoText = "从CSV加载...";
            _data = await _csvService.LoadFromCsvAsync(_csvFile);
            InfoText = $"从CSV加载完成! 共 {_data.Count} 个边界框";
        }
        else
        {
            InfoText = "从JSON加载...";
            _data = await _jsonService.LoadFromJsonAsync(_labelDir, _imageDir);
            InfoText = $"从JSON加载完成! 共 {_data.Count} 个边界框（未计算灰度值）";
        }

        _originalData = new List<BBoxData>(_data);
        UpdateButtonStates();
    }

    private async Task SelectImageFolderAsync()
    {
        // TODO: 实现文件夹选择对话框
        InfoText = "选择图像文件夹功能待实现";
    }

    private async Task SelectLabelFolderAsync()
    {
        // TODO: 实现文件夹选择对话框
        InfoText = "选择标注文件夹功能待实现";
    }

    private void PreviousImage()
    {
        if (_data.Count > 0 && _currentIndex > 0)
        {
            CurrentIndex--;
        }
    }

    private void NextImage()
    {
        if (_data.Count > 0 && _currentIndex < _data.Count - 1)
        {
            CurrentIndex++;
        }
    }

    private void SortAscending()
    {
        if (_sortReverse)
        {
            _sortReverse = false;
            _data = _data.OrderBy(x => x.GrayMean ?? double.MaxValue).ToList();
            this.RaisePropertyChanged(nameof(FileList));
            CurrentIndex = 0;
        }
    }

    private void SortDescending()
    {
        if (!_sortReverse)
        {
            _sortReverse = true;
            _data = _data.OrderByDescending(x => x.GrayMean ?? double.MinValue).ToList();
            this.RaisePropertyChanged(nameof(FileList));
            CurrentIndex = 0;
        }
    }

    private async Task CalculateGrayMeanAsync()
    {
        if (_data.Count == 0) return;

        IsCalculating = true;
        InfoText = "正在计算灰度均值...";

        try
        {
            foreach (var item in _data.Where(x => x.GrayMean == null))
            {
                var imagePath = Path.Combine(_imageDir, item.ImageName);
                if (File.Exists(imagePath))
                {
                    item.GrayMean = await _imageProcessingService.CalculateGrayMeanAsync(
                        imagePath, item.X1, item.Y1, item.X2, item.Y2,
                        item.X1Inner, item.Y1Inner, item.X2Inner, item.Y2Inner);
                }
            }

            // 保存到CSV
            await _csvService.SaveToCsvAsync(_csvFile!, _data);

            InfoText = $"计算完成! 共 {_data.Count} 个边界框";
            this.RaisePropertyChanged(nameof(FileList));
        }
        finally
        {
            IsCalculating = false;
        }
    }

    private async Task ShowCurrentImageAsync()
    {
        if (_data.Count == 0 || _currentIndex < 0 || _currentIndex >= _data.Count)
        {
            CurrentImage = null;
            return;
        }

        var currentData = _data[_currentIndex];
        var imagePath = Path.Combine(_imageDir, currentData.ImageName);

        if (!File.Exists(imagePath))
        {
            InfoText = $"图像文件不存在: {currentData.ImageName}";
            return;
        }

        try
        {
            var bitmap = await _imageProcessingService.LoadImageAsync(imagePath);
            CurrentImage = _imageProcessingService.DrawBoundingBox(bitmap, currentData, _showTarget);
            InfoText = $"{currentData.ImageName}\n边界框: ({currentData.X1}, {currentData.Y1}) - ({currentData.X2}, {currentData.Y2})\n灰度均值: {currentData.GrayMean?.ToString("F2") ?? "N/A"}";
        }
        catch (Exception ex)
        {
            InfoText = $"加载图像失败: {ex.Message}";
        }

        UpdateButtonStates();
    }

    private void UpdateButtonStates()
    {
        // 更新按钮状态（在视图中通过CanExecute实现）
    }

    private async Task DeleteAllAsync()
    {
        // TODO: 实现全部删除功能
        InfoText = "全部删除功能待实现";
    }
}

public class ViewModelBase : ReactiveObject
{
}