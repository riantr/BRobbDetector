using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Detector = BRbboxDetector.Services.BRbboxDetector;
using BRbboxDetector.Models;
using BRbboxDetector.Services;
using BRbboxDetector.Views;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;
using OpenCvSharp;

namespace BRbboxDetector.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private string _inputDir = "";
    public string InputDir
    {
        get => _inputDir;
        set
        {
            if (SetProperty(ref _inputDir, value))
            {
                OnPropertyChanged(nameof(InputDirDisplayName));
                ImagePreview.LoadImagesFromDirectory(value);
            }
        }
    }
    
    public string InputDirDisplayName => string.IsNullOrEmpty(InputDir) ? "" : new DirectoryInfo(InputDir).Name;
    
    private string _templateDir = "";
    public string TemplateDir
    {
        get => _templateDir;
        set
        {
            if (SetProperty(ref _templateDir, value))
            {
                OnPropertyChanged(nameof(TemplateDirDisplayName));
            }
        }
    }
    
    public string TemplateDirDisplayName => string.IsNullOrEmpty(TemplateDir) ? "" : new DirectoryInfo(TemplateDir).Name;
    
    private string _outputDir = "";
    public string OutputDir
    {
        get => _outputDir;
        set
        {
            if (SetProperty(ref _outputDir, value))
            {
                OnPropertyChanged(nameof(OutputDirDisplayName));
                ImagePreview.OutputDir = value;
            }
        }
    }
    
    public string OutputDirDisplayName => string.IsNullOrEmpty(OutputDir) ? "" : new DirectoryInfo(OutputDir).Name;
    
    [ObservableProperty]
    private bool _usePreprocess = true;
    
    private int _minThreshold = 0;
    public int MinThreshold
    {
        get => _minThreshold;
        set
        {
            if (value > MaxThreshold)
                value = MaxThreshold;
            if (SetProperty(ref _minThreshold, value))
            {
                ImagePreview.MinThreshold = value;
            }
        }
    }
    
    private int _maxThreshold = 20;
    public int MaxThreshold
    {
        get => _maxThreshold;
        set
        {
            if (value < MinThreshold)
                value = MinThreshold;
            if (SetProperty(ref _maxThreshold, value))
            {
                ImagePreview.MaxThreshold = value;
            }
        }
    }
    
    [ObservableProperty]
    private bool _useDenoise = true;
    
    [ObservableProperty]
    private int _denoiseThreshold = 50;
    
    partial void OnUsePreprocessChanged(bool value)
    {
        ImagePreview.UsePreprocess = value;
    }
    
    partial void OnUseDenoiseChanged(bool value)
    {
        ImagePreview.UseDenoise = value;
    }
    
    partial void OnDenoiseThresholdChanged(int value)
    {
        ImagePreview.DenoiseThreshold = value;
    }
    
    [ObservableProperty]
    private bool _useHollowTemplate = false;
    
    [ObservableProperty]
    private double _matchThreshold = 0.9;
    
    [ObservableProperty]
    private bool _isProcessing = false;
    
    [ObservableProperty]
    private string _logText = "";
    
    [ObservableProperty]
    private int _progressValue = 0;
    
    [ObservableProperty]
    private int _progressMax = 0;
    
    private CancellationTokenSource? _cts;
    private Detector? _detector;
    
    private ImagePreviewViewModel? _imagePreview;
    
    public ImagePreviewViewModel ImagePreview
    {
        get
        {
            if (_imagePreview == null)
            {
                _imagePreview = new ImagePreviewViewModel();
                _imagePreview.LogAction = Log;
            }
            return _imagePreview;
        }
    }
    
    public MainViewModel()
    {
        InitializeDefaultDirectories();
    }
    
    private void InitializeDefaultDirectories()
    {
        // 获取应用程序基础目录
        string baseDir = AppContext.BaseDirectory;
        
        // 尝试查找Assets目录（可能在项目根目录或bin目录下）
        string? assetsDir = FindAssetsDirectory(baseDir);
        
        if (assetsDir != null)
        {
            // 设置默认目录路径
            string inputDir = Path.Combine(assetsDir, "Image@Example");
            string templateDir = Path.Combine(assetsDir, "Template");
            string outputDir = Path.Combine(assetsDir, "Label@Example");
            
            // 设置模板目录
            if (Directory.Exists(templateDir))
            {
                TemplateDir = templateDir;
            }
            
            // 设置输出目录
            if (Directory.Exists(outputDir))
            {
                OutputDir = outputDir;
            }
            
            // 设置输入目录（最后设置，会触发图像加载）
            if (Directory.Exists(inputDir))
            {
                InputDir = inputDir;
                Log(string.Format("自动加载输入目录: {0}", inputDir));
            }
            
            if (Directory.Exists(templateDir))
            {
                Log(string.Format("自动加载模板目录: {0}", templateDir));
            }
            
            if (Directory.Exists(outputDir))
            {
                Log(string.Format("自动加载输出目录: {0}", outputDir));
            }
        }
    }
    
    private string? FindAssetsDirectory(string baseDir)
    {
        // 检查当前目录及上级目录中的Assets文件夹
        string? currentDir = baseDir;
        
        for (int i = 0; i < 5 && currentDir != null; i++)
        {
            string assetsPath = Path.Combine(currentDir, "Assets");
            if (Directory.Exists(assetsPath))
            {
                return assetsPath;
            }
            currentDir = Directory.GetParent(currentDir)?.FullName;
        }
        
        return null;
    }
    
    [RelayCommand]
    private void ShowGrayscalePreview()
    {
        var selectedPath = ImagePreview.SelectedImagePath;
        if (string.IsNullOrEmpty(selectedPath) || !File.Exists(selectedPath))
        {
            Log("请先选择一张图像");
            return;
        }

        Log("正在生成灰度图预览...");

        try
        {
            var imageFiles = ImagePreview.ImageFiles.ToList();
            
            var viewModel = new GrayscalePreviewViewModel
            {
                MainViewModel = this
            };
            
            viewModel.SyncFromMainViewModel();

            var window = new GrayscalePreviewWindow
            {
                DataContext = viewModel
            };
            
            viewModel.SetWindow(window);
            viewModel.SetImageList(imageFiles, selectedPath);

            if (App.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
                && desktop.MainWindow != null)
            {
                window.ShowDialog(desktop.MainWindow);
            }

            Log("灰度图预览弹窗已显示");
        }
        catch (Exception ex)
        {
            Log(string.Format("生成灰度图失败: {0}", ex.Message));
        }
    }
    
    private void Log(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            LogText += string.Format("[{0:HH:mm:ss}] {1}\n", DateTime.Now, message);
        });
    }
    
    [RelayCommand]
    private async Task StartDetectionAsync()
    {
        if (IsProcessing)
            return;
            
        LogText = "";
        
        if (!Directory.Exists(InputDir))
        {
            Log(string.Format("Error: Input directory does not exist: {0}", InputDir));
            return;
        }
        
        if (!Directory.Exists(TemplateDir))
        {
            Log(string.Format("Error: Template directory does not exist: {0}", TemplateDir));
            return;
        }
        
        Log("Starting detection...");
        Log(string.Format("Input directory: {0}", InputDir));
        Log(string.Format("Template directory: {0}", TemplateDir));
        Log(string.Format("Output directory: {0}", OutputDir));
        
        IsProcessing = true;
        _cts = new CancellationTokenSource();
        
        try
        {
            _detector = new Detector(TemplateDir, OutputDir);
            
            PreprocessParams? preprocessParams = null;
            if (UsePreprocess)
            {
                preprocessParams = new PreprocessParams
                {
                    MinThreshold = MinThreshold,
                    MaxThreshold = MaxThreshold,
                    DenoiseThreshold = UseDenoise ? DenoiseThreshold : null
                };
                Log(string.Format("Preprocess params: Min={0}, Max={1}, Denoise={2}, Threshold={3}", MinThreshold, MaxThreshold, UseDenoise, DenoiseThreshold));
            }
            
            Log(string.Format("Match threshold: {0}", MatchThreshold));
            Log(string.Format("Hollow template: {0}", UseHollowTemplate ? "Enabled" : "Disabled"));
            
            var supportedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
            var files = Directory.GetFiles(InputDir)
                .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .ToList();
            
            if (files.Count == 0)
            {
                Log("No image files found");
                return;
            }
            
            Log(string.Format("Found {0} images", files.Count));
            ProgressMax = files.Count;
            ProgressValue = 0;
            
            await Task.Run(() =>
            {
                var jsonPaths = _detector.DetectBatch(
                    InputDir,
                    preprocessParams,
                    MatchThreshold,
                    UseHollowTemplate,
                    (current, total, filename) =>
                    {
                        Dispatcher.UIThread.Post(() =>
                        {
                            ProgressValue = current;
                            Log(string.Format("[{0}/{1}] Processing: {2}", current, total, filename));
                        });
                    },
                    _cts.Token
                );
                
                Dispatcher.UIThread.Post(() =>
                {
                    Log(string.Format("Detection complete! Processed {0} images", jsonPaths.Count));
                    Log("JSON files saved to:");
                    foreach (var path in jsonPaths)
                    {
                        Log(string.Format("  - {0}", path));
                    }
                });
            }, _cts.Token);
        }
        catch (OperationCanceledException)
        {
            Log("Detection cancelled");
        }
        catch (Exception ex)
        {
            Log(string.Format("Error: {0}", ex.Message));
        }
        finally
        {
            IsProcessing = false;
            _detector?.Dispose();
            _detector = null;
            _cts?.Dispose();
            _cts = null;
        }
    }
    
    [RelayCommand]
    private void CancelDetection()
    {
        _cts?.Cancel();
    }
    
    [RelayCommand]
    private async Task BrowseInputDir()
    {
        if (App.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
            && desktop.MainWindow != null)
        {
            var topLevel = TopLevel.GetTopLevel(desktop.MainWindow);
            if (topLevel == null) return;
            
            var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = "选择输入图像目录",
                AllowMultiple = false
            });
            
            if (folders.Count > 0)
            {
                InputDir = folders[0].Path.LocalPath;
            }
        }
    }
    
    [RelayCommand]
    private async Task BrowseTemplateDir()
    {
        if (App.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
            && desktop.MainWindow != null)
        {
            var topLevel = TopLevel.GetTopLevel(desktop.MainWindow);
            if (topLevel == null) return;
            
            var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = "选择模板目录",
                AllowMultiple = false
            });
            
            if (folders.Count > 0)
            {
                TemplateDir = folders[0].Path.LocalPath;
            }
        }
    }
    
    [RelayCommand]
    private async Task BrowseOutputDir()
    {
        if (App.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
            && desktop.MainWindow != null)
        {
            var topLevel = TopLevel.GetTopLevel(desktop.MainWindow);
            if (topLevel == null) return;
            
            var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = "选择输出目录",
                AllowMultiple = false
            });
            if (folders.Count > 0)
            {
                OutputDir = folders[0].Path.LocalPath;
            }
        }
    }
}

