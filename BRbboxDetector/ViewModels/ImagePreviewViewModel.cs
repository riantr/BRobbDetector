using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using BRbboxDetector.Services;
using BRbboxDetector.Models;
using OpenCvSharp;

namespace BRbboxDetector.ViewModels;

public partial class ImagePreviewViewModel : ObservableObject
{
    private readonly ImageProcessor _processor = new();

    // 日志回调
    public Action<string>? LogAction { get; set; }

    [ObservableProperty]
    private ObservableCollection<string> _imageFiles = new();

    [ObservableProperty]
    private string? _selectedImagePath;

    [ObservableProperty]
    private Bitmap? _originalImage;

    [ObservableProperty]
    private Bitmap? _preprocessedImage;

    [ObservableProperty]
    private bool _hasOriginalImage;

    [ObservableProperty]
    private bool _hasPreprocessedImage;

    [ObservableProperty]
    private bool _usePreprocess;

    [ObservableProperty]
    private int _minThreshold;

    [ObservableProperty]
    private int _maxThreshold = 20;

    [ObservableProperty]
    private bool _useDenoise = true;

    [ObservableProperty]
    private int _denoiseThreshold = 50;

    [ObservableProperty]
    private bool _isPreviewMode;

    // 输出目录（用于查找JSON文件）
    [ObservableProperty]
    private string _outputDir = "";

    // 边界框列表用于绑定
    [ObservableProperty]
    private ObservableCollection<BoundingBoxDisplay> _boundingBoxes = new();

    // 图像尺寸（用于Canvas缩放）
    [ObservableProperty]
    private double _imageWidth;

    [ObservableProperty]
    private double _imageHeight;

    // 是否有边界框数据
    [ObservableProperty]
    private bool _hasBoundingBoxes;

    public void LoadImagesFromDirectory(string directoryPath)
    {
        ImageFiles.Clear();
        
        if (string.IsNullOrEmpty(directoryPath) || !Directory.Exists(directoryPath))
            return;

        var supportedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(directoryPath)
            .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToList();

        foreach (var file in files)
        {
            ImageFiles.Add(file);
        }

        if (ImageFiles.Count > 0)
        {
            SelectedImagePath = ImageFiles[0];
        }
    }

    partial void OnSelectedImagePathChanged(string? value)
    {
        if (!string.IsNullOrEmpty(value) && File.Exists(value))
        {
            LoadOriginalImage(value);
            UpdatePreprocessedImage();
            LoadBoundingBoxes(value);
        }
        else
        {
            OriginalImage = null;
            PreprocessedImage = null;
            HasOriginalImage = false;
            HasPreprocessedImage = false;
            BoundingBoxes.Clear();
            HasBoundingBoxes = false;
        }
    }

    partial void OnOutputDirChanged(string value)
    {
        // 输出目录改变时，重新加载当前图像的边界框
        if (!string.IsNullOrEmpty(SelectedImagePath))
        {
            LoadBoundingBoxes(SelectedImagePath);
        }
    }

    private void LoadOriginalImage(string path)
    {
        try
        {
            OriginalImage = new Bitmap(path);
            HasOriginalImage = true;
            ImageWidth = OriginalImage.Size.Width;
            ImageHeight = OriginalImage.Size.Height;
        }
        catch
        {
            OriginalImage = null;
            HasOriginalImage = false;
            ImageWidth = 0;
            ImageHeight = 0;
        }
    }

    private void LoadBoundingBoxes(string imagePath)
    {
        BoundingBoxes.Clear();
        HasBoundingBoxes = false;

        if (string.IsNullOrEmpty(OutputDir) || !Directory.Exists(OutputDir))
            return;

        try
        {
            // 构建JSON文件路径
            string imageName = Path.GetFileNameWithoutExtension(imagePath);
            string jsonPath = Path.Combine(OutputDir, imageName + ".json");

            if (!File.Exists(jsonPath))
            {
                LogAction?.Invoke(string.Format("未找到JSON文件: {0}", jsonPath));
                return;
            }

            // 读取并解析JSON
            string jsonContent = File.ReadAllText(jsonPath);
            var result = JsonSerializer.Deserialize<BboxResult>(jsonContent);

            if (result?.Infos == null || result.Infos.Count == 0)
            {
                LogAction?.Invoke(string.Format("JSON文件无边界框数据: {0}", jsonPath));
                return;
            }

            // 输出解析到的JSON内容
            LogAction?.Invoke(string.Format("加载JSON: {0}", jsonPath));
            LogAction?.Invoke(string.Format("  图像名称: {0}", result.ImageName));
            LogAction?.Invoke(string.Format("  边界框数量: {0}", result.Infos.Count));

            // 转换为显示用的边界框
            for (int i = 0; i < result.Infos.Count; i++)
            {
                var info = result.Infos[i];
                var points = info.GetPoints();
                
                // 输出每个边界框信息
                LogAction?.Invoke(string.Format("  [{0}] Points: {1}, Type: {2}", i + 1, info.Points, info.Type));
                
                if (points.Count >= 4)
                {
                    BoundingBoxes.Add(new BoundingBoxDisplay
                    {
                        X1 = points[0].X,
                        Y1 = points[0].Y,
                        X2 = points[1].X,
                        Y2 = points[1].Y,
                        X3 = points[2].X,
                        Y3 = points[2].Y,
                        X4 = points[3].X,
                        Y4 = points[3].Y,
                        Label = string.Format("#{0}", i + 1)
                    });
                }
            }

            HasBoundingBoxes = BoundingBoxes.Count > 0;
            LogAction?.Invoke(string.Format("成功加载 {0} 个边界框", BoundingBoxes.Count));
        }
        catch (Exception ex)
        {
            LogAction?.Invoke(string.Format("解析JSON失败: {0}", ex.Message));
        }
    }

    partial void OnUsePreprocessChanged(bool value)
    {
        UpdatePreprocessedImage();
    }

    partial void OnMinThresholdChanged(int value)
    {
        UpdatePreprocessedImage();
    }

    partial void OnMaxThresholdChanged(int value)
    {
        UpdatePreprocessedImage();
    }

    partial void OnUseDenoiseChanged(bool value)
    {
        UpdatePreprocessedImage();
    }

    partial void OnDenoiseThresholdChanged(int value)
    {
        UpdatePreprocessedImage();
    }

    public void UpdatePreprocessedImage()
    {
        if (string.IsNullOrEmpty(SelectedImagePath) || !File.Exists(SelectedImagePath))
        {
            PreprocessedImage = null;
            HasPreprocessedImage = false;
            return;
        }

        if (!UsePreprocess)
        {
            PreprocessedImage = OriginalImage;
            HasPreprocessedImage = HasOriginalImage;
            return;
        }

        try
        {
            using var mat = Cv2.ImRead(SelectedImagePath);
            if (mat.Empty())
            {
                PreprocessedImage = null;
                HasPreprocessedImage = false;
                return;
            }

            // Convert to grayscale first for preview
            using var grayMat = new Mat();
            if (mat.Channels() == 1)
            {
                mat.CopyTo(grayMat);
            }
            else
            {
                Cv2.CvtColor(mat, grayMat, ColorConversionCodes.BGR2GRAY);
            }

            // Apply preprocessing (threshold mask + denoise)
            var preprocessParams = new PreprocessParams
            {
                MinThreshold = MinThreshold,
                MaxThreshold = MaxThreshold,
                DenoiseThreshold = UseDenoise ? DenoiseThreshold : null
            };

            using var preprocessed = _processor.PreprocessImage(
                mat,
                preprocessParams.MinThreshold,
                preprocessParams.MaxThreshold,
                preprocessParams.DenoiseThreshold
            );

            // Convert preprocessed result to grayscale for display
            using var grayPreview = new Mat();
            if (preprocessed.Channels() == 1)
            {
                preprocessed.CopyTo(grayPreview);
            }
            else
            {
                Cv2.CvtColor(preprocessed, grayPreview, ColorConversionCodes.BGR2GRAY);
            }
            
            // Convert grayscale (1-channel) to BGRA (4-channel) for proper display
            using var bgraPreview = new Mat();
            Cv2.CvtColor(grayPreview, bgraPreview, ColorConversionCodes.GRAY2BGRA);
            
            // Encode BGRA to PNG
            byte[] pngBytes = bgraPreview.ImEncode(".png");
            
            // Load PNG bytes into Bitmap
            using var ms = new MemoryStream(pngBytes);
            PreprocessedImage?.Dispose();
            PreprocessedImage = new Bitmap(ms);
            HasPreprocessedImage = true;
        }
        catch
        {
            PreprocessedImage = null;
            HasPreprocessedImage = false;
        }
    }

    public void ClearImages()
    {
        ImageFiles.Clear();
        SelectedImagePath = null;
        OriginalImage?.Dispose();
        OriginalImage = null;
        PreprocessedImage?.Dispose();
        PreprocessedImage = null;
        HasOriginalImage = false;
        HasPreprocessedImage = false;
        BoundingBoxes.Clear();
        HasBoundingBoxes = false;
    }
}