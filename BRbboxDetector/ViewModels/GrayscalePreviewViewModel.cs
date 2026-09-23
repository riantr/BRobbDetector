using System;
using System.Collections.Generic;
using System.IO;
using Avalonia.Controls;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using BRbboxDetector.Services;
using OpenCvSharp;

namespace BRbboxDetector.ViewModels;

public partial class GrayscalePreviewViewModel : ObservableObject
{
    private readonly ImageProcessor _processor = new();
    private Mat? _originalMat;
    private bool _isUpdating;
    private List<string> _imageFiles = new();
    private int _currentIndex = -1;
    private Avalonia.Controls.Window? _window;

    [ObservableProperty]
    private Bitmap? _grayscaleImage;

    [ObservableProperty]
    private string _imagePath = "";

    [ObservableProperty]
    private int _minThreshold;

    [ObservableProperty]
    private int _maxThreshold = 20;

    [ObservableProperty]
    private bool _useDenoise;

    [ObservableProperty]
    private int _denoiseThreshold = 50;

    [ObservableProperty]
    private string _imageCounter = "1 / 1";

    // Reference to main view model for sync
    public MainViewModel? MainViewModel { get; set; }

    public void SetWindow(Avalonia.Controls.Window window)
    {
        _window = window;
    }

    public void SetImageList(List<string> imageFiles, string currentPath)
    {
        _imageFiles = imageFiles;
        _currentIndex = _imageFiles.IndexOf(currentPath);
        
        if (_currentIndex < 0 && _imageFiles.Count > 0)
        {
            _currentIndex = 0;
        }
        
        UpdateCounter();
        LoadCurrentImage();
    }

    private void UpdateCounter()
    {
        if (_imageFiles.Count == 0)
        {
            ImageCounter = "0 / 0";
        }
        else
        {
            ImageCounter = string.Format("{0} / {1}", _currentIndex + 1, _imageFiles.Count);
        }
    }

    private void LoadCurrentImage()
    {
        if (_currentIndex < 0 || _currentIndex >= _imageFiles.Count) return;
        
        var path = _imageFiles[_currentIndex];
        ImagePath = path;
        _originalMat?.Dispose();
        _originalMat = Cv2.ImRead(path);
        UpdatePreview();
        UpdateCounter();
    }

    [RelayCommand]
    private void PreviousImage()
    {
        if (_imageFiles.Count == 0) return;
        
        _currentIndex--;
        if (_currentIndex < 0)
        {
            _currentIndex = _imageFiles.Count - 1; // 循环到尾部
        }
        LoadCurrentImage();
    }

    [RelayCommand]
    private void NextImage()
    {
        if (_imageFiles.Count == 0) return;
        
        _currentIndex++;
        if (_currentIndex >= _imageFiles.Count)
        {
            _currentIndex = 0; // 循环到头部
        }
        LoadCurrentImage();
    }

    [RelayCommand]
    private void Close()
    {
        _window?.Close();
    }

    partial void OnMinThresholdChanged(int value)
    {
        if (_isUpdating) return;
        
        // Constraint: min cannot exceed max
        if (value > MaxThreshold)
        {
            _isUpdating = true;
            MinThreshold = MaxThreshold;
            _isUpdating = false;
            return;
        }
        
        SyncToMainViewModel();
        UpdatePreview();
    }

    partial void OnMaxThresholdChanged(int value)
    {
        if (_isUpdating) return;
        
        // Constraint: max cannot be less than min
        if (value < MinThreshold)
        {
            _isUpdating = true;
            MaxThreshold = MinThreshold;
            _isUpdating = false;
            return;
        }
        
        SyncToMainViewModel();
        UpdatePreview();
    }

    partial void OnUseDenoiseChanged(bool value)
    {
        SyncToMainViewModel();
        UpdatePreview();
    }

    partial void OnDenoiseThresholdChanged(int value)
    {
        SyncToMainViewModel();
        UpdatePreview();
    }

    private void SyncToMainViewModel()
    {
        if (MainViewModel == null) return;
        
        MainViewModel.MinThreshold = MinThreshold;
        MainViewModel.MaxThreshold = MaxThreshold;
        MainViewModel.UseDenoise = UseDenoise;
        MainViewModel.DenoiseThreshold = DenoiseThreshold;
    }

    public void SyncFromMainViewModel()
    {
        if (MainViewModel == null) return;
        
        _isUpdating = true;
        MinThreshold = MainViewModel.MinThreshold;
        MaxThreshold = MainViewModel.MaxThreshold;
        UseDenoise = MainViewModel.UseDenoise;
        DenoiseThreshold = MainViewModel.DenoiseThreshold;
        _isUpdating = false;
    }

    private void UpdatePreview()
    {
        if (_originalMat == null || _originalMat.Empty()) return;

        try
        {
            int? denoise = UseDenoise ? DenoiseThreshold : null;
            using var preprocessed = _processor.PreprocessImage(
                _originalMat,
                MinThreshold,
                MaxThreshold,
                denoise
            );

            // Convert to BGRA for display
            using var bgra = new Mat();
            if (preprocessed.Channels() == 1)
            {
                Cv2.CvtColor(preprocessed, bgra, ColorConversionCodes.GRAY2BGRA);
            }
            else
            {
                Cv2.CvtColor(preprocessed, bgra, ColorConversionCodes.BGR2BGRA);
            }

            byte[] pngBytes = bgra.ImEncode(".png");
            using var ms = new MemoryStream(pngBytes);
            
            GrayscaleImage?.Dispose();
            GrayscaleImage = new Bitmap(ms);
        }
        catch
        {
            // Ignore errors during preview update
        }
    }

    public void Cleanup()
    {
        _originalMat?.Dispose();
        _originalMat = null;
        GrayscaleImage?.Dispose();
        GrayscaleImage = null;
    }
}
