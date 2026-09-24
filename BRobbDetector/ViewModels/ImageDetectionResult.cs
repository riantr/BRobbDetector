using System.Collections.Generic;
using Avalonia.Media.Imaging;
using BRobbDetector.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace BRobbDetector.ViewModels;

public partial class ImageDetectionResult : ObservableObject
{
    public string ImagePath { get; set; } = string.Empty;
    public string ImageName { get; set; } = string.Empty;
    public string JsonPath { get; set; } = string.Empty;

    [ObservableProperty]
    private List<DetectedBox> _boxes = new();

    [ObservableProperty]
    private Bitmap? _thumbnail;

    public string Summary => $"{ImageName} ({Boxes.Count} box)";
    public int BoxCount => Boxes.Count;

    partial void OnBoxesChanged(List<DetectedBox> value)
    {
        OnPropertyChanged(nameof(BoxCount));
        OnPropertyChanged(nameof(Summary));
    }
}
