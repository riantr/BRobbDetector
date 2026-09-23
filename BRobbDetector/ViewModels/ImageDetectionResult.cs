using System.Collections.Generic;
using System.Linq;
using BRobbDetector.Services;

namespace BRobbDetector.ViewModels;

public class ImageDetectionResult
{
    public string ImagePath { get; set; } = string.Empty;
    public string ImageName { get; set; } = string.Empty;
    public string JsonPath { get; set; } = string.Empty;
    public List<DetectedBox> Boxes { get; set; } = new();

    public string Summary => $"{ImageName} ({Boxes.Count} box)";
    public int BoxCount => Boxes.Count;
}