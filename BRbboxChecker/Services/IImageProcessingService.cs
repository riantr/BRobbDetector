using Avalonia.Media.Imaging;
using BRbboxChecker.Models;
using System;
using System.IO;
using System.Threading.Tasks;

namespace BRbboxChecker.Services;

public interface IImageProcessingService
{
    Task<Bitmap?> LoadImageAsync(string imagePath);
    Task<double?> CalculateGrayMeanAsync(string imagePath, int x1, int y1, int x2, int y2, 
                                          int x1Inner, int y1Inner, int x2Inner, int y2Inner);
    Bitmap? DrawBoundingBox(Bitmap? bitmap, BBoxData bboxData, bool showTarget);
}