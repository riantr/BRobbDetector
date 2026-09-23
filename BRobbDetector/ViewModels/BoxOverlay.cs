using System;
using Avalonia.Media;
using BRobbDetector.Models;

namespace BRobbDetector.ViewModels;

public class BoxOverlay
{
    public string Points { get; set; } = string.Empty;
    public IBrush Stroke { get; set; } = Brushes.Cyan;

    public static string ToPointsString(IEnumerable<(double X, double Y)> corners)
    {
        return string.Join(" ", corners.Select(c => $"{c.X:0},{c.Y:0}"));
    }

    /// <summary>
    /// Computes the complementary color brush of an annotation color rule
    /// (hue rotated by 180 degrees, full saturation and brightness).
    /// </summary>
    public static IBrush CreateComplementBrush(ColorRule rule)
    {
        double hueMidDegrees = (rule.HueLow + rule.HueHigh) / 2.0 * 2.0;
        double h = (hueMidDegrees + 180.0) % 360.0;
        var (r, g, b) = HsvToRgb(h, 1.0, 1.0);
        return new SolidColorBrush(Color.FromRgb(r, g, b));
    }

    private static (byte R, byte G, byte B) HsvToRgb(double h, double s, double v)
    {
        double c = v * s;
        double x = c * (1 - Math.Abs((h / 60.0) % 2 - 1));
        double m = v - c;
        double r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        return ((byte)Math.Round((r + m) * 255), (byte)Math.Round((g + m) * 255), (byte)Math.Round((b + m) * 255));
    }
}
