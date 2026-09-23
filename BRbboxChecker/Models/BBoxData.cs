using System;

namespace BRbboxChecker.Models;

public class BBoxData
{
    public string ImageName { get; set; } = string.Empty;
    public int BBoxIndex { get; set; }
    public int X1 { get; set; }
    public int Y1 { get; set; }
    public int X2 { get; set; }
    public int Y2 { get; set; }
    public int X1Inner { get; set; }
    public int Y1Inner { get; set; }
    public int X2Inner { get; set; }
    public int Y2Inner { get; set; }
    public double? GrayMean { get; set; }

    public override string ToString()
    {
        return GrayMean.HasValue 
            ? $"{ImageName} - {GrayMean:F2}" 
            : $"{ImageName} - (未计算)";
    }
}