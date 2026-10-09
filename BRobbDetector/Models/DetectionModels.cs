using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BRobbDetector.Models;

public enum TextPosition
{
    Above,
    Below,
    Left,
    Right,
    Inside
}

public class ColorRule
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "液体";

    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("h_low")]
    public int HueLow { get; set; } = 0;

    [JsonPropertyName("h_high")]
    public int HueHigh { get; set; } = 10;

    [JsonPropertyName("h_low2")]
    public int HueLow2 { get; set; } = 170;

    [JsonPropertyName("h_high2")]
    public int HueHigh2 { get; set; } = 180;

    [JsonPropertyName("s_min")]
    public int SatMin { get; set; } = 80;

    [JsonPropertyName("s_max")]
    public int SatMax { get; set; } = 255;

    [JsonPropertyName("v_min")]
    public int ValMin { get; set; } = 110;

    [JsonPropertyName("v_max")]
    public int ValMax { get; set; } = 255;

    public bool UseSecondRange { get; set; } = true;

    public int LabAMin { get; set; } = 165;
    public int LabAMax { get; set; } = 255;
    public int LabBMin { get; set; } = 110;
    public int LabBMax { get; set; } = 185;
    public int LabLMin { get; set; } = 60;
    public int LabLMax { get; set; } = 200;

    public int RgbRMin { get; set; } = 120;
    public int RgbRMax { get; set; } = 255;
    public int RgbGMin { get; set; } = 0;
    public int RgbGMax { get; set; } = 120;
    public int RgbBMin { get; set; } = 0;
    public int RgbBMax { get; set; } = 120;
}

public class DetectionParams
{
    public List<ColorRule> ColorRules { get; set; } = new()
    {
        new ColorRule()
    };

    public double MaxImageFraction { get; set; } = 0.5;

    public int TextGlyphMaxSize { get; set; } = 28;
    public int TextGlyphMaxArea { get; set; } = 450;
    public int TextMinCluster { get; set; } = 2;
    public int TextJoinX { get; set; } = 21;
    public int TextJoinY { get; set; } = 7;

    public int LineOpenLength { get; set; } = 15;
    public int LineOpenStep { get; set; } = 15;

    public int HoleMinArea { get; set; } = 300;
    public int HoleMaxArea { get; set; } = 100000;
    public int HoleMinSide { get; set; } = 12;
    public int HoleMaxSide { get; set; } = 500;
    public double HoleMinFill { get; set; } = 0.50;
    public double HoleMaxFill { get; set; } = 1.0;
    public double HoleMinSupport { get; set; } = 0.50;
    public double HoleMaxSupport { get; set; } = 1.0;
    public int HoleStroke { get; set; } = 3;
    public int CloseIterations { get; set; } = 2;
    public double HoleSplitSolidity { get; set; } = 0.88;

    public double NmsThreshold { get; set; } = 0.4;
    public double DedupContainThreshold { get; set; } = 0.85;

    public double EdgeMinSupport { get; set; } = 0.25;
    public int MinEdgesWithSupport { get; set; } = 3;

    public int TwoPassEdgeBand { get; set; } = 5;
    public int TwoPassShrink { get; set; } = 2;
    public int TwoPassMaxCompSize { get; set; } = 30;
    public int TwoPassMaxCompArea { get; set; } = 600;

    public TextPosition OcrTextPosition { get; set; } = TextPosition.Above;
    public int OcrTextExtent { get; set; } = 35;

    public string DetectionMethod { get; set; } = "hole";
    public string FilterMethod { get; set; } = "none";
    public string ColorSpace { get; set; } = "hsvLabRgb";
    public string CloseMode { get; set; } = "iter1";
    public string PurifyMode { get; set; } = "none";
}
public class AutoOptimizeSweepPoint
{
    public string Parameter { get; set; } = "";
    public int Value { get; set; }
    public double Retention { get; set; }
    public double Signal { get; set; }
}

public class AutoOptimizeResult
{
    public int OptimalSatMin { get; set; }
    public int OptimalSatMax { get; set; } = 255;
    public int OptimalValMin { get; set; }
    public int OptimalValMax { get; set; } = 255;
    public double OptimalRetention { get; set; }
    public List<AutoOptimizeSweepPoint> SweepData { get; set; } = new();
}
