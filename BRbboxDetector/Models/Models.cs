using System.Text.Json.Serialization;
using Avalonia;

namespace BRbboxDetector.Models;

public class BboxResult
{
    [JsonPropertyName("img_name")]
    public string ImageName { get; set; } = string.Empty;
    
    [JsonPropertyName("infos")]
    public List<BboxInfo> Infos { get; set; } = new();
}

public class BboxInfo
{
    [JsonPropertyName("points")]
    public string Points { get; set; } = string.Empty;
    
    [JsonPropertyName("type")]
    public string Type { get; set; } = "bbox";
    
    // 解析Points字符串获取坐标点
    public List<PointD> GetPoints()
    {
        var result = new List<PointD>();
        if (string.IsNullOrEmpty(Points))
            return result;
            
        // Points格式: "x1,y1;x2,y2;x3,y3;x4,y4"
        var pointStrs = Points.Split(';');
        foreach (var ps in pointStrs)
        {
            var coords = ps.Split(',');
            if (coords.Length == 2 && 
                double.TryParse(coords[0], out double x) && 
                double.TryParse(coords[1], out double y))
            {
                result.Add(new PointD { X = x, Y = y });
            }
        }
        return result;
    }
}

public class PointD
{
    public double X { get; set; }
    public double Y { get; set; }
}

// 用于UI绑定的边界框
public class BoundingBoxDisplay
{
    // 四个顶点坐标
    public double X1 { get; set; }
    public double Y1 { get; set; }
    public double X2 { get; set; }
    public double Y2 { get; set; }
    public double X3 { get; set; }
    public double Y3 { get; set; }
    public double X4 { get; set; }
    public double Y4 { get; set; }
    
    // 用于显示的标签
    public string Label { get; set; } = "";
    
    // 用于Avalonia Polygon绑定的Points字符串
    public string Points => string.Format("{0},{1} {2},{3} {4},{5} {6},{7}", 
        X1, Y1, X2, Y2, X3, Y3, X4, Y4);
    
    // 对角线1：点1到点3
    public Point Diagonal1Start => new Point(X1, Y1);
    public Point Diagonal1End => new Point(X3, Y3);
    
    // 对角线2：点2到点4
    public Point Diagonal2Start => new Point(X2, Y2);
    public Point Diagonal2End => new Point(X4, Y4);
}

public class PreprocessParams
{
    public int? MinThreshold { get; set; }
    public int? MaxThreshold { get; set; }
    public int? DenoiseThreshold { get; set; }
}

public class MatchResult
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public double Score { get; set; }
    public double Scale { get; set; }
    public string Method { get; set; } = "normal";
}
