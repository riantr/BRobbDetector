namespace BRobbDetector.Models;

public class BBoxData
{
    public string ImageName { get; set; } = string.Empty;
    public int BBoxIndex { get; set; }
    public double? GrayMean { get; set; }
    public int X1 { get; set; }
    public int Y1 { get; set; }
    public int X2 { get; set; }
    public int Y2 { get; set; }
    public int X1Inner { get; set; }
    public int Y1Inner { get; set; }
    public int X2Inner { get; set; }
    public int Y2Inner { get; set; }
    public string Type { get; set; } = string.Empty;
    public string Points { get; set; } = string.Empty;
    public bool MarkedForDelete { get; set; }
}

public class BBoxDataCsvRecord
{
    public int Index { get; set; }
    public string ImageName { get; set; } = string.Empty;
    public int BBoxIndex { get; set; }
    public double? GrayMean { get; set; }
    public int X1 { get; set; }
    public int Y1 { get; set; }
    public int X2 { get; set; }
    public int Y2 { get; set; }
    public int X1Inner { get; set; }
    public int Y1Inner { get; set; }
    public int X2Inner { get; set; }
    public int Y2Inner { get; set; }
    public string Type { get; set; } = string.Empty;
}