using CommunityToolkit.Mvvm.ComponentModel;

namespace BRobbDetector.ViewModels;

public partial class CategoryEntry : ObservableObject
{
    public static readonly (string Name, int HueLow, int HueHigh, int HueLow2, int HueHigh2, int SatMin, int SatMax, int ValMin, int ValMax)[] ColorPresets =
    {
        ("黑", 0, 180, -1, -1, 0, 255, 0, 46),
        ("灰", 0, 180, -1, -1, 0, 43, 46, 220),
        ("白", 0, 180, -1, -1, 0, 30, 221, 255),
        ("红", 0, 10, 170, 180, 43, 255, 46, 255),
        ("橙", 11, 25, -1, -1, 43, 255, 46, 255),
        ("黄", 26, 34, -1, -1, 43, 255, 46, 255),
        ("绿", 35, 77, -1, -1, 43, 255, 46, 255),
        ("青", 78, 99, -1, -1, 43, 255, 46, 255),
        ("蓝", 100, 124, -1, -1, 43, 255, 46, 255),
        ("紫", 125, 155, -1, -1, 43, 255, 46, 255),
    };

    [ObservableProperty]
    private string _name = string.Empty;

    [ObservableProperty]
    private int _targetColorIndex = 3;

    [ObservableProperty]
    private int _hueLow = 0;

    [ObservableProperty]
    private int _hueHigh = 10;

    [ObservableProperty]
    private int _hueLow2 = 170;

    [ObservableProperty]
    private int _hueHigh2 = 180;

    [ObservableProperty]
    private int _satMin = 80;

    [ObservableProperty]
    private int _satMax = 255;

    [ObservableProperty]
    private int _valMin = 110;

    [ObservableProperty]
    private int _valMax = 255;

    [ObservableProperty]
    private int _labLMin = 60;

    [ObservableProperty]
    private int _labLMax = 200;

    [ObservableProperty]
    private int _labAMin = 165;

    [ObservableProperty]
    private int _labAMax = 255;

    [ObservableProperty]
    private int _labBMin = 110;

    [ObservableProperty]
    private int _labBMax = 185;

    [ObservableProperty]
    private int _rgbRMin = 120;

    [ObservableProperty]
    private int _rgbRMax = 255;

    [ObservableProperty]
    private int _rgbGMin = 0;

    [ObservableProperty]
    private int _rgbGMax = 120;

    [ObservableProperty]
    private int _rgbBMin = 0;

    [ObservableProperty]
    private int _rgbBMax = 120;

    partial void OnTargetColorIndexChanged(int value)
    {
        if (value < 0 || value >= ColorPresets.Length) return;
        var p = ColorPresets[value];
        HueLow = p.HueLow;
        HueHigh = p.HueHigh;
        HueLow2 = p.HueLow2;
        HueHigh2 = p.HueHigh2;
        SatMin = p.SatMin;
        SatMax = p.SatMax;
        ValMin = p.ValMin;
        ValMax = p.ValMax;
    }
}
