using CommunityToolkit.Mvvm.ComponentModel;

namespace BRobbDetector.ViewModels;

public partial class CategoryEntry : ObservableObject
{
    [ObservableProperty]
    private string _name = string.Empty;

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
}
