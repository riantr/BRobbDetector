using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace BRobbDetector.ViewModels;

public partial class ColorOption : ObservableObject
{
    public string Name { get; set; } = string.Empty;
    public IBrush Swatch { get; set; } = Brushes.Gray;
    public int HueLow { get; set; }
    public int HueHigh { get; set; }
    public int HueLow2 { get; set; } = -1;
    public int HueHigh2 { get; set; } = -1;
    public int SatMin { get; set; }
    public int SatMax { get; set; } = 255;
    public int ValMin { get; set; }
    public int ValMax { get; set; } = 255;

    [ObservableProperty]
    private bool _isSelected;
}
