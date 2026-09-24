using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace BRobbDetector.ViewModels;

public partial class ColorOption : ObservableObject
{
    public string Name { get; init; } = string.Empty;
    public IBrush Swatch { get; init; } = Brushes.Gray;
    public int HueLow { get; init; }
    public int HueHigh { get; init; }
    public int HueLow2 { get; init; } = -1;
    public int HueHigh2 { get; init; } = -1;

    [ObservableProperty]
    private bool _isSelected;
}