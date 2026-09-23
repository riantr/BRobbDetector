using Avalonia.Controls;
using BRbboxDetector.ViewModels;

namespace BRbboxDetector.Views;

public partial class GrayscalePreviewWindow : Window
{
    public GrayscalePreviewWindow()
    {
        InitializeComponent();
    }

    private void OnWindowClosed(object? sender, System.EventArgs e)
    {
        if (DataContext is GrayscalePreviewViewModel viewModel)
        {
            viewModel.Cleanup();
        }
    }
}