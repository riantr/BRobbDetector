using Avalonia.Controls;
using Avalonia.Platform.Storage;
using BRobbDetector.ViewModels;

namespace BRobbDetector.Views;

public partial class CheckerView : UserControl
{
    public CheckerView()
    {
        InitializeComponent();
    }

    private CheckerViewModel? ViewModel => DataContext as CheckerViewModel;

    private async void OnPickImageDir(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var folder = await PickFolderAsync();
        if (folder != null && ViewModel != null)
            ViewModel.ImageDirectory = folder;
    }

    private async void OnPickLabelDir(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var folder = await PickFolderAsync();
        if (folder != null && ViewModel != null)
            ViewModel.LabelDirectory = folder;
    }

    private async System.Threading.Tasks.Task<string?> PickFolderAsync()
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel == null)
            return null;

        var result = await topLevel.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Select folder"
        });

        return result.Count > 0 ? result[0].Path.LocalPath : null;
    }
}