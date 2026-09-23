using Avalonia.Controls;
using Avalonia.Platform.Storage;
using BRobbDetector.ViewModels;

namespace BRobbDetector.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private MainViewModel? ViewModel => DataContext as MainViewModel;

    private async void OnPickInputDir(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var folder = await PickFolderAsync();
        if (folder != null && ViewModel != null)
            ViewModel.InputDirectory = folder;
    }

    private async void OnPickOutputDir(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var folder = await PickFolderAsync();
        if (folder != null && ViewModel != null)
            ViewModel.OutputDirectory = folder;
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