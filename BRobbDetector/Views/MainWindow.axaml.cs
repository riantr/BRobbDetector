using System;
using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using BRobbDetector.ViewModels;

namespace BRobbDetector.Views;

public partial class MainWindow : Window
{
    private int _contextBoxIndex = -1;

    public MainWindow()
    {
        InitializeComponent();
    }

    private MainViewModel? ViewModel => DataContext as MainViewModel;

    private async void OnPickInputDir(object? sender, RoutedEventArgs e)
    {
        var folder = await PickFolderAsync();
        if (folder != null && ViewModel != null)
            ViewModel.InputDirectory = folder;
    }

    private async void OnPickOutputDir(object? sender, RoutedEventArgs e)
    {
        var folder = await PickFolderAsync();
        if (folder != null && ViewModel != null)
            ViewModel.OutputDirectory = folder;
    }

    private void OnRemoveCategory(object? sender, RoutedEventArgs e)
    {
        if (sender is Button btn && btn.Tag is CategoryEntry entry && ViewModel != null)
            ViewModel.RemoveCategoryCommand.Execute(entry);
    }

    private void OnCanvasPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (sender is not Control control || ViewModel == null) return;
        var point = e.GetPosition(control);
        int boxIndex = ViewModel.HitTestBox(point.X, point.Y);

        var props = e.GetCurrentPoint(control).Properties;
        if (props.IsRightButtonPressed)
        {
            _contextBoxIndex = boxIndex;
            if (boxIndex >= 0)
                ViewModel.SelectedBoxIndex = boxIndex;
        }
        else if (props.IsLeftButtonPressed)
        {
            ViewModel.SelectedBoxIndex = boxIndex;
        }
    }

    private void OnContextMenuOpening(object? sender, CancelEventArgs e)
    {
        if (_contextBoxIndex < 0)
            e.Cancel = true;
    }

    private void OnGetBoxHsv(object? sender, RoutedEventArgs e)
    {
        ViewModel?.GetBoxHsvValues(_contextBoxIndex);
    }

    private void OnComputeHoleArea(object? sender, RoutedEventArgs e)
    {
        ViewModel?.ComputeHoleArea(_contextBoxIndex);
    }

    private void OnMeasureMinSide(object? sender, RoutedEventArgs e)
    {
        ViewModel?.MeasureMinSide(_contextBoxIndex);
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