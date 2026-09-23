using Avalonia.Controls;
using Avalonia.Interactivity;
using BRbboxChecker.ViewModels;

namespace BRbboxChecker;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainWindowViewModel();
    }
}