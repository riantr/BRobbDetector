using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using BRobbDetector.Services;
using BRobbDetector.ViewModels;
using BRobbDetector.Views;
using Microsoft.Extensions.DependencyInjection;

namespace BRobbDetector;

public class App : Application
{
    public static ServiceProvider Services { get; private set; } = null!;

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        var services = new ServiceCollection();
        ConfigureServices(services);
        Services = services.BuildServiceProvider();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = new MainWindow
            {
                DataContext = Services.GetRequiredService<MainViewModel>()
            };
        }

        base.OnFrameworkInitializationCompleted();
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        services.AddSingleton<ColorBoxDetector>();
        services.AddSingleton<LabelJsonService>();
        services.AddSingleton<CsvService>();
        services.AddSingleton<CheckerService>();
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<CheckerViewModel>();
    }
}