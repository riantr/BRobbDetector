using Avalonia;
using System;

namespace BRobbDetector;

internal class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        if (args.Length >= 1 && args[0].StartsWith("--", StringComparison.Ordinal))
        {
            CliRunner.Run(args);
            return;
        }

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .LogToTrace();
}