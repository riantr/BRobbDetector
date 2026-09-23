using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace BRobbDetector.Views;

public class BoxCountColorConverter : IValueConverter
{
    public static readonly BoxCountColorConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is int count)
            return count == 0 ? Brushes.Gray : Brushes.LimeGreen;
        return Brushes.White;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotSupportedException();
    }
}