using System;
using System.IO;
using System.Linq;
using BRobbDetector.Models;

namespace BRobbDetector;

public static class CliRunner
{
    public static void Run(string[] args)
    {
        if (args.Length >= 3 && args[0] == "--check")
        {
            RunCheck(args[1], args[2]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--cli")
        {
            RunDetect(args[1], args[2]);
            return;
        }

        Console.WriteLine("Usage: BRobbDetector --cli <inputDir> <outputDir>");
        Console.WriteLine("       BRobbDetector --check <imageDir> <labelDir>");
    }

    private static void RunDetect(string inputDir, string outputDir)
    {
        if (!Directory.Exists(inputDir))
        {
            Console.WriteLine($"Input directory not found: {inputDir}");
            return;
        }

        Directory.CreateDirectory(outputDir);

        var detector = new Services.ColorBoxDetector();
        var jsonService = new Services.LabelJsonService();
        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };

        var files = Directory.GetFiles(inputDir)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToList();

        int errors = 0;
        int totalBoxes = 0;
        for (int i = 0; i < files.Count; i++)
        {
            var file = files[i];
            var name = Path.GetFileName(file);

            List<Services.DetectedBox> boxes;
            try
            {
                boxes = detector.DetectFile(file, new DetectionParams());
            }
            catch (Exception ex)
            {
                errors++;
                Console.WriteLine($"[Error] {name}: {ex.Message}");
                continue;
            }

            var outputPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(file) + ".json");
            jsonService.WriteResult(outputPath, name, boxes);
            totalBoxes += boxes.Count;

            if ((i + 1) % 500 == 0)
                Console.WriteLine($"[Progress] {i + 1}/{files.Count} boxes={totalBoxes} errors={errors}");
        }

        Console.WriteLine($"[Done] {files.Count} images, {totalBoxes} boxes, {errors} errors -> {outputDir}");
    }

    private static void RunCheck(string imageDir, string labelDir)
    {
        if (!Directory.Exists(imageDir) || !Directory.Exists(labelDir))
        {
            Console.WriteLine($"Directory not found: {imageDir} or {labelDir}");
            return;
        }

        var jsonService = new Services.LabelJsonService();
        var checker = new Services.CheckerService();
        var csv = new Services.CsvService();

        var boxes = jsonService.LoadBoxesFromDir(labelDir, imageDir);
        Console.WriteLine($"Loaded {boxes.Count} box(es) from {labelDir}");

        foreach (var box in boxes)
        {
            var imagePath = Path.Combine(imageDir, box.ImageName);
            if (File.Exists(imagePath))
                box.GrayMean = checker.CalculateGrayMeanAsync(imagePath, box).GetAwaiter().GetResult();
            Console.WriteLine(
                $"  {box.ImageName} #{box.BBoxIndex} [{box.Type}] rect=({box.X1},{box.Y1})-({box.X2},{box.Y2}) GrayMean={box.GrayMean:0.##}");
        }

        var csvPath = Path.Combine(AppContext.BaseDirectory, $"{Path.GetFileName(imageDir.TrimEnd(Path.DirectorySeparatorChar))}.csv");
        csv.SaveToCsvAsync(csvPath, boxes).GetAwaiter().GetResult();
        Console.WriteLine($"CSV saved: {csvPath}");
        Console.WriteLine(File.ReadAllText(csvPath));
    }
}