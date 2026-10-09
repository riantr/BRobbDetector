using System;
using System.IO;
using System.Linq;
using BRobbDetector.Models;
using OpenCvSharp;

namespace BRobbDetector;

public static class CliRunner
{
    public static void Run(string[] args)
    {
        if (args.Length >= 2 && args[0] == "--autoopt")
        {
            RunAutoOpt(args[1]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--check")
        {
            RunCheck(args[1], args[2]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--step0")
        {
            RunStep0(args[1], args[2]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--step1")
        {
            RunStep1(args[1], args[2]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--step2")
        {
            RunStep2(args[1], args[2]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--step3")
        {
            RunStep3(args[1], args[2]);
            return;
        }

        if (args.Length >= 3 && args[0] == "--cli")
        {
            var useOcr = args.Contains("--ocr");
            var methodIdx = Array.IndexOf(args, "--method");
            var method = methodIdx >= 0 && methodIdx + 1 < args.Length ? args[methodIdx + 1] : "hole";
            var filterIdx = Array.IndexOf(args, "--filter");
            var filter = filterIdx >= 0 && filterIdx + 1 < args.Length ? args[filterIdx + 1] : "none";
            var csIdx = Array.IndexOf(args, "--colorspace");
            var colorspace = csIdx >= 0 && csIdx + 1 < args.Length ? args[csIdx + 1] : "hsvLabRgb";
            var cmIdx = Array.IndexOf(args, "--closemode");
            var closemode = cmIdx >= 0 && cmIdx + 1 < args.Length ? args[cmIdx + 1] : "iter1";
            var purifyIdx = Array.IndexOf(args, "--purify");
            var purify = purifyIdx >= 0 && purifyIdx + 1 < args.Length ? args[purifyIdx + 1] : "none";
            RunDetect(args[1], args[2], useOcr, method, filter, colorspace, closemode, purify);
            return;
        }

        Console.WriteLine("Usage: BRobbDetector --cli <inputDir> <outputDir> [--ocr] [--method hole|...] [--filter none|...] [--colorspace hsv|hsvLab|hsvRgb|hsvLabRgb] [--closemode default|iter1|directional|directional2] [--purify none|lab]");
        Console.WriteLine("       BRobbDetector --step0 <inputDir> <outputDir>");
        Console.WriteLine("       BRobbDetector --step1 <inputDir> <outputDir>");
        Console.WriteLine("       BRobbDetector --step2 <inputDir> <outputDir>");
        Console.WriteLine("       BRobbDetector --step3 <inputDir> <outputDir>");
        Console.WriteLine("       BRobbDetector --check <imageDir> <labelDir>");
    }

    private static void RunDetect(string inputDir, string outputDir, bool useOcr, string method = "hole", string filter = "none", string colorspace = "hsvLabRgb", string closemode = "iter1", string purify = "none")
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

        Services.OcrService? ocrService = null;
        Services.CategoryNormalizer? normalizer = null;

        if (useOcr)
        {
            try
            {
                ocrService = new Services.OcrService();
                ocrService.Start();
                normalizer = new Services.CategoryNormalizer();
                Console.WriteLine("[OCR] Service started.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[OCR] Failed to start: {ex.Message}");
                ocrService = null;
                normalizer = null;
            }
        }

        int errors = 0;
        int totalBoxes = 0;
        for (int i = 0; i < files.Count; i++)
        {
            var file = files[i];
            var name = Path.GetFileName(file);

            List<Services.DetectedBox> boxes;
            try
            {
                boxes = detector.DetectFile(file, new DetectionParams { DetectionMethod = method, FilterMethod = filter, ColorSpace = colorspace, CloseMode = closemode, PurifyMode = purify });
            }
            catch (Exception ex)
            {
                errors++;
                Console.WriteLine($"[Error] {name}: {ex.Message}");
                continue;
            }

            if (ocrService != null && normalizer != null && boxes.Count > 0)
            {
                boxes = RunOcrForBoxes(file, boxes, ocrService, normalizer, TextPosition.Above, 35);
            }

            var outputPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(file) + ".json");
            jsonService.WriteResult(outputPath, name, boxes);
            totalBoxes += boxes.Count;

            if ((i + 1) % 50 == 0 || i == 0)
                Console.WriteLine($"[Progress] {i + 1}/{files.Count} boxes={totalBoxes} errors={errors}");
        }

        ocrService?.Dispose();
        Console.WriteLine($"[Done] {files.Count} images, {totalBoxes} boxes, {errors} errors -> {outputDir}");
    }

    private static List<Services.DetectedBox> RunOcrForBoxes(string imagePath, List<Services.DetectedBox> boxes,
        Services.OcrService ocrService, Services.CategoryNormalizer normalizer,
        TextPosition textPos = TextPosition.Above, int textExtent = 35)
    {
        using var image = Cv2.ImRead(imagePath);
        if (image.Empty())
            return boxes;

        var tempDir = Path.Combine(Path.GetTempPath(), "brobb_ocr");
        Directory.CreateDirectory(tempDir);
        var result = new List<Services.DetectedBox>();

        foreach (var box in boxes)
        {
            int minX = box.Corners.Min(c => c.X);
            int minY = box.Corners.Min(c => c.Y);
            int maxX = box.Corners.Max(c => c.X);
            int maxY = box.Corners.Max(c => c.Y);

            var rect = Services.OcrService.CropForOcr(image.Width, image.Height, minX, minY, maxX, maxY,
                textPos, textExtent);

            if (rect.Width < 10 || rect.Height < 10)
            {
                result.Add(box);
                continue;
            }

            using var crop = new Mat(image, rect);
            var tempPath = Path.Combine(tempDir, $"crop_{Guid.NewGuid():N}.jpg");
            Cv2.ImWrite(tempPath, crop);

            try
            {
                var ocrResult = ocrService.Recognize(tempPath);
                var catResult = normalizer.Normalize(ocrResult.Text);
                result.Add(new Services.DetectedBox
                {
                    Corners = box.Corners,
                    Type = string.IsNullOrEmpty(catResult.Category) ? box.Type : catResult.Category,
                    OcrConfidence = catResult.Confidence
                });
                if (!string.IsNullOrEmpty(ocrResult.Text))
                    Console.WriteLine($"    [OCR] raw='{ocrResult.Text}' -> '{catResult.Category}' conf={catResult.Confidence:0.00}");
            }
            finally
            {
                try { File.Delete(tempPath); } catch { }
            }
        }

        return result;
    }

    private static void RunStep0(string inputDir, string outputDir)
    {
        if (!Directory.Exists(inputDir)) { Console.WriteLine($"Input not found: {inputDir}"); return; }
        Directory.CreateDirectory(outputDir);
        var detector = new Services.ColorBoxDetector();
        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(inputDir)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f).ToList();
        int count = 0;
        foreach (var file in files)
        {
            using var image = Cv2.ImRead(file);
            using var mask = detector.Step0Binarize(image, new DetectionParams());
            var outPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(file) + ".jpg");
            Cv2.ImWrite(outPath, mask);
            count++;
            if (count % 50 == 0 || count == 1) Console.WriteLine($"[Step0] {count}/{files.Count}");
        }
        Console.WriteLine($"[Step0 Done] {count} images -> {outputDir}");
    }

    private static void RunStep1(string inputDir, string outputDir)
    {
        if (!Directory.Exists(inputDir)) { Console.WriteLine($"Input not found: {inputDir}"); return; }
        Directory.CreateDirectory(outputDir);
        var detector = new Services.ColorBoxDetector();
        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(inputDir)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f).ToList();
        int count = 0;
        foreach (var file in files)
        {
            using var mask = Cv2.ImRead(file, ImreadModes.Grayscale);
            using var filtered = detector.Step1Filter(mask, new DetectionParams());
            var outPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(file) + ".jpg");
            Cv2.ImWrite(outPath, filtered);
            count++;
            if (count % 50 == 0 || count == 1) Console.WriteLine($"[Step1] {count}/{files.Count}");
        }
        Console.WriteLine($"[Step1 Done] {count} images -> {outputDir}");
    }

    private static void RunStep2(string inputDir, string outputDir)
    {
        if (!Directory.Exists(inputDir)) { Console.WriteLine($"Input not found: {inputDir}"); return; }
        Directory.CreateDirectory(outputDir);
        var detector = new Services.ColorBoxDetector();
        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(inputDir)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f).ToList();
        int count = 0;
        foreach (var file in files)
        {
            using var mask = Cv2.ImRead(file, ImreadModes.Grayscale);
            using var preserved = detector.Step2PreserveHoles(mask, new DetectionParams());
            var stem = Path.GetFileNameWithoutExtension(file);
            Cv2.ImWrite(Path.Combine(outputDir, stem + ".jpg"), preserved);
            count++;
            if (count % 50 == 0 || count == 1) Console.WriteLine($"[Step2] {count}/{files.Count}");
        }
        Console.WriteLine($"[Step2 Done] {count} images -> {outputDir}");
    }

    private static void RunStep3(string inputDir, string outputDir)
    {
        if (!Directory.Exists(inputDir)) { Console.WriteLine($"Input not found: {inputDir}"); return; }
        Directory.CreateDirectory(outputDir);
        var detector = new Services.ColorBoxDetector();
        var jsonService = new Services.LabelJsonService();
        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(inputDir)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f).ToList();
        int count = 0, totalBoxes = 0;
        foreach (var file in files)
        {
            using var mask = Cv2.ImRead(file, ImreadModes.Grayscale);
            var boxes = detector.Step3Detect(mask, new DetectionParams());
            var stem = Path.GetFileNameWithoutExtension(file);
            jsonService.WriteResult(Path.Combine(outputDir, stem + ".json"), stem + ".jpg", boxes);
            using var vis = new Mat();
            Cv2.CvtColor(mask, vis, ColorConversionCodes.GRAY2BGR);
            foreach (var box in boxes)
            {
                var pts = box.Corners.Select(c => new OpenCvSharp.Point(c.X, c.Y)).ToArray();
                Cv2.Polylines(vis, new[] { pts }, true, new Scalar(0, 255, 0), 2);
            }
            Cv2.ImWrite(Path.Combine(outputDir, stem + ".jpg"), vis);
            totalBoxes += boxes.Count;
            count++;
            if (count % 50 == 0 || count == 1) Console.WriteLine($"[Step3] {count}/{files.Count} boxes={totalBoxes}");
        }
        Console.WriteLine($"[Step3 Done] {count} images, {totalBoxes} boxes -> {outputDir}");
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

    private static void RunAutoOpt(string inputDir)
    {
        if (!Directory.Exists(inputDir)) { Console.WriteLine($"Input not found: {inputDir}"); return; }
        var detector = new Services.ColorBoxDetector();
        var supported = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        var files = Directory.GetFiles(inputDir)
            .Where(f => supported.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f).Take(10).ToList();
        if (files.Count == 0) { Console.WriteLine("No images found."); return; }

        Console.WriteLine($"Running auto-optimize on {files.Count} images from {inputDir}");
        var dp = new DetectionParams();
        var r0 = dp.ColorRules[0];
        var result = detector.AutoOptimizeThreshold(files, dp, r0.SatMin, r0.ValMin, 255);

        Console.WriteLine($"Optimal: Sat[{result.OptimalSatMin}-{result.OptimalSatMax}] Val[{result.OptimalValMin}-{result.OptimalValMax}] retention={result.OptimalRetention:F3}");
        Console.WriteLine();
        foreach (var p in result.SweepData)
            Console.WriteLine($"  {p.Parameter,6}={p.Value,3}  retention={p.Retention:F3}  signal={p.Signal:F0}");
    }
}
