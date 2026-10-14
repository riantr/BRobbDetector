using System;
using System.Collections.Generic;
using System.Linq;
using BRobbDetector.Models;
using OpenCvSharp;

namespace BRobbDetector.Services;

public class DetectedBox
{
    public List<Point> Corners { get; set; } = new();
    public string Type { get; set; } = string.Empty;
    public string RuleName { get; set; } = string.Empty;
    public double OcrConfidence { get; set; }

    public string PointsText => string.Join(";", Corners.Select(c => $"{c.X},{c.Y}"));
}

public class HoleStats
{
    public double Area;
    public double MinSide;
    public double MaxSide;
    public double FillRatio;
    public double Support;
}

public class ColorBoxDetector
{
    /// <summary>
    /// Purifies the target color in Lab space. Two modes:
    /// "lab": snap core red pixels (±10 around median) to median Lab values
    /// "labBlur": median blur on a* and b* channels to smooth chroma transitions
    /// Lab a* and b* have 3-4x less scattering than HSV hue, making Lab the
    /// optimal space for color normalization.
    /// </summary>
    private static Mat PurifyTargetColor(Mat image, ColorRule rule, string mode = "lab")
    {
        using var lab = new Mat();
        Cv2.CvtColor(image, lab, ColorConversionCodes.BGR2Lab);

        if (mode == "labBlur")
        {
            var ch = new Mat[3];
            Cv2.Split(lab, out ch);
            using var blurredA = new Mat();
            using var blurredB = new Mat();
            Cv2.MedianBlur(ch[1], blurredA, 5);
            Cv2.MedianBlur(ch[2], blurredB, 5);
            ch[1].Dispose();
            ch[2].Dispose();
            ch[1] = blurredA.Clone();
            ch[2] = blurredB.Clone();
            using var merged = new Mat();
            Cv2.Merge(ch, merged);
            foreach (var c in ch) c.Dispose();
            var result = new Mat();
            Cv2.CvtColor(merged, result, ColorConversionCodes.Lab2BGR);
            return result;
        }

        if (mode == "bilateral")
        {
            var result = new Mat();
            Cv2.BilateralFilter(image, result, 7, 50, 7);
            return result;
        }

        if (mode.StartsWith("svd"))
        {
            return PurifySvd(lab, rule, mode);
        }

        using var broadMask = new Mat();
        Cv2.InRange(lab,
            new Scalar(rule.LabLMin, rule.LabAMin, rule.LabBMin),
            new Scalar(rule.LabLMax, rule.LabAMax, rule.LabBMax),
            broadMask);

        int count = Cv2.CountNonZero(broadMask);
        if (count < 10)
            return image.Clone();

        using var maskedLab = new Mat();
        lab.CopyTo(maskedLab, broadMask);

        var channels = new Mat[3];
        Cv2.Split(maskedLab, out channels);
        byte medL = MedianByte(channels[0], broadMask);
        byte medA = MedianByte(channels[1], broadMask);
        byte medB = MedianByte(channels[2], broadMask);
        foreach (var ch in channels) ch.Dispose();

        using var tightMask = new Mat();
        Cv2.InRange(lab,
            new Scalar(Math.Max(0, medL - 10), Math.Max(0, medA - 10), Math.Max(0, medB - 10)),
            new Scalar(Math.Min(255, medL + 10), Math.Min(255, medA + 10), Math.Min(255, medB + 10)),
            tightMask);

        lab.SetTo(new Scalar(medL, medA, medB), tightMask);

        var purified = new Mat();
        Cv2.CvtColor(lab, purified, ColorConversionCodes.Lab2BGR);
        return purified;
    }

    /// <summary>
    /// SVD-based color purification. Applies truncated SVD to a* and b* channels
    /// per connected component of red pixels. A rectangle outline is approximately
    /// rank-4, so keeping top-k singular values (k=5..20) preserves box structure
    /// while removing JPEG block noise (Eckart-Young optimal rank-k approximation).
    /// Mode format: "svd" (default k=10), "svd5", "svd20", etc.
    /// </summary>
    private static Mat PurifySvd(Mat lab, ColorRule rule, string mode)
    {
        int k = 10;
        if (mode.Length > 3 && int.TryParse(mode.AsSpan(3), out int kv))
            k = kv;

        using var broadMask = new Mat();
        Cv2.InRange(lab,
            new Scalar(rule.LabLMin, rule.LabAMin, rule.LabBMin),
            new Scalar(rule.LabLMax, rule.LabAMax, rule.LabBMax),
            broadMask);

        if (Cv2.CountNonZero(broadMask) < 10)
            return lab.Clone();

        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int nComp = Cv2.ConnectedComponentsWithStats(broadMask, labels, stats, centroids, PixelConnectivity.Connectivity8);
        var rects = new List<Rect>();
        for (int i = 1; i < nComp; i++)
        {
            int area = (int)stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            if (area < 50) continue;
            int x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
            int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
            int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
            if (w < 5 || h < 5 || w > 400 || h > 400) continue;
            rects.Add(new Rect(x, y, w, h));
        }

        if (rects.Count == 0)
            return lab.Clone();

        var channels = new Mat[3];
        Cv2.Split(lab, out channels);

        Mat result;
        try
        {
            foreach (var rect in rects)
            {
                SvdDenoiseChannel(channels[1], rect, k);
                SvdDenoiseChannel(channels[2], rect, k);
            }

            using var merged = new Mat();
            Cv2.Merge(channels, merged);
            result = new Mat();
            Cv2.CvtColor(merged, result, ColorConversionCodes.Lab2BGR);
        }
        finally
        {
            foreach (var c in channels) c.Dispose();
        }
        return result;
    }

    private static void SvdDenoiseChannel(Mat channel, Rect rect, int k)
    {
        using var roi = new Mat(channel, rect);
        using var sub = roi.Clone();
        using var f32 = new Mat();
        sub.ConvertTo(f32, MatType.CV_32FC1);

        using var w = new Mat();
        using var u = new Mat();
        using var vt = new Mat();
        Cv2.SVDecomp(f32, w, u, vt);

        int kk = Math.Min(k, Math.Min(u.Cols, vt.Rows));
        if (kk < 1) return;

        using var uCols = new Mat(u, new Rect(0, 0, kk, u.Rows));
        using var uSub = uCols.Clone();
        using var vtRows = new Mat(vt, new Rect(0, 0, vt.Cols, kk));
        using var vtSub = vtRows.Clone();

        using var diagW = Mat.Zeros(kk, kk, MatType.CV_32FC1).ToMat();
        for (int i = 0; i < kk; i++)
            diagW.At<float>(i, i) = w.At<float>(i);

        using var temp = new Mat();
        Cv2.Gemm(uSub, diagW, 1, new Mat(), 0, temp);

        using var reconF = new Mat();
        Cv2.Gemm(temp, vtSub, 1, new Mat(), 0, reconF);

        using var reconU8 = new Mat();
        reconF.ConvertTo(reconU8, MatType.CV_8UC1);

        using var roiBack = new Mat(channel, rect);
        reconU8.CopyTo(roiBack);
    }

    /// <summary>
    /// Post-binarization rank-2 projection cleaning (SVD of a binary rectangle
    /// outline reduces to row/column projections). For each connected component,
    /// keeps only pixels on high-support rows/columns, removing noise pixels.
    /// Mode format: "svdPost" (default mult=0.3), "svdPost2" (0.2), "svdPost4" (0.4).
    /// </summary>
    private static Mat SvdCleanBinaryMask(Mat mask, string mode)
    {
        double thrMult = 0.3;
        if (mode.Length > 7 && int.TryParse(mode.AsSpan(7), out int tv))
            thrMult = tv / 10.0;

        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int nComp = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);

        var result = mask.Clone();
        for (int i = 1; i < nComp; i++)
        {
            int area = (int)stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            if (area < 50) continue;
            int x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
            int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
            int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
            if (w < 5 || h < 5 || w > 300 || h > 300) continue;

            SvdCleanBinaryRegion(result, new Rect(x, y, w, h), thrMult);
        }
        return result;
    }

    private static void SvdCleanBinaryRegion(Mat mask, Rect rect, double thrMult)
    {
        using var roi = new Mat(mask, rect);
        using var sub = roi.Clone();

        int h = sub.Rows, w = sub.Cols;
        var subData = new byte[h * w];
        sub.GetArray(out subData);

        int[] rowSum = new int[h];
        int[] colSum = new int[w];
        int total = 0;
        for (int y = 0; y < h; y++)
        {
            int offset = y * w;
            for (int x = 0; x < w; x++)
            {
                if (subData[offset + x] > 0)
                {
                    rowSum[y]++;
                    colSum[x]++;
                    total++;
                }
            }
        }
        if (total == 0) return;

        double rowMean = (double)total / h;
        double colMean = (double)total / w;
        double rowThr = Math.Max(2, rowMean * thrMult);
        double colThr = Math.Max(2, colMean * thrMult);

        var cleanedData = new byte[h * w];
        for (int y = 0; y < h; y++)
        {
            bool rowEdge = rowSum[y] >= rowThr;
            int offset = y * w;
            for (int x = 0; x < w; x++)
            {
                if (subData[offset + x] > 0 && (rowEdge || colSum[x] >= colThr))
                    cleanedData[offset + x] = 255;
            }
        }

        using var cleaned = new Mat(h, w, MatType.CV_8UC1);
        cleaned.SetArray(cleanedData);
        cleaned.CopyTo(roi);
    }

    private static byte MedianByte(Mat channel, Mat mask)
    {
        int[] hist = new int[256];
        int h = channel.Rows, w = channel.Cols;
        int total = 0;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (mask.At<byte>(y, x) > 0)
                {
                    hist[channel.At<byte>(y, x)]++;
                    total++;
                }
            }
        }
        if (total == 0) return 0;
        int half = total / 2;
        int cum = 0;
        for (int i = 0; i < 256; i++)
        {
            cum += hist[i];
            if (cum >= half)
                return (byte)i;
        }
        return 0;
    }

    public List<DetectedBox> Detect(Mat image, DetectionParams? parameters = null)
    {
        parameters ??= new DetectionParams();
        var results = new List<DetectedBox>();

        if (image.Empty())
            return results;

        if (parameters.DetectionMethod == "yolo")
            return DetectWithYolo(image);

        if (parameters.DetectionMethod == "cvhybrid")
            return CvHybridDetector.Detect(image);

        Mat working = image;
        Mat? purified = null;
        if (parameters.PurifyMode != "none" && !parameters.PurifyMode.StartsWith("svdPost") && parameters.ColorRules.Count > 0)
        {
            purified = PurifyTargetColor(image, parameters.ColorRules[0], parameters.PurifyMode);
            working = purified;
        }
        try
        {
        using var hsv = new Mat();
        Cv2.CvtColor(working, hsv, ColorConversionCodes.BGR2HSV);

        Mat? lab = null, rgb = null;
        try
        {
            if (parameters.ColorSpace.Contains("Lab"))
            {
                lab = new Mat();
                Cv2.CvtColor(working, lab, ColorConversionCodes.BGR2Lab);
            }
            if (parameters.ColorSpace.Contains("Rgb"))
                rgb = working.Clone();

            foreach (var rule in parameters.ColorRules)
            {
                var boxes = DetectByRule(hsv, working, rule, parameters, working.Width, working.Height, lab, rgb);
                results.AddRange(boxes);
            }
        }
        finally
        {
            lab?.Dispose();
            rgb?.Dispose();
        }
        }
        finally
        {
            purified?.Dispose();
        }

        return results;
    }

    public List<DetectedBox> DetectFile(string imagePath, DetectionParams? parameters = null)
    {
        using var image = Cv2.ImRead(imagePath);
        return Detect(image, parameters);
    }

    private static YoloObbDetector? _yoloDetector;
    private static readonly object _yoloLock = new();

    /// <summary>
    /// YOLOv8s-OBB end-to-end inference (LOOP-6). Lazily loads the ONNX model
    /// once and reuses it. Selected via DetectionMethod == "yolo".
    /// </summary>
    public static List<DetectedBox> DetectWithYolo(Mat image)
    {
        if (_yoloDetector == null)
        {
            lock (_yoloLock)
            {
                if (_yoloDetector == null)
                {
                    var path = YoloObbDetector.ResolveDefaultModelPath();
                    _yoloDetector = new YoloObbDetector(path);
                }
            }
        }
        return _yoloDetector.Detect(image);
    }

    /// <summary>
    /// Step 0: Binarize image by target color. Returns a binary mask where
    /// white = target color, black = background.
    /// </summary>
    public Mat Step0Binarize(Mat image, DetectionParams? parameters = null)
    {
        parameters ??= new DetectionParams();
        var result = new Mat(image.Size(), MatType.CV_8UC1, Scalar.Black);
        if (image.Empty()) return result;

        Mat working = image;
        Mat? purified = null;
        if (parameters.PurifyMode != "none" && !parameters.PurifyMode.StartsWith("svdPost") && parameters.ColorRules.Count > 0)
        {
            purified = PurifyTargetColor(image, parameters.ColorRules[0], parameters.PurifyMode);
            working = purified;
        }
        try
        {
        using var hsv = new Mat();
        Cv2.CvtColor(working, hsv, ColorConversionCodes.BGR2HSV);

        Mat? lab = null, rgb = null;
        try
        {
            if (parameters.ColorSpace.Contains("Lab"))
            {
                lab = new Mat();
                Cv2.CvtColor(working, lab, ColorConversionCodes.BGR2Lab);
            }
            if (parameters.ColorSpace.Contains("Rgb"))
            {
                rgb = working.Clone();
            }

            foreach (var rule in parameters.ColorRules)
            {
                using var ruleMask = BuildRuleMask(hsv, rule);
                using var csMask = ApplyColorSpaceFilter(working.Size(), rule, lab, rgb);
                using var combined = new Mat();
                Cv2.BitwiseAnd(ruleMask, csMask, combined);
                Cv2.BitwiseOr(result, combined, result);
            }
        }
        finally
        {
            lab?.Dispose();
            rgb?.Dispose();
        }
        }
        finally
        {
            purified?.Dispose();
        }
        return result;
    }

    /// <summary>
    /// Step 1: Filter binary mask — remove text glyphs and non-line regions.
    /// Input is a binary mask from Step 0, output is a cleaned binary mask.
    /// </summary>
    public Mat Step1Filter(Mat binaryMask, DetectionParams? parameters = null)
    {
        parameters ??= new DetectionParams();
        using var noText = RemoveText(binaryMask, parameters);
        using var opened = LineOpen(noText, parameters.LineOpenLength, parameters.LineOpenStep);
        return LineClose(opened, parameters.LineOpenLength, parameters.LineOpenStep);
    }

    /// <summary>
    /// Step 2: Preserve holes — morphological close to strengthen hole enclosures.
    /// Input is a filtered binary mask from Step 1, output is a cleaned mask
    /// with hole structures preserved.
    /// </summary>
    public Mat Step2PreserveHoles(Mat binaryMask, DetectionParams? parameters = null)
    {
        parameters ??= new DetectionParams();
        return ApplyClose(binaryMask, parameters);
    }

    private static Mat ApplyClose(Mat mask, DetectionParams p)
    {
        var result = new Mat();
        switch (p.CloseMode)
        {
            case "iter1":
                using (var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3)))
                    Cv2.MorphologyEx(mask, result, MorphTypes.Close, k, iterations: 1);
                break;
            case "directional":
                using (var kh = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 1)))
                using (var kv = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 5)))
                {
                    var tmp = new Mat();
                    Cv2.MorphologyEx(mask, tmp, MorphTypes.Close, kh, iterations: 1);
                    Cv2.MorphologyEx(tmp, result, MorphTypes.Close, kv, iterations: 1);
                    tmp.Dispose();
                }
                break;
            case "directional2":
                using (var kh = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(7, 1)))
                using (var kv = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 7)))
                {
                    var tmp = new Mat();
                    Cv2.MorphologyEx(mask, tmp, MorphTypes.Close, kh, iterations: 1);
                    Cv2.MorphologyEx(tmp, result, MorphTypes.Close, kv, iterations: 1);
                    tmp.Dispose();
                }
                break;
            default:
                using (var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3)))
                    Cv2.MorphologyEx(mask, result, MorphTypes.Close, kernel, iterations: p.CloseIterations);
                break;
        }
        return result;
    }

    /// <summary>
    /// Step 3: Hole detection from filtered mask. Returns detected boxes.
    /// Input is a preserved binary mask from Step 2.
    /// </summary>
    public List<DetectedBox> Step3Detect(Mat filteredMask, DetectionParams? parameters = null)
    {
        parameters ??= new DetectionParams();
        int imgWidth = filteredMask.Cols, imgHeight = filteredMask.Rows;

        var rects = DetectHoles(filteredMask, parameters, imgWidth, imgHeight);
        rects = ValidateEdges(rects, filteredMask, parameters);
        rects = Nms(rects, parameters.NmsThreshold);
        rects = DedupContained(rects, parameters.DedupContainThreshold);
        rects = rects.Select(r => RefineBox(r, filteredMask)).ToList();

        using var refineKernelS = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(3, 3));
        using var maskDS = new Mat();
        Cv2.Dilate(filteredMask, maskDS, refineKernelS, iterations: 3);
        rects = rects.Select(r => BoxRefinement.RefineWithPreserveSup(r, maskDS)).ToList();

        var type = parameters.ColorRules.Count > 0 ? parameters.ColorRules[0].Type : "";
        var ruleName = parameters.ColorRules.Count > 0 ? parameters.ColorRules[0].Name : "";
        return rects.Select(r => RectToDetectedBox(r, imgWidth, imgHeight, type, ruleName)).ToList();
    }

    /// <summary>
    /// Auto-optimize HSV bounds (SatMin, ValMin, SatMax, ValMax) using knee-point
    /// detection on the retention curve. Retention = |Step1| / |Step0| since Step1 ⊆ Step0.
    /// Sweeps ±sweepRange around the picked color center, finds where retention drops
    /// sharply (knee), and picks the most aggressive value before the cliff.
    /// </summary>
    public AutoOptimizeResult AutoOptimizeThreshold(List<string> imagePaths, DetectionParams baseParams,
        int centerSat, int centerVal, int sweepRange = 25, double kneeRatio = 0.9)
    {
        var result = new AutoOptimizeResult();
        var images = new List<(Mat hsv, Mat? lab, Mat? rgb)>();
        try
        {
            foreach (var path in imagePaths)
            {
                using var img = Cv2.ImRead(path);
                if (img.Empty()) continue;
                var hsv = new Mat();
                Cv2.CvtColor(img, hsv, ColorConversionCodes.BGR2HSV);
                Mat? lab = null, rgb = null;
                if (baseParams.ColorSpace.Contains("Lab"))
                {
                    lab = new Mat();
                    Cv2.CvtColor(img, lab, ColorConversionCodes.BGR2Lab);
                }
                if (baseParams.ColorSpace.Contains("Rgb"))
                    rgb = img.Clone();
                images.Add((hsv, lab, rgb));
            }
            if (images.Count == 0) return result;

            var rule0 = baseParams.ColorRules[0];
            int satMin = rule0.SatMin, satMax = rule0.SatMax;
            int valMin = rule0.ValMin, valMax = rule0.ValMax;

            int satLo = Math.Max(0, centerSat - sweepRange);
            int satHi = Math.Min(255, centerSat + sweepRange);
            int valLo = Math.Max(0, centerVal - sweepRange);
            int valHi = Math.Min(255, centerVal + sweepRange);
            int step = 5;

            // 1. SatMin: sweep [satLo→satHi], pick highest value on the retention plateau
            var satMinPts = SweepParam(images, baseParams, "SatMin", satLo, satHi, step,
                v => (v, satMax, valMin, valMax));
            result.SweepData.AddRange(satMinPts);
            satMin = FindKneeHigh(satMinPts, kneeRatio);

            // 2. ValMin: sweep [valLo→valHi], pick highest value on the retention plateau
            var valMinPts = SweepParam(images, baseParams, "ValMin", valLo, valHi, step,
                v => (satMin, satMax, v, valMax));
            result.SweepData.AddRange(valMinPts);
            valMin = FindKneeHigh(valMinPts, kneeRatio);

            // 3. SatMax: sweep [satHi→satLo], pick lowest value on the retention plateau
            var satMaxPts = SweepParam(images, baseParams, "SatMax", satHi, satLo, -step,
                v => (satMin, v, valMin, valMax));
            result.SweepData.AddRange(satMaxPts);
            satMax = FindKneeLow(satMaxPts, kneeRatio);

            // 4. ValMax: sweep [valHi→valLo], pick lowest value on the retention plateau
            var valMaxPts = SweepParam(images, baseParams, "ValMax", valHi, valLo, -step,
                v => (satMin, satMax, valMin, v));
            result.SweepData.AddRange(valMaxPts);
            valMax = FindKneeLow(valMaxPts, kneeRatio);

            var (finalRet, _) = EvalRetention(images, baseParams, satMin, satMax, valMin, valMax);
            result.OptimalSatMin = satMin;
            result.OptimalSatMax = satMax;
            result.OptimalValMin = valMin;
            result.OptimalValMax = valMax;
            result.OptimalRetention = finalRet;
        }
        finally
        {
            foreach (var (hsv, lab, rgb) in images) { hsv.Dispose(); lab?.Dispose(); rgb?.Dispose(); }
        }
        return result;
    }

    private List<AutoOptimizeSweepPoint> SweepParam(List<(Mat hsv, Mat? lab, Mat? rgb)> images, DetectionParams baseParams,
        string param, int from, int to, int step,
        Func<int, (int satMin, int satMax, int valMin, int valMax)> getBounds)
    {
        var pts = new List<AutoOptimizeSweepPoint>();
        for (int v = from; step > 0 ? v <= to : v >= to; v += step)
        {
            var (sMin, sMax, vMin, vMax) = getBounds(v);
            var (ret, sig) = EvalRetention(images, baseParams, sMin, sMax, vMin, vMax);
            pts.Add(new AutoOptimizeSweepPoint { Parameter = param, Value = v, Retention = ret, Signal = sig });
        }
        return pts;
    }

    private static int FindKneeHigh(List<AutoOptimizeSweepPoint> pts, double ratio)
    {
        if (pts.Count == 0) return 0;
        double maxRet = pts.Max(p => p.Retention);
        if (maxRet <= 0) return pts[0].Value;
        double threshold = maxRet * ratio;
        return pts.Where(p => p.Retention >= threshold).Max(p => p.Value);
    }

    private static int FindKneeLow(List<AutoOptimizeSweepPoint> pts, double ratio)
    {
        if (pts.Count == 0) return 255;
        double maxRet = pts.Max(p => p.Retention);
        if (maxRet <= 0) return pts[0].Value;
        double threshold = maxRet * ratio;
        return pts.Where(p => p.Retention >= threshold).Min(p => p.Value);
    }

    private (double retention, double signal) EvalRetention(List<(Mat hsv, Mat? lab, Mat? rgb)> images, DetectionParams baseParams,
        int satMin, int satMax, int valMin, int valMax, double maxFraction = 0.25)
    {
        double totalRetention = 0, totalSignal = 0;
        int validCount = 0;
        foreach (var (hsv, lab, rgb) in images)
        {
            using var mask = new Mat(hsv.Size(), MatType.CV_8UC1, Scalar.Black);
            foreach (var rule in baseParams.ColorRules)
            {
                var adjusted = new ColorRule
                {
                    HueLow = rule.HueLow, HueHigh = rule.HueHigh,
                    HueLow2 = rule.HueLow2, HueHigh2 = rule.HueHigh2,
                    SatMin = satMin, SatMax = satMax,
                    ValMin = valMin, ValMax = valMax,
                    UseSecondRange = rule.UseSecondRange
                };
                using var ruleMask = BuildRuleMask(hsv, adjusted);
                using var csMask = ApplyColorSpaceFilter(hsv.Size(), rule, lab, rgb);
                using var combined = new Mat();
                Cv2.BitwiseAnd(ruleMask, csMask, combined);
                Cv2.BitwiseOr(mask, combined, mask);
            }
            int count0 = Cv2.CountNonZero(mask);
            if (count0 == 0) continue;
            if (count0 > maxFraction * hsv.Rows * hsv.Cols) continue;
            using var filtered = Step1Filter(mask, baseParams);
            int count1 = Cv2.CountNonZero(filtered);
            totalRetention += (double)count1 / count0;
            totalSignal += count1;
            validCount++;
        }
        if (validCount == 0) return (0, 0);
        return (totalRetention / validCount, totalSignal / validCount);
    }

    public List<HoleStats> CollectHoleStats(string imagePath, DetectionParams parameters)
    {
        using var image = Cv2.ImRead(imagePath);
        if (image.Empty()) return new List<HoleStats>();
        using var hsv = new Mat();
        Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);

        Mat? lab = null, rgb = null;
        try
        {
            if (parameters.ColorSpace.Contains("Lab"))
            {
                lab = new Mat();
                Cv2.CvtColor(image, lab, ColorConversionCodes.BGR2Lab);
            }
            if (parameters.ColorSpace.Contains("Rgb"))
                rgb = image.Clone();

            var allStats = new List<HoleStats>();
            foreach (var rule in parameters.ColorRules)
            {
                using var hsvMask = BuildRuleMask(hsv, rule);
                using var csMask = ApplyColorSpaceFilter(image.Size(), rule, lab, rgb);
                using var mask = new Mat();
                Cv2.BitwiseAnd(hsvMask, csMask, mask);
                using var noText = RemoveText(mask, parameters);
                using var opened = LineOpen(noText, parameters.LineOpenLength, parameters.LineOpenStep);
                var stats = CollectHoleStatsFromMask(opened, parameters, image.Width, image.Height);
                allStats.AddRange(stats);
            }
            return allStats;
        }
        finally
        {
            lab?.Dispose();
            rgb?.Dispose();
        }
    }

    private static List<HoleStats> CollectHoleStatsFromMask(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        var stats = new List<HoleStats>();
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);

        using var closed = ApplyClose(cleaned, p);

        Cv2.FindContours(closed, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);
        if (hierarchy == null || hierarchy.Length == 0) return stats;

        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent < 0) continue;

            foreach (var holeContour in SplitIfMerged(contours[i], p))
            {
                double area = Cv2.ContourArea(holeContour);
                if (area < 50 || area > maxPixels) continue;

                var rect = Cv2.MinAreaRect(holeContour);
                double rw = rect.Size.Width;
                double rh = rect.Size.Height;
                double minSide = Math.Min(rw, rh);
                double maxSide = Math.Max(rw, rh);
                if (minSide < 3) continue;

                double rectArea = rw * rh;
                if (rectArea <= 0) continue;
                double fillRatio = area / rectArea;

                var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
                double support = PerimeterSupport(expanded, cleaned, 2);

                stats.Add(new HoleStats { Area = area, MinSide = minSide, MaxSide = maxSide, FillRatio = fillRatio, Support = support });
            }
        }
        return stats;
    }

    private List<DetectedBox> DetectByRule(Mat hsv, Mat image, ColorRule rule, DetectionParams p, int imgWidth, int imgHeight, Mat? lab = null, Mat? rgb = null)
    {
        using var hsvMask = BuildRuleMask(hsv, rule);
        using var csMask = ApplyColorSpaceFilter(image.Size(), rule, lab, rgb);
        using var mask = new Mat();
        Cv2.BitwiseAnd(hsvMask, csMask, mask);

        var rects = DetectPass(mask, p, imgWidth, imgHeight, mask);
        if (rects.Count == 0)
            return new List<DetectedBox>();

        using var cleaned = ObbRemoveTextComponents(mask, rects, p);
        var rects2 = DetectPass(cleaned, p, imgWidth, imgHeight, mask);

        var allRects = new List<RotatedRect>(rects);
        allRects.AddRange(rects2);
        allRects = Nms(allRects, p.NmsThreshold);
        allRects = DedupContained(allRects, p.DedupContainThreshold);
        allRects = allRects.Select(r => RefineBox(r, mask)).ToList();

        using var refineKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(3, 3));
        using var maskD = new Mat();
        Cv2.Dilate(mask, maskD, refineKernel, iterations: 3);
        allRects = allRects.Select(r => BoxRefinement.RefineWithPreserveSup(r, maskD)).ToList();

        return allRects.Select(r => RectToDetectedBox(r, imgWidth, imgHeight, rule.Type, rule.Name)).ToList();
    }

    /// <summary>
    /// Refines a detected OBB so each edge aligns with the median position of
    /// target-color pixels along that edge. For each edge, collects red pixels
    /// in a perpendicular band, computes their median offset, shifts the edge
    /// to that offset, then intersects the 4 shifted lines for new corners.
    /// </summary>
    private static RotatedRect RefineBox(RotatedRect rect, Mat mask, int band = 15)
    {
        var pts = Cv2.BoxPoints(rect);
        int h = mask.Rows, w = mask.Cols;

        var shiftedLines = new (Point2d point, Point2d dir)[4];
        for (int i = 0; i < 4; i++)
        {
            var p1 = pts[i];
            var p2 = pts[(i + 1) % 4];
            double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
            double edgeLen = Math.Sqrt(dx * dx + dy * dy);
            if (edgeLen < 1)
            {
                shiftedLines[i] = (new Point2d(p1.X, p1.Y), new Point2d(dx, dy));
                continue;
            }

            double tx = dx / edgeLen, ty = dy / edgeLen;
            double nx = -ty, ny = tx;

            var offsets = new List<double>();
            int nSamples = (int)edgeLen;
            for (int t = 0; t <= nSamples; t++)
            {
                double frac = (double)t / nSamples;
                double cx = p1.X + dx * frac;
                double cy = p1.Y + dy * frac;
                for (int d = -band; d <= band; d++)
                {
                    int px = (int)Math.Round(cx + nx * d);
                    int py = (int)Math.Round(cy + ny * d);
                    if (px >= 0 && px < w && py >= 0 && py < h && mask.At<byte>(py, px) > 0)
                        offsets.Add((double)d);
                }
            }

            double shift = offsets.Count >= 10 ? Median(offsets) : 0;
            shiftedLines[i] = (new Point2d(p1.X + nx * shift, p1.Y + ny * shift), new Point2d(dx, dy));
        }

        var newCorners = new Point2d[4];
        for (int i = 0; i < 4; i++)
        {
            var (c1, d1) = shiftedLines[i];
            var (c2, d2) = shiftedLines[(i + 1) % 4];
            double det = d1.X * (-d2.Y) - (-d2.X) * d1.Y;
            if (Math.Abs(det) < 1e-6)
            {
                newCorners[i] = new Point2d(pts[i].X, pts[i].Y);
                continue;
            }
            double t1 = ((c2.X - c1.X) * (-d2.Y) - (-d2.X) * (c2.Y - c1.Y)) / det;
            newCorners[i] = new Point2d(c1.X + t1 * d1.X, c1.Y + t1 * d1.Y);
        }

        var contour = newCorners.Select(c => new Point((int)Math.Round(c.X), (int)Math.Round(c.Y))).ToArray();
        return Cv2.MinAreaRect(contour);
    }

    /// <summary>
    /// Grows a strip-like rect by scanning each edge outward to find the farthest
    /// red pixel cluster. For each edge, samples lines parallel to the edge at
    /// increasing distances, finds the farthest distance with ≥50% pixel support,
    /// and shifts the edge there. Reconstructs the rect from 4 shifted lines.
    /// </summary>
    private static RotatedRect GrowBox(RotatedRect rect, Mat mask, int maxScan, double supportThreshold = 0.30)
    {
        var pts = Cv2.BoxPoints(rect);
        int h = mask.Rows, w = mask.Cols;
        double rw = rect.Size.Width, rh = rect.Size.Height;
        double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
        var shiftedLines = new (Point2d point, Point2d dir)[4];

        for (int i = 0; i < 4; i++)
        {
            var p1 = pts[i];
            var p2 = pts[(i + 1) % 4];
            double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
            double edgeLen = Math.Sqrt(dx * dx + dy * dy);
            if (edgeLen < 1)
            {
                shiftedLines[i] = (new Point2d(p1.X, p1.Y), new Point2d(dx, dy));
                continue;
            }

            double tx = dx / edgeLen, ty = dy / edgeLen;
            double nx = -ty, ny = tx;

            // Only grow LONG edges (perpendicular to short axis)
            int bestDist = 0;
            if (edgeLen > minSide * 1.5)
            {
                int nSamples = (int)edgeLen;

                int maxScanSafe = Math.Min(maxScan, Math.Max(w, h));
                var hitCounts = new int[maxScanSafe + 1];
                int maxHit = 0;

                for (int d = 0; d <= maxScanSafe; d++)
                {
                    int hit = 0;
                    for (int t = 0; t <= nSamples; t++)
                    {
                        double frac = (double)t / nSamples;
                        double cx = p1.X + dx * frac + nx * d;
                        double cy = p1.Y + dy * frac + ny * d;
                        int px = (int)Math.Round(cx), py = (int)Math.Round(cy);
                        if (px >= 0 && px < w && py >= 0 && py < h && mask.At<byte>(py, px) > 0)
                            hit++;
                    }
                    hitCounts[d] = hit;
                    if (hit > maxHit)
                        maxHit = hit;
                }

                int adaptiveThreshold = Math.Max(3, maxHit / 4);
                int growthLimit = (int)(minSide * 3);
                for (int d = 1; d <= maxScanSafe && d <= growthLimit; d++)
                {
                    if (hitCounts[d] >= adaptiveThreshold)
                        bestDist = d;
                }

                {
                    int minRunLen = Math.Max(5, nSamples / 7);
                    int startDist = Math.Max(1, (int)(minSide * 0.5));
                    for (int d = startDist; d <= maxScanSafe && d <= growthLimit; d++)
                    {
                        int maxRun = 0, curRun = 0, gap = 0;
                        for (int t = 0; t <= nSamples; t++)
                        {
                            double frac = (double)t / nSamples;
                            double cx = p1.X + dx * frac + nx * d;
                            double cy = p1.Y + dy * frac + ny * d;
                            int px = (int)Math.Round(cx), py = (int)Math.Round(cy);
                            bool isRed = px >= 0 && px < w && py >= 0 && py < h && mask.At<byte>(py, px) > 0;
                            if (isRed)
                            {
                                curRun++;
                                gap = 0;
                            }
                            else
                            {
                                gap++;
                                if (gap > 1)
                                {
                                    if (curRun > maxRun) maxRun = curRun;
                                    curRun = 0;
                                }
                            }
                        }
                        if (curRun > maxRun) maxRun = curRun;
                        if (maxRun >= minRunLen && d > bestDist)
                        {
                            bestDist = d;
                            break;
                        }
                    }
                }
            }

            shiftedLines[i] = (new Point2d(p1.X + nx * bestDist, p1.Y + ny * bestDist), new Point2d(dx, dy));
        }

        var newCorners = new Point2d[4];
        for (int i = 0; i < 4; i++)
        {
            var (c1, d1) = shiftedLines[i];
            var (c2, d2) = shiftedLines[(i + 1) % 4];
            double det = d1.X * (-d2.Y) - (-d2.X) * d1.Y;
            if (Math.Abs(det) < 1e-6)
            {
                newCorners[i] = new Point2d(pts[i].X, pts[i].Y);
                continue;
            }
            double t1 = ((c2.X - c1.X) * (-d2.Y) - (-d2.X) * (c2.Y - c1.Y)) / det;
            newCorners[i] = new Point2d(c1.X + t1 * d1.X, c1.Y + t1 * d1.Y);
        }

        var contour2 = newCorners.Select(c => new Point((int)Math.Round(c.X), (int)Math.Round(c.Y))).ToArray();
        return Cv2.MinAreaRect(contour2);
    }

    private static double Median(List<double> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        int n = sorted.Count;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
    }


    /// <summary>
    /// Canonical detection pass shared by <see cref="DetectByRule"/> and the
    /// Step0..Step3 visualization/CLI pipeline:
    ///   denoise (RemoveText + LineOpen + LineClose) -> preserve holes (ApplyClose)
    ///   -> detect rects -> validate/filter/NMS.
    /// </summary>
    private static List<RotatedRect> DetectPass(Mat mask, DetectionParams p, int imgWidth, int imgHeight, Mat? validationMask = null)
    {
        Mat input = mask;
        Mat? svdCleaned = null;
        if (p.PurifyMode != null && p.PurifyMode.StartsWith("svdPost"))
        {
            svdCleaned = SvdCleanBinaryMask(mask, p.PurifyMode);
            input = svdCleaned;
        }
        try
        {
            using var preserved = DenoiseAndPreserve(input, p);
            return DetectFromPreserved(preserved, p, imgWidth, imgHeight, validationMask ?? mask);
        }
        finally
        {
            svdCleaned?.Dispose();
        }
    }

    /// <summary>
    /// Step 1 + Step 2 of the canonical pipeline: RemoveText -> LineOpen ->
    /// LineClose -> ApplyClose. Used by both the detector and the step buttons.
    /// </summary>
    private static Mat DenoiseAndPreserve(Mat mask, DetectionParams p)
    {
        using var noText = RemoveText(mask, p);
        using var opened = LineOpen(noText, p.LineOpenLength, p.LineOpenStep);
        using var stitched = StitchLineFragments(opened);
        return ApplyClose(stitched, p);
    }

    /// <summary>
    /// Bridges gaps in broken outline fragments by applying a directional close
    /// (21×1 + 1×21) only to line-like components (thin strokes, fill ratio &lt; 0.3).
    /// This repairs broken box outlines without filling thin-box holes.
    /// </summary>
    private static Mat StitchLineFragments(Mat mask)
    {
        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);
        int nLabels = stats.Rows;
        if (nLabels <= 2) return mask.Clone();

        using var lineMask = new Mat(mask.Size(), MatType.CV_8UC1, Scalar.Black);
        for (int i = 1; i < nLabels; i++)
        {
            int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
            int bboxArea = w * h;
            if (bboxArea == 0 || area < 20) continue;
            double fillRatio = (double)area / bboxArea;
            if (fillRatio < 0.3)
            {
                using var componentMask = new Mat();
                Cv2.Compare(labels, new Scalar(i), componentMask, CmpType.EQ);
                Cv2.BitwiseOr(lineMask, componentMask, lineMask);
            }
        }

        using var kh = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(7, 1));
        using var kv = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(1, 7));
        using var closedH = new Mat();
        using var closedV = new Mat();
        Cv2.MorphologyEx(lineMask, closedH, MorphTypes.Close, kh);
        Cv2.MorphologyEx(closedH, closedV, MorphTypes.Close, kv);

        var result = new Mat();
        Cv2.BitwiseOr(mask, closedV, result);
        return result;
    }


    /// <summary>
    /// Step 3 of the canonical pipeline: rectangle detection from a mask that has
    /// already been denoised and hole-preserved. No further morphological close.
    /// </summary>
    private static List<RotatedRect> DetectFromPreserved(Mat preserved, DetectionParams p, int imgWidth, int imgHeight, Mat validationMask)
    {
        var rects = DetectByMethod(preserved, p, imgWidth, imgHeight);
        rects = ValidateEdges(rects, validationMask, p);
        rects = ApplyFilter(rects, validationMask, p);
        rects = Nms(rects, p.NmsThreshold);
        rects = DedupContained(rects, p.DedupContainThreshold);

        return rects;
    }

    private static List<RotatedRect> ApplyFilter(List<RotatedRect> rects, Mat mask, DetectionParams p)
    {
        return p.FilterMethod switch
        {
            "none" => rects,
            "strokeWidth" => FilterByStrokeWidth(rects, mask),
            "rightAngle" => FilterByRightAngles(rects, mask),
            "straightness" => FilterByEdgeStraightness(rects, mask),
            "symmetry" => FilterBySymmetry(rects),
            "dtWidth" => FilterByDistanceTransformWidth(rects, mask),
            "all" => FilterBySymmetry(FilterByRightAngles(FilterByStrokeWidth(rects, mask), mask)),
            _ => rects
        };
    }

    /// <summary>
    /// Filter 1: Stroke width consistency. For each OBB edge, sample perpendicular
    /// cross-sections at regular intervals, count red pixels (= local stroke width).
    /// Reject if width variance along an edge is high or width differs across edges.
    /// </summary>
    private static List<RotatedRect> FilterByStrokeWidth(List<RotatedRect> rects, Mat mask, int band = 15, int nSamples = 20, double maxCv = 1.2)
    {
        var keep = new List<RotatedRect>();
        int h = mask.Rows, w = mask.Cols;
        foreach (var rect in rects)
        {
            var pts = Cv2.BoxPoints(rect);
            var edgeWidths = new List<double>();
            bool ok = true;
            for (int edge = 0; edge < 4 && ok; edge++)
            {
                var p1 = pts[edge];
                var p2 = pts[(edge + 1) % 4];
                double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
                double len = Math.Sqrt(dx * dx + dy * dy);
                if (len < 5) { ok = false; break; }
                double tx = dx / len, ty = dy / len;
                double nx = -ty, ny = tx;
                var widths = new List<double>();
                for (int s = 0; s <= nSamples; s++)
                {
                    double frac = (double)s / nSamples;
                    double cx = p1.X + dx * frac;
                    double cy = p1.Y + dy * frac;
                    int count = 0;
                    for (int d = -band; d <= band; d++)
                    {
                        int px = (int)Math.Round(cx + nx * d);
                        int py = (int)Math.Round(cy + ny * d);
                        if (px >= 0 && px < w && py >= 0 && py < h && mask.At<byte>(py, px) > 0)
                            count++;
                    }
                    widths.Add(count);
                }
                if (widths.Count == 0) { ok = false; break; }
                double mean = widths.Average();
                if (mean < 1) { ok = false; break; }
                double std = Math.Sqrt(widths.Sum(v => (v - mean) * (v - mean)) / widths.Count);
                if (std / mean > maxCv) { ok = false; break; }
                edgeWidths.Add(mean);
            }
            if (ok && edgeWidths.Count == 4)
            {
                double overallMean = edgeWidths.Average();
                if (overallMean >= 1)
                {
                    double overallStd = Math.Sqrt(edgeWidths.Sum(v => (v - overallMean) * (v - overallMean)) / 4);
                    if (overallStd / overallMean <= maxCv)
                        keep.Add(rect);
                }
            }
        }
        return keep;
    }

    /// <summary>
    /// Filter 2: 90° angle validation. For each OBB corner, collect red pixels in
    /// two perpendicular bands along the adjacent edges, fit lines, compute angle.
    /// Reject if any corner angle deviates >15° from 90°.
    /// </summary>
    private static List<RotatedRect> FilterByRightAngles(List<RotatedRect> rects, Mat mask, double maxDeviation = 25.0, int cornerRadius = 25)
    {
        var keep = new List<RotatedRect>();
        int h = mask.Rows, w = mask.Cols;
        foreach (var rect in rects)
        {
            var pts = Cv2.BoxPoints(rect);
            bool ok = true;
            for (int c = 0; c < 4 && ok; c++)
            {
                var corner = pts[c];
                var prev = pts[(c + 3) % 4];
                var next = pts[(c + 1) % 4];
                var dir1 = new Point2d(prev.X - corner.X, prev.Y - corner.Y);
                var dir2 = new Point2d(next.X - corner.X, next.Y - corner.Y);
                double l1 = Math.Sqrt(dir1.X * dir1.X + dir1.Y * dir1.Y);
                double l2 = Math.Sqrt(dir2.X * dir2.X + dir2.Y * dir2.Y);
                if (l1 < 1 || l2 < 1) { ok = false; break; }
                dir1 = new Point2d(dir1.X / l1, dir1.Y / l1);
                dir2 = new Point2d(dir2.X / l2, dir2.Y / l2);

                var pts1 = new List<Point2d>();
                var pts2 = new List<Point2d>();
                for (int t = 3; t <= cornerRadius; t++)
                {
                    for (int d = -3; d <= 3; d++)
                    {
                        double x1 = corner.X + dir1.X * t + (-dir1.Y) * d;
                        double y1 = corner.Y + dir1.Y * t + dir1.X * d;
                        int px1 = (int)Math.Round(x1), py1 = (int)Math.Round(y1);
                        if (px1 >= 0 && px1 < w && py1 >= 0 && py1 < h && mask.At<byte>(py1, px1) > 0)
                            pts1.Add(new Point2d(x1, y1));

                        double x2 = corner.X + dir2.X * t + (-dir2.Y) * d;
                        double y2 = corner.Y + dir2.Y * t + dir2.X * d;
                        int px2 = (int)Math.Round(x2), py2 = (int)Math.Round(y2);
                        if (px2 >= 0 && px2 < w && py2 >= 0 && py2 < h && mask.At<byte>(py2, px2) > 0)
                            pts2.Add(new Point2d(x2, y2));
                    }
                }
                if (pts1.Count < 5 || pts2.Count < 5) { ok = false; break; }

                var line1 = FitLine(pts1);
                var line2 = FitLine(pts2);
                double angle = Math.Abs(Math.Atan2(line1.dirX * line2.dirY - line1.dirY * line2.dirX, line1.dirX * line2.dirX + line1.dirY * line2.dirY));
                angle = Math.Min(angle, Math.PI - angle);
                double deviation = Math.Abs(angle - Math.PI / 2) * 180.0 / Math.PI;
                if (deviation > maxDeviation) { ok = false; break; }
            }
            if (ok) keep.Add(rect);
        }
        return keep;
    }

    /// <summary>
    /// Filter 3: Edge straightness. For each OBB edge, collect red pixels in a
    /// perpendicular band, fit a line, compute max residual. Reject if any edge
    /// has max residual > 3 pixels (curved/jagged).
    /// </summary>
    private static List<RotatedRect> FilterByEdgeStraightness(List<RotatedRect> rects, Mat mask, int band = 8, double maxResidual = 5.0)
    {
        var keep = new List<RotatedRect>();
        int h = mask.Rows, w = mask.Cols;
        foreach (var rect in rects)
        {
            var pts = Cv2.BoxPoints(rect);
            bool ok = true;
            for (int edge = 0; edge < 4 && ok; edge++)
            {
                var p1 = pts[edge];
                var p2 = pts[(edge + 1) % 4];
                double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
                double len = Math.Sqrt(dx * dx + dy * dy);
                if (len < 5) continue;
                double tx = dx / len, ty = dy / len;
                double nx = -ty, ny = tx;

                var centers = new List<Point2d>();
                int nSamp = Math.Min((int)len, 40);
                for (int s = 0; s <= nSamp; s++)
                {
                    double frac = (double)s / nSamp;
                    double cx = p1.X + dx * frac;
                    double cy = p1.Y + dy * frac;
                    double sumX = 0, sumY = 0;
                    int count = 0;
                    for (int d = -band; d <= band; d++)
                    {
                        int px = (int)Math.Round(cx + nx * d);
                        int py = (int)Math.Round(cy + ny * d);
                        if (px >= 0 && px < w && py >= 0 && py < h && mask.At<byte>(py, px) > 0)
                        {
                            sumX += px;
                            sumY += py;
                            count++;
                        }
                    }
                    if (count > 0)
                        centers.Add(new Point2d(sumX / count, sumY / count));
                }
                if (centers.Count < 5) { ok = false; break; }

                var fit = FitLine(centers);
                double maxR = centers.Max(p => Math.Abs((p.X - fit.point.X) * fit.dirY - (p.Y - fit.point.Y) * fit.dirX));
                if (maxR > maxResidual) { ok = false; break; }
            }
            if (ok) keep.Add(rect);
        }
        return keep;
    }

    /// <summary>
    /// Filter 4: Opposite edge parallelism + equal length. Check edge1∥edge3,
    /// edge2∥edge4, |edge1|≈|edge3|, |edge2|≈|edge4|. Reject asymmetric shapes.
    /// </summary>
    private static List<RotatedRect> FilterBySymmetry(List<RotatedRect> rects, double maxAngleDiff = 5.0, double minLengthRatio = 0.85)
    {
        var keep = new List<RotatedRect>();
        foreach (var rect in rects)
        {
            var pts = Cv2.BoxPoints(rect);
            var edges = new double[4];
            var angles = new double[4];
            for (int i = 0; i < 4; i++)
            {
                double dx = pts[(i + 1) % 4].X - pts[i].X;
                double dy = pts[(i + 1) % 4].Y - pts[i].Y;
                edges[i] = Math.Sqrt(dx * dx + dy * dy);
                angles[i] = Math.Atan2(dy, dx);
            }
            double dAngle13 = Math.Abs(AngleDiff(angles[0], angles[2])) * 180.0 / Math.PI;
            double dAngle24 = Math.Abs(AngleDiff(angles[1], angles[3])) * 180.0 / Math.PI;
            double ratio13 = Math.Min(edges[0], edges[2]) / Math.Max(edges[0], edges[2]);
            double ratio24 = Math.Min(edges[1], edges[3]) / Math.Max(edges[1], edges[3]);
            if (dAngle13 <= maxAngleDiff && dAngle24 <= maxAngleDiff &&
                ratio13 >= minLengthRatio && ratio24 >= minLengthRatio)
                keep.Add(rect);
        }
        return keep;
    }

    /// <summary>
    /// Filter 5: Distance transform stroke width. Compute DT of binary mask,
    /// measure stroke width = 2×max DT value along the OBB centerline.
    /// Filter by width consistency across all 4 edges.
    /// </summary>
    private static List<RotatedRect> FilterByDistanceTransformWidth(List<RotatedRect> rects, Mat mask, double maxCv = 0.8)
    {
        using var dt = new Mat();
        Cv2.DistanceTransform(mask, dt, DistanceTypes.L2, 0);
        int h = mask.Rows, w = mask.Cols;
        var keep = new List<RotatedRect>();
        foreach (var rect in rects)
        {
            var pts = Cv2.BoxPoints(rect);
            var edgeWidths = new List<double>();
            bool ok = true;
            for (int edge = 0; edge < 4 && ok; edge++)
            {
                var p1 = pts[edge];
                var p2 = pts[(edge + 1) % 4];
                double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
                double len = Math.Sqrt(dx * dx + dy * dy);
                if (len < 5) { ok = false; break; }
                double maxDt = 0;
                int nSamp = (int)len;
                for (int s = 0; s <= nSamp; s++)
                {
                    double frac = (double)s / nSamp;
                    int px = (int)Math.Round(p1.X + dx * frac);
                    int py = (int)Math.Round(p1.Y + dy * frac);
                    if (px >= 0 && px < w && py >= 0 && py < h)
                        maxDt = Math.Max(maxDt, dt.At<float>(py, px));
                }
                edgeWidths.Add(maxDt * 2);
            }
            if (ok && edgeWidths.Count == 4)
            {
                double mean = edgeWidths.Average();
                if (mean >= 1)
                {
                    double std = Math.Sqrt(edgeWidths.Sum(v => (v - mean) * (v - mean)) / 4);
                    if (std / mean <= maxCv)
                        keep.Add(rect);
                }
            }
        }
        return keep;
    }

    private static (Point2d point, double dirX, double dirY) FitLine(List<Point2d> pts)
    {
        double mx = pts.Average(p => p.X), my = pts.Average(p => p.Y);
        double sxx = pts.Sum(p => (p.X - mx) * (p.X - mx));
        double syy = pts.Sum(p => (p.Y - my) * (p.Y - my));
        double sxy = pts.Sum(p => (p.X - mx) * (p.Y - my));
        double dirX, dirY;
        if (sxx >= syy)
        {
            dirX = 1;
            dirY = sxx > 0 ? sxy / sxx : 0;
        }
        else
        {
            dirX = syy > 0 ? sxy / syy : 0;
            dirY = 1;
        }
        double mag = Math.Sqrt(dirX * dirX + dirY * dirY);
        if (mag > 0) { dirX /= mag; dirY /= mag; }
        return (new Point2d(mx, my), dirX, dirY);
    }

    /// <summary>
    /// Filters out detected rects whose edges don't align with target-color line
    /// segments. For each edge, opens the mask with a line kernel in the edge's
    /// own direction — only red pixels forming a line parallel to the edge survive.
    /// A spurious hole whose edges merely cross through unrelated red pixels is
    /// rejected because those pixels don't form lines in the edge direction.
    /// </summary>
    private static List<RotatedRect> ValidateEdges(List<RotatedRect> rects, Mat mask, DetectionParams p)
    {
        if (p.MinEdgesWithSupport <= 0 || p.EdgeMinSupport <= 0)
            return rects;

        var keep = new List<RotatedRect>();
        foreach (var rect in rects)
        {
            if (HasEdgeSupport(rect, mask, p))
                keep.Add(rect);
        }
        return keep;
    }

    private static bool HasEdgeSupport(RotatedRect rect, Mat mask, DetectionParams p)
    {
        var pts = Cv2.BoxPoints(rect);
        int h = mask.Rows, w = mask.Cols;
        int klen = p.LineOpenLength;

        int edgesWithSupport = 0;
        for (int i = 0; i < 4; i++)
        {
            var p1 = pts[i];
            var p2 = pts[(i + 1) % 4];
            double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
            double edgeLen = Math.Sqrt(dx * dx + dy * dy);
            if (edgeLen < 1) continue;

            int angleDeg = (int)Math.Round(Math.Atan2(dy, dx) * 180.0 / Math.PI);
            angleDeg = ((angleDeg % 180) + 180) % 180;

            int kl = Math.Min(klen, (int)edgeLen);
            if (kl < 3) continue;
            int half = kl / 2;
            int c = kl / 2;
            double rad = angleDeg * Math.PI / 180.0;
            var k1 = new Point((int)Math.Round(c - Math.Cos(rad) * half), (int)Math.Round(c - Math.Sin(rad) * half));
            var k2 = new Point((int)Math.Round(c + Math.Cos(rad) * half), (int)Math.Round(c + Math.Sin(rad) * half));

            using var kernel = new Mat(kl, kl, MatType.CV_8UC1, Scalar.Black);
            Cv2.Line(kernel, k1, k2, new Scalar(1), 1);
            using var opened = new Mat();
            Cv2.MorphologyEx(mask, opened, MorphTypes.Open, kernel);

            using var dilateKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(3, 3));
            using var dilated = new Mat();
            Cv2.Dilate(opened, dilated, dilateKernel);

            int n = Math.Max(24, (int)edgeLen / 2);
            int total = 0, hit = 0;
            for (int t = 0; t <= n; t++)
            {
                double frac = (double)t / n;
                int x = (int)Math.Round(p1.X + dx * frac);
                int y = (int)Math.Round(p1.Y + dy * frac);
                if (x >= 0 && x < w && y >= 0 && y < h)
                {
                    total++;
                    if (dilated.At<byte>(y, x) > 0)
                        hit++;
                }
            }
            double edgeSupport = total > 0 ? (double)hit / total : 0.0;
            if (edgeSupport >= p.EdgeMinSupport)
                edgesWithSupport++;
        }
        return edgesWithSupport >= p.MinEdgesWithSupport;
    }


    private Mat BuildRuleMask(Mat hsv, ColorRule rule)
    {
        var mask = new Mat();
        Cv2.InRange(hsv, new Scalar(rule.HueLow, rule.SatMin, rule.ValMin), new Scalar(rule.HueHigh, rule.SatMax, rule.ValMax), mask);

        if (rule.UseSecondRange)
        {
            using var mask2 = new Mat();
            Cv2.InRange(hsv, new Scalar(rule.HueLow2, rule.SatMin, rule.ValMin), new Scalar(rule.HueHigh2, rule.SatMax, rule.ValMax), mask2);
            Cv2.BitwiseOr(mask, mask2, mask);
        }

        return mask;
    }

    /// <summary>
    /// Applies Lab and/or RGB conjunctive constraints. Returns a binary mask
    /// where white = passes all active color space filters.
    /// </summary>
    private static Mat ApplyColorSpaceFilter(OpenCvSharp.Size size, ColorRule rule, Mat? lab, Mat? rgb)
    {
        var result = new Mat(size, MatType.CV_8UC1, new Scalar(255));

        if (lab != null)
        {
            using var labMask = new Mat();
            Cv2.InRange(lab,
                new Scalar(rule.LabLMin, rule.LabAMin, rule.LabBMin),
                new Scalar(rule.LabLMax, rule.LabAMax, rule.LabBMax), labMask);
            Cv2.BitwiseAnd(result, labMask, result);
        }

        if (rgb != null)
        {
            using var rgbMask = new Mat();
            Cv2.InRange(rgb,
                new Scalar(rule.RgbBMin, rule.RgbGMin, rule.RgbRMin),
                new Scalar(rule.RgbBMax, rule.RgbGMax, rule.RgbRMax), rgbMask);
            Cv2.BitwiseAnd(result, rgbMask, result);
        }

        return result;
    }

    private static Mat RemoveText(Mat mask, DetectionParams p)
    {
        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);

        var glyphIds = new List<int>();
        for (int i = 1; i < n; i++)
        {
            int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
            int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            if (Math.Max(w, h) <= p.TextGlyphMaxSize && a <= p.TextGlyphMaxArea)
                glyphIds.Add(i);
        }

        if (glyphIds.Count == 0)
            return mask.Clone();

        using var glyphMask = new Mat(mask.Size(), MatType.CV_8UC1, Scalar.Black);
        foreach (int id in glyphIds)
        {
            using var comp = new Mat();
            Cv2.Compare(labels, new Scalar(id), comp, CmpType.EQ);
            Cv2.BitwiseOr(glyphMask, comp, glyphMask);
        }

        using var joinKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(p.TextJoinX, p.TextJoinY));
        using var clusters = new Mat();
        Cv2.Dilate(glyphMask, clusters, joinKernel);

        using var clabels = new Mat();
        using var cstats = new Mat();
        using var ccentroids = new Mat();
        int nc = Cv2.ConnectedComponentsWithStats(clusters, clabels, cstats, ccentroids, PixelConnectivity.Connectivity8);

        var glyphSet = new HashSet<int>(glyphIds);
        var remove = new HashSet<int>();

        for (int c = 1; c < nc; c++)
        {
            using var clusterComp = new Mat();
            Cv2.Compare(clabels, new Scalar(c), clusterComp, CmpType.EQ);
            using var both = new Mat();
            Cv2.BitwiseAnd(clusterComp, mask, both);

            var members = new List<int>();
            for (int i = 1; i < n; i++)
            {
                if (!glyphSet.Contains(i)) continue;
                int x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
                int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
                if (x < 0 || x >= clabels.Cols || y < 0 || y >= clabels.Rows) continue;
                int cl = (int)clabels.At<int>(y, x);
                if (cl == c)
                    members.Add(i);
            }

            if (members.Count >= p.TextMinCluster)
                remove.UnionWith(members);
        }

        if (remove.Count == 0)
            return mask.Clone();

        var result = mask.Clone();
        foreach (int id in remove)
        {
            using var comp = new Mat();
            Cv2.Compare(labels, new Scalar(id), comp, CmpType.EQ);
            Cv2.BitwiseNot(comp, comp);
            Cv2.BitwiseAnd(result, comp, result);
        }
        return result;
    }

    private static Mat ObbRemoveTextComponents(Mat mask, List<RotatedRect> rects, DetectionParams p)
    {
        int h = mask.Rows, w = mask.Cols;
        using var interiorMask = new Mat(h, w, MatType.CV_8UC1, Scalar.Black);

        foreach (var rect in rects)
        {
            var pts = Cv2.BoxPoints(rect);
            var ptsInt = pts.Select(f => new Point((int)Math.Round(f.X), (int)Math.Round(f.Y))).ToArray();

            using var fill = new Mat(h, w, MatType.CV_8UC1, Scalar.Black);
            Cv2.FillPoly(fill, new[] { ptsInt }, new Scalar(255));

            Mat eroded = fill;
            if (p.TwoPassShrink > 0)
            {
                eroded = new Mat();
                using var shrinkKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(2 * p.TwoPassShrink + 1, 2 * p.TwoPassShrink + 1));
                Cv2.Erode(fill, eroded, shrinkKernel);
            }

            using var edge = new Mat(h, w, MatType.CV_8UC1, Scalar.Black);
            Cv2.Polylines(edge, new[] { ptsInt }, true, new Scalar(255), 2 * p.TwoPassEdgeBand + 1);

            using var notEdge = new Mat();
            Cv2.BitwiseNot(edge, notEdge);
            using var interior = new Mat();
            Cv2.BitwiseAnd(eroded, notEdge, interior);
            Cv2.BitwiseOr(interiorMask, interior, interiorMask);

            if (eroded != fill) eroded.Dispose();
        }

        using var inside = new Mat();
        Cv2.BitwiseAnd(mask, interiorMask, inside);

        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(inside, labels, stats, centroids, PixelConnectivity.Connectivity8);

        var removeIds = new List<int>();
        for (int i = 1; i < n; i++)
        {
            int cw = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            int ch = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
            int ca = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            if (Math.Max(cw, ch) <= p.TwoPassMaxCompSize && ca <= p.TwoPassMaxCompArea)
                removeIds.Add(i);
        }

        var result = mask.Clone();
        foreach (int id in removeIds)
        {
            using var comp = new Mat();
            Cv2.Compare(labels, new Scalar(id), comp, CmpType.EQ);
            Cv2.BitwiseNot(comp, comp);
            Cv2.BitwiseAnd(result, comp, result);
        }
        return result;
    }

    private static Mat LineOpen(Mat mask, int length, int step)
    {
        var output = new Mat(mask.Size(), MatType.CV_8UC1, Scalar.Black);
        int c = length / 2;
        int half = length / 2;

        for (int angleDeg = 0; angleDeg < 180; angleDeg += step)
        {
            double rad = angleDeg * Math.PI / 180.0;
            double dx = Math.Cos(rad), dy = Math.Sin(rad);
            var p1 = new Point((int)Math.Round(c - dx * half), (int)Math.Round(c - dy * half));
            var p2 = new Point((int)Math.Round(c + dx * half), (int)Math.Round(c + dy * half));

            using var kernel = new Mat(length, length, MatType.CV_8UC1, Scalar.Black);
            Cv2.Line(kernel, p1, p2, new Scalar(1), 1);

            using var opened = new Mat();
            Cv2.MorphologyEx(mask, opened, MorphTypes.Open, kernel);
            Cv2.BitwiseOr(output, opened, output);
        }
        return output;
    }

    private static Mat LineClose(Mat mask, int length, int step)
    {
        using var inverted = new Mat();
        Cv2.BitwiseNot(mask, inverted);
        using var opened = LineOpen(inverted, length, step);
        var result = new Mat();
        Cv2.BitwiseNot(opened, result);
        return result;
    }

    private static List<RotatedRect> DetectByMethod(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        return p.DetectionMethod switch
        {
            "hole" => DetectHoles(cleaned, p, imgWidth, imgHeight),
            "contour" => DetectContourShape(cleaned, p, imgWidth, imgHeight),
            "holeShape" => DetectHolePlusShape(cleaned, p, imgWidth, imgHeight),
            "hough" => DetectHoughLines(cleaned, p, imgWidth, imgHeight),
            "ring" => DetectRingShape(cleaned, p, imgWidth, imgHeight),
            "hybrid" => DetectHybrid(cleaned, p, imgWidth, imgHeight),
            "perpEdges" => DetectPerpendicularEdges(cleaned, p, imgWidth, imgHeight),
            "holePlusPerp" => DetectHolePlusPerp(cleaned, p, imgWidth, imgHeight),
            _ => DetectHoles(cleaned, p, imgWidth, imgHeight)
        };
    }

    /// <summary>
    /// Approach B: Outer contour + ApproxPolyDP — find contours that approximate
    /// to 4 vertices with near-90° angles. Directly finds rectangular shapes
    /// without requiring hole topology.
    /// </summary>
    private static List<RotatedRect> DetectContourShape(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);

        Cv2.FindContours(cleaned, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        var rects = new List<RotatedRect>();

        foreach (var contour in contours)
        {
            double area = Cv2.ContourArea(contour);
            if (area < p.HoleMinArea || area > p.HoleMaxArea || area > maxPixels) continue;

            double perimeter = Cv2.ArcLength(contour, true);
            if (perimeter < 4) continue;

            var approx = Cv2.ApproxPolyDP(contour, 0.04 * perimeter, true);
            if (approx.Length != 4) continue;

            if (!IsApproximatelyRectangular(approx)) continue;

            var rect = Cv2.MinAreaRect(contour);
            double rw = rect.Size.Width, rh = rect.Size.Height;
            double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
            if (minSide < p.HoleMinSide || maxSide > p.HoleMaxSide) continue;

            var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
            double support = PerimeterSupport(expanded, cleaned, 2);
            if (support < p.HoleMinSupport) continue;

            rects.Add(expanded);
        }
        return rects;
    }

    /// <summary>
    /// Approach C: Hole + shape validation — current hole detection with additional
    /// ApproxPolyDP shape check on the hole contour. Hybrid: topology + geometry.
    /// </summary>
    private static List<RotatedRect> DetectHolePlusShape(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);

        Cv2.FindContours(cleaned, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);
        var rects = new List<RotatedRect>();
        if (hierarchy == null || hierarchy.Length == 0) return rects;

        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent < 0) continue;

            foreach (var holeContour in SplitIfMerged(contours[i], p))
            {
                double area = Cv2.ContourArea(holeContour);
                if (area < p.HoleMinArea || area > p.HoleMaxArea || area > maxPixels) continue;

                double perimeter = Cv2.ArcLength(holeContour, true);
                var approx = Cv2.ApproxPolyDP(holeContour, 0.04 * perimeter, true);
                if (approx.Length is < 4 or > 6) continue;

                var rect = Cv2.MinAreaRect(holeContour);
                double rw = rect.Size.Width, rh = rect.Size.Height;
                double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
                if (minSide < p.HoleMinSide || maxSide > p.HoleMaxSide) continue;

                double rectArea = rw * rh;
                if (rectArea <= 0) continue;
                double fillRatio = area / rectArea;
                if (fillRatio < p.HoleMinFill || fillRatio > p.HoleMaxFill) continue;

                var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
                double support = PerimeterSupport(expanded, cleaned, 2);
                if (support < p.HoleMinSupport || support > p.HoleMaxSupport) continue;

                rects.Add(expanded);
            }
        }
        return rects;
    }

    /// <summary>
    /// Approach D: Hough line assembly — detect line segments, group by angle,
    /// find parallel pairs and perpendicular intersections to form rectangles.
    /// </summary>
    private static List<RotatedRect> DetectHoughLines(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);
        using var edges = new Mat();
        Cv2.Canny(cleaned, edges, 50, 150);

        var lines = Cv2.HoughLinesP(edges, 1, Cv2.PI / 180, 30, p.HoleMinSide, 5);
        if (lines.Length < 4) return new List<RotatedRect>();

        var segments = new List<(Point2d a, Point2d b, double angle, double length)>();
        foreach (var line in lines)
        {
            var a = new Point2d(line.P1.X, line.P1.Y);
            var b = new Point2d(line.P2.X, line.P2.Y);
            double dx = b.X - a.X, dy = b.Y - a.Y;
            double len = Math.Sqrt(dx * dx + dy * dy);
            if (len < p.HoleMinSide) continue;
            double angle = Math.Atan2(dy, dx);
            segments.Add((a, b, angle, len));
        }
        if (segments.Count < 4) return new List<RotatedRect>();

        var rects = new List<RotatedRect>();
        for (int i = 0; i < segments.Count; i++)
        {
            for (int j = i + 1; j < segments.Count; j++)
            {
                double dAngle = AngleDiff(segments[i].angle, segments[j].angle);
                if (dAngle > 0.15) continue;

                for (int k = 0; k < segments.Count; k++)
                {
                    if (k == i || k == j) continue;
                    double aIK = AngleDiff(segments[i].angle, segments[k].angle);
                    if (Math.Abs(aIK - Math.PI / 2) > 0.2) continue;

                    for (int m = k + 1; m < segments.Count; m++)
                    {
                        if (m == i || m == j) continue;
                        double aJM = AngleDiff(segments[j].angle, segments[m].angle);
                        if (Math.Abs(aJM - Math.PI / 2) > 0.2) continue;
                        double aKM = AngleDiff(segments[k].angle, segments[m].angle);
                        if (aKM > 0.15) continue;

                        var corners = new[]
                        {
                            LineIntersect(segments[i].a, segments[i].b, segments[k].a, segments[k].b),
                            LineIntersect(segments[i].a, segments[i].b, segments[m].a, segments[m].b),
                            LineIntersect(segments[j].a, segments[j].b, segments[m].a, segments[m].b),
                            LineIntersect(segments[j].a, segments[j].b, segments[k].a, segments[k].b)
                        };
                        if (corners.Any(c => c == null)) continue;

                        var pts = corners.Select(c => new Point((int)c!.Value.X, (int)c.Value.Y)).ToArray();
                        double area = Cv2.ContourArea(pts);
                        if (area < p.HoleMinArea || area > p.HoleMaxArea || area > maxPixels) continue;

                        var rect = Cv2.MinAreaRect(pts);
                        double rw = rect.Size.Width, rh = rect.Size.Height;
                        double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
                        if (minSide < p.HoleMinSide || maxSide > p.HoleMaxSide) continue;

                        var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
                        rects.Add(expanded);
                    }
                }
            }
        }
        return rects;
    }

    /// <summary>
    /// Approach E: Ring shape — outer contour with hole child + rectangular
    /// outer boundary. Combines topology (has hole) + geometry (4 vertices).
    /// </summary>
    private static List<RotatedRect> DetectRingShape(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);

        Cv2.FindContours(cleaned, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);
        var rects = new List<RotatedRect>();
        if (hierarchy == null || hierarchy.Length == 0) return rects;

        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent >= 0) continue;
            bool hasHole = false;
            for (int j = 0; j < contours.Length; j++)
            {
                if (hierarchy[j].Parent == i) { hasHole = true; break; }
            }
            if (!hasHole) continue;

            double outerArea = Cv2.ContourArea(contours[i]);
            if (outerArea < p.HoleMinArea || outerArea > maxPixels) continue;

            double perimeter = Cv2.ArcLength(contours[i], true);
            if (perimeter < 4) continue;
            var approx = Cv2.ApproxPolyDP(contours[i], 0.04 * perimeter, true);
            if (approx.Length is < 4 or > 8) continue;

            var rect = Cv2.MinAreaRect(contours[i]);
            double rw = rect.Size.Width, rh = rect.Size.Height;
            double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
            if (minSide < p.HoleMinSide || maxSide > p.HoleMaxSide) continue;

            var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
            double support = PerimeterSupport(expanded, cleaned, 2);
            if (support < p.HoleMinSupport) continue;

            rects.Add(expanded);
        }
        return rects;
    }

    /// <summary>
    /// Approach F: Hybrid — union of hole-based (A) and contour-shape (B),
    /// then NMS to deduplicate. Orthogonal: topology + geometry.
    /// </summary>
    private static List<RotatedRect> DetectHybrid(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        var holeRects = DetectHoles(cleaned, p, imgWidth, imgHeight);
        var shapeRects = DetectContourShape(cleaned, p, imgWidth, imgHeight);
        var all = new List<RotatedRect>(holeRects);
        all.AddRange(shapeRects);
        return Nms(all, p.NmsThreshold);
    }

    /// <summary>
    /// Approach: Perpendicular edge pair detection. Detects line segments via
    /// HoughLinesP, finds L-shaped perpendicular pairs, then searches for a
    /// closing parallel pair to form a rectangle. Catches boxes with broken
    /// holes (incomplete outlines that hole detection misses).
    /// </summary>
    private static List<RotatedRect> DetectPerpendicularEdges(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);
        using var edges = new Mat();
        Cv2.Canny(cleaned, edges, 50, 150);
        var lines = Cv2.HoughLinesP(edges, 1, Cv2.PI / 180, 20, p.HoleMinSide, 3);
        if (lines.Length < 4) return new List<RotatedRect>();

        var segs = new List<(Point2d a, Point2d b, double angle, double length)>();
        foreach (var line in lines)
        {
            var a = new Point2d(line.P1.X, line.P1.Y);
            var b = new Point2d(line.P2.X, line.P2.Y);
            double dx = b.X - a.X, dy = b.Y - a.Y;
            double len = Math.Sqrt(dx * dx + dy * dy);
            if (len < p.HoleMinSide) continue;
            segs.Add((a, b, Math.Atan2(dy, dx), len));
        }
        if (segs.Count < 4) return new List<RotatedRect>();
        segs = segs.OrderByDescending(s => s.length).Take(40).ToList();

        var rects = new List<RotatedRect>();
        double perpTol = 0.25;
        double paraTol = 0.20;

        for (int i = 0; i < segs.Count; i++)
        {
            for (int j = 0; j < segs.Count; j++)
            {
                if (i == j) continue;
                double dAngle = AngleDiff(segs[i].angle, segs[j].angle);
                if (Math.Abs(dAngle - Math.PI / 2) > perpTol) continue;

                var ci = segs[i].a;
                var cj = segs[j].a;
                double dist = Math.Sqrt((ci.X - cj.X) * (ci.X - cj.X) + (ci.Y - cj.Y) * (ci.Y - cj.Y));
                if (dist > 15) continue;

                for (int k = 0; k < segs.Count; k++)
                {
                    if (k == i || k == j) continue;
                    if (AngleDiff(segs[k].angle, segs[i].angle) > paraTol) continue;

                    for (int m = 0; m < segs.Count; m++)
                    {
                        if (m == i || m == j || m == k) continue;
                        if (AngleDiff(segs[m].angle, segs[j].angle) > paraTol) continue;

                        var c1 = LineIntersect(segs[i].a, segs[i].b, segs[j].a, segs[j].b);
                        var c2 = LineIntersect(segs[i].a, segs[i].b, segs[m].a, segs[m].b);
                        var c3 = LineIntersect(segs[k].a, segs[k].b, segs[m].a, segs[m].b);
                        var c4 = LineIntersect(segs[k].a, segs[k].b, segs[j].a, segs[j].b);
                        if (c1 == null || c2 == null || c3 == null || c4 == null) continue;

                        var pts = new[] { c1, c2, c3, c4 }
                            .Select(c => new Point((int)Math.Round(c!.Value.X), (int)Math.Round(c.Value.Y)))
                            .ToArray();
                        double area = Cv2.ContourArea(pts);
                        if (area < p.HoleMinArea || area > p.HoleMaxArea || area > maxPixels) continue;

                        var rect = Cv2.MinAreaRect(pts);
                        double rw = rect.Size.Width, rh = rect.Size.Height;
                        double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
                        if (minSide < p.HoleMinSide || maxSide > p.HoleMaxSide) continue;

                        var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
                        double support = PerimeterSupport(expanded, cleaned, 2);
                        if (support < p.HoleMinSupport) continue;

                        rects.Add(expanded);
                    }
                }
            }
        }
        return rects;
    }

    /// <summary>
    /// Hybrid: hole-based + perpendicular edge detection. Union + NMS.
    /// Hole catches complete outlines, perpEdges catches broken ones.
    /// </summary>
    private static List<RotatedRect> DetectHolePlusPerp(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        var holeRects = DetectHoles(cleaned, p, imgWidth, imgHeight);
        var perpRects = DetectPerpendicularEdges(cleaned, p, imgWidth, imgHeight);
        var all = new List<RotatedRect>(holeRects);
        all.AddRange(perpRects);
        return Nms(all, p.NmsThreshold);
    }

    private static bool IsApproximatelyRectangular(Point[] pts)
    {
        if (pts.Length != 4) return false;
        for (int i = 0; i < 4; i++)
        {
            var prev = pts[(i + 3) % 4];
            var curr = pts[i];
            var next = pts[(i + 1) % 4];
            double v1x = prev.X - curr.X, v1y = prev.Y - curr.Y;
            double v2x = next.X - curr.X, v2y = next.Y - curr.Y;
            double dot = v1x * v2x + v1y * v2y;
            double mag1 = Math.Sqrt(v1x * v1x + v1y * v1y);
            double mag2 = Math.Sqrt(v2x * v2x + v2y * v2y);
            if (mag1 < 1 || mag2 < 1) return false;
            double cosAngle = Math.Abs(dot) / (mag1 * mag2);
            if (cosAngle > 0.35) return false;
        }
        return true;
    }

    private static double AngleDiff(double a1, double a2)
    {
        double d = Math.Abs(a1 - a2) % Math.PI;
        return Math.Min(d, Math.PI - d);
    }

    private static Point2d? LineIntersect(Point2d p1, Point2d p2, Point2d p3, Point2d p4)
    {
        double d1x = p2.X - p1.X, d1y = p2.Y - p1.Y;
        double d2x = p4.X - p3.X, d2y = p4.Y - p3.Y;
        double det = d1x * (-d2y) - (-d2x) * d1y;
        if (Math.Abs(det) < 1e-6) return null;
        double t = ((p3.X - p1.X) * (-d2y) - (-d2x) * (p3.Y - p1.Y)) / det;
        return new Point2d(p1.X + t * d1x, p1.Y + t * d1y);
    }

    private static List<RotatedRect> DetectHoles(Mat preserved, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);

        Cv2.FindContours(preserved, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);

        var rects = new List<RotatedRect>();
        if (hierarchy == null || hierarchy.Length == 0)
            return rects;

        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent < 0)
                continue;

            foreach (var holeContour in SplitIfMerged(contours[i], p))
            {
                double area = Cv2.ContourArea(holeContour);
                if (area < p.HoleMinArea || area > p.HoleMaxArea || area > maxPixels)
                    continue;

                var rect = Cv2.MinAreaRect(holeContour);
                double rw = rect.Size.Width;
                double rh = rect.Size.Height;
                double minSide = Math.Min(rw, rh);
                double maxSide = Math.Max(rw, rh);
                if (minSide < p.HoleMinSide || maxSide > p.HoleMaxSide)
                    continue;

                double rectArea = rw * rh;
                if (rectArea <= 0)
                    continue;
                double fillRatio = area / rectArea;
                if (fillRatio < p.HoleMinFill || fillRatio > p.HoleMaxFill)
                    continue;

                var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
                double support = PerimeterSupport(expanded, preserved, 2);
                if (support < p.HoleMinSupport || support > p.HoleMaxSupport)
                    continue;

                rects.Add(expanded);
            }
        }

        // Fallback: for strip-like detections (narrow fragments of larger boxes),
        // grow each edge outward to find the farthest red pixel cluster.
        // Iterative: repeat until box stabilizes or max 3 iterations.
        var grown = new List<RotatedRect>();
        for (int idx = 0; idx < rects.Count; idx++)
        {
            var rect = rects[idx];
            double rw = rect.Size.Width, rh = rect.Size.Height;
            double minSide = Math.Min(rw, rh), maxSide = Math.Max(rw, rh);
            if (minSide / maxSide >= 0.5 || maxSide < p.HoleMinSide * 2)
            {
                grown.Add(rect);
                continue;
            }

            var current = rect;
            for (int iter = 0; iter < 3; iter++)
            {
                double curMin = Math.Min(current.Size.Width, current.Size.Height);
                double curMax = Math.Max(current.Size.Width, current.Size.Height);
                if (curMin / curMax >= 0.5)
                    break;

                var next = GrowBox(current, preserved, maxScan: (int)(curMax * 1.5));
                double nextMin = Math.Min(next.Size.Width, next.Size.Height);
                if (nextMin <= curMin * 1.05)
                    break;

                current = next;
            }
            grown.Add(current);
        }

        return grown;
    }

    /// <summary>
    /// Splits a hole contour that is actually the union of two or more overlapping
    /// rectangles into its constituent parts, using shape concavity as the signal.
    /// A single clean box hole is convex (solidity ~0.95); an overlapping union is
    /// non-convex (solidity ~0.82-0.84). Splits at the concave "seam" junctions.
    /// </summary>
    private static List<Point[]> SplitIfMerged(Point[] contour, DetectionParams p, int maxDepth = 2)
    {
        var output = new List<Point[]>();
        SplitRecursive(contour, p, maxDepth, output);
        return output;
    }

    private static void SplitRecursive(Point[] contour, DetectionParams p, int depth, List<Point[]> output)
    {
        if (depth <= 0 || contour.Length < 12)
        {
            output.Add(contour);
            return;
        }

        double area = Cv2.ContourArea(contour);
        if (area <= 0)
        {
            output.Add(contour);
            return;
        }

        var hull = Cv2.ConvexHull(contour);
        double hullArea = Cv2.ContourArea(hull);
        if (hullArea <= 0 || area / hullArea >= p.HoleSplitSolidity)
        {
            output.Add(contour);
            return;
        }

        var parts = TrySplitAtConcavities(contour);
        if (parts == null)
        {
            output.Add(contour);
            return;
        }

        foreach (var part in parts)
            SplitRecursive(part, p, depth - 1, output);
    }

    private static List<Point[]>? TrySplitAtConcavities(Point[] contour)
    {
        double peri = Cv2.ArcLength(contour, true);
        if (peri <= 0)
            return null;

        var approx = Cv2.ApproxPolyDP(contour, 0.01 * peri, true);
        if (approx.Length < 5)
            return null;

        var poly = SignedArea(approx) < 0 ? approx.Reverse().ToArray() : approx;

        int n = poly.Length;
        var concave = new List<Point>();
        for (int i = 0; i < n; i++)
        {
            var pp = poly[(i - 1 + n) % n];
            var c = poly[i];
            var nx = poly[(i + 1) % n];
            double cross = (double)(c.X - pp.X) * (nx.Y - c.Y) - (double)(c.Y - pp.Y) * (nx.X - c.X);
            if (cross < 0)
                concave.Add(c);
        }

        // A union of exactly two rectangles has exactly two reflex (concave) vertices.
        // Anything else is noise / a single irregular box, so don't split.
        if (concave.Count != 2)
            return null;

        Point pa = concave[0], pb = concave[1];

        int ia = NearestIndex(contour, pa);
        int ib = NearestIndex(contour, pb);
        if (ia == ib)
            return null;
        if (ia > ib)
            (ia, ib) = (ib, ia);

        var subA = new List<Point>();
        for (int i = ia; i <= ib; i++) subA.Add(contour[i]);
        var subB = new List<Point>();
        for (int i = ib; i < contour.Length; i++) subB.Add(contour[i]);
        for (int i = 0; i <= ia; i++) subB.Add(contour[i]);

        if (subA.Count < 4 || subB.Count < 4)
            return null;

        double total = Cv2.ContourArea(contour);
        double areaA = Cv2.ContourArea(subA);
        double areaB = Cv2.ContourArea(subB);
        if (total <= 0 || areaA < 0.15 * total || areaB < 0.15 * total)
            return null;

        // Each part must itself be a clean (near-convex) rectangle.
        if (Solidity(subA) < PartSolidity || Solidity(subB) < PartSolidity)
            return null;

        return new List<Point[]> { subA.ToArray(), subB.ToArray() };
    }

    private const double PartSolidity = 0.88;

    private static double Solidity(List<Point> pts)
    {
        double area = Cv2.ContourArea(pts);
        var hull = Cv2.ConvexHull(pts);
        double hullArea = Cv2.ContourArea(hull);
        return hullArea > 0 ? area / hullArea : 0.0;
    }

    private static double SignedArea(Point[] pts)
    {
        double s = 0;
        for (int i = 0; i < pts.Length; i++)
        {
            var a = pts[i];
            var b = pts[(i + 1) % pts.Length];
            s += (double)a.X * b.Y - (double)b.X * a.Y;
        }
        return s / 2.0;
    }

    private static int NearestIndex(Point[] contour, Point target)
    {
        int best = 0;
        double bd = double.MaxValue;
        for (int i = 0; i < contour.Length; i++)
        {
            double dx = contour[i].X - target.X;
            double dy = contour[i].Y - target.Y;
            double d = dx * dx + dy * dy;
            if (d < bd)
            {
                bd = d;
                best = i;
            }
        }
        return best;
    }

    private static double PerimeterSupport(RotatedRect rect, Mat evidence, int dilate)
    {
        using var ev = new Mat();
        using var dilateKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(2 * dilate + 1, 2 * dilate + 1));
        Cv2.Dilate(evidence, ev, dilateKernel);

        var pts = Cv2.BoxPoints(rect);
        int n = Math.Max(24, (int)Math.Max(rect.Size.Width, rect.Size.Height) / 2);
        int total = 0, hit = 0;
        int h = evidence.Rows, w = evidence.Cols;

        for (int i = 0; i < 4; i++)
        {
            var p1 = pts[i];
            var p2 = pts[(i + 1) % 4];
            for (int t = 0; t <= n; t++)
            {
                double frac = (double)t / n;
                int x = (int)Math.Round(p1.X + (p2.X - p1.X) * frac);
                int y = (int)Math.Round(p1.Y + (p2.Y - p1.Y) * frac);
                if (x >= 0 && x < w && y >= 0 && y < h)
                {
                    total++;
                    if (ev.At<byte>(y, x) > 0)
                        hit++;
                }
            }
        }
        return total > 0 ? (double)hit / total : 0.0;
    }

    private static List<RotatedRect> Nms(List<RotatedRect> rects, double threshold)
    {
        var sorted = rects.OrderByDescending(r => r.Size.Width * r.Size.Height).ToList();
        var keep = new List<RotatedRect>();
        foreach (var r in sorted)
        {
            if (keep.All(k => IoU(r, k) < threshold))
                keep.Add(r);
        }
        return keep;
    }

    private static double IoU(RotatedRect a, RotatedRect b)
    {
        var pa = Cv2.BoxPoints(a).Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();
        var pb = Cv2.BoxPoints(b).Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();

        var ra = Cv2.BoundingRect(pa);
        var rb = Cv2.BoundingRect(pb);
        int W = Math.Max(ra.Right, rb.Right) + 1;
        int H = Math.Max(ra.Bottom, rb.Bottom) + 1;

        using var m1 = new Mat(H, W, MatType.CV_8UC1, Scalar.Black);
        using var m2 = new Mat(H, W, MatType.CV_8UC1, Scalar.Black);
        Cv2.FillPoly(m1, new[] { pa }, new Scalar(255));
        Cv2.FillPoly(m2, new[] { pb }, new Scalar(255));

        using var inter = new Mat();
        using var union = new Mat();
        Cv2.BitwiseAnd(m1, m2, inter);
        Cv2.BitwiseOr(m1, m2, union);
        int interCount = Cv2.CountNonZero(inter);
        int unionCount = Cv2.CountNonZero(union);
        return unionCount > 0 ? (double)interCount / unionCount : 0.0;
    }

    private static List<RotatedRect> DedupContained(List<RotatedRect> rects, double containThreshold)
    {
        var sorted = rects.OrderByDescending(r => r.Size.Width * r.Size.Height).ToList();
        var keep = new List<RotatedRect>();
        foreach (var r in sorted)
        {
            if (keep.All(k => InsideFrac(r, k) < containThreshold))
                keep.Add(r);
        }
        return keep;
    }

    private static double InsideFrac(RotatedRect inner, RotatedRect outer)
    {
        var pi = Cv2.BoxPoints(inner).Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();
        var po = Cv2.BoxPoints(outer).Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();

        var rb = Cv2.BoundingRect(po);
        int offX = rb.X - 1, offY = rb.Y - 1;
        var poOff = po.Select(p => new Point(p.X - offX, p.Y - offY)).ToArray();
        var piOff = pi.Select(p => new Point(p.X - offX, p.Y - offY)).ToArray();

        int H = rb.Height + 2, W = rb.Width + 2;
        using var mo = new Mat(H, W, MatType.CV_8UC1, Scalar.Black);
        using var mi = new Mat(H, W, MatType.CV_8UC1, Scalar.Black);
        Cv2.FillPoly(mo, new[] { poOff }, new Scalar(255));
        Cv2.FillPoly(mi, new[] { piOff }, new Scalar(255));

        using var inter = new Mat();
        Cv2.BitwiseAnd(mo, mi, inter);
        int interCount = Cv2.CountNonZero(inter);
        int innerCount = Cv2.CountNonZero(mi);
        return innerCount > 0 ? (double)interCount / innerCount : 0.0;
    }

    private static DetectedBox RectToDetectedBox(RotatedRect rect, int imgWidth, int imgHeight, string type, string ruleName = "")
    {
        var pts = Cv2.BoxPoints(rect);
        var corners = pts.Select(f => new Point(
            Math.Clamp((int)Math.Round(f.X), 0, imgWidth - 1),
            Math.Clamp((int)Math.Round(f.Y), 0, imgHeight - 1))).ToList();
        return new DetectedBox { Corners = OrderCorners(corners), Type = type, RuleName = ruleName };
    }

    public static List<Point> OrderCorners(List<Point> pts)
    {
        int tlIndex = 0;
        int minSum = int.MaxValue;
        for (int i = 0; i < pts.Count; i++)
        {
            int sum = pts[i].X + pts[i].Y;
            if (sum < minSum)
            {
                minSum = sum;
                tlIndex = i;
            }
        }

        var tl = pts[tlIndex];
        var rest = pts.Where((_, i) => i != tlIndex)
            .OrderBy(p => Math.Atan2((double)p.Y - tl.Y, (double)p.X - tl.X))
            .ToList();
        var ordered = new List<Point> { tl };
        ordered.AddRange(rest);
        return ordered;
    }
}
