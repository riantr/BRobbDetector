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

    public string PointsText => string.Join(";", Corners.Select(c => $"{c.X},{c.Y}"));
}

public class ColorBoxDetector
{
    public List<DetectedBox> Detect(Mat image, DetectionParams? parameters = null)
    {
        parameters ??= new DetectionParams();
        var results = new List<DetectedBox>();

        if (image.Empty())
            return results;

        using var hsv = new Mat();
        Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);

        foreach (var rule in parameters.ColorRules)
        {
            var boxes = DetectByRule(hsv, rule, parameters, image.Width, image.Height);
            results.AddRange(boxes);
        }

        return results;
    }

    public List<DetectedBox> DetectFile(string imagePath, DetectionParams? parameters = null)
    {
        using var image = Cv2.ImRead(imagePath);
        return Detect(image, parameters);
    }

    private List<DetectedBox> DetectByRule(Mat hsv, ColorRule rule, DetectionParams p, int imgWidth, int imgHeight)
    {
        using var mask = BuildRuleMask(hsv, rule);

        var rects = DetectPass(mask, p, imgWidth, imgHeight);
        if (rects.Count == 0)
            return rects.Select(r => RectToDetectedBox(r, imgWidth, imgHeight, rule.Type)).ToList();

        using var cleaned = ObbRemoveTextComponents(mask, rects, p);
        var rects2 = DetectPass(cleaned, p, imgWidth, imgHeight);
        return rects2.Select(r => RectToDetectedBox(r, imgWidth, imgHeight, rule.Type)).ToList();
    }

    private static List<RotatedRect> DetectPass(Mat mask, DetectionParams p, int imgWidth, int imgHeight)
    {
        using var noText = RemoveText(mask, p);
        using var cleaned = LineOpen(noText, p.LineOpenLength, p.LineOpenStep);
        var rects = DetectHoles(cleaned, p, imgWidth, imgHeight);
        rects = Nms(rects, p.NmsThreshold);
        rects = DedupContained(rects, p.DedupContainThreshold);
        return rects;
    }

    private Mat BuildRuleMask(Mat hsv, ColorRule rule)
    {
        var mask = new Mat();
        Cv2.InRange(hsv, new Scalar(rule.HueLow, rule.SatMin, rule.ValMin), new Scalar(rule.HueHigh, 255, 255), mask);

        if (rule.UseSecondRange)
        {
            using var mask2 = new Mat();
            Cv2.InRange(hsv, new Scalar(rule.HueLow2, rule.SatMin, rule.ValMin), new Scalar(rule.HueHigh2, 255, 255), mask2);
            Cv2.BitwiseOr(mask, mask2, mask);
        }

        return mask;
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

    private static List<RotatedRect> DetectHoles(Mat cleaned, DetectionParams p, int imgWidth, int imgHeight)
    {
        int maxPixels = (int)(p.MaxImageFraction * imgWidth * imgHeight);

        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(3, 3));
        using var closed = new Mat();
        Cv2.MorphologyEx(cleaned, closed, MorphTypes.Close, kernel, iterations: p.CloseIterations);

        Cv2.FindContours(closed, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);

        var rects = new List<RotatedRect>();
        if (hierarchy == null || hierarchy.Length == 0)
            return rects;

        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent < 0)
                continue;

            double area = Cv2.ContourArea(contours[i]);
            if (area < p.HoleMinArea || area > maxPixels)
                continue;

            var rect = Cv2.MinAreaRect(contours[i]);
            double rw = rect.Size.Width;
            double rh = rect.Size.Height;
            if (Math.Min(rw, rh) < p.HoleMinSide)
                continue;

            double rectArea = rw * rh;
            if (rectArea <= 0 || area / rectArea < p.HoleMinFill)
                continue;

            var expanded = new RotatedRect(rect.Center, new OpenCvSharp.Size(rw + p.HoleStroke, rh + p.HoleStroke), rect.Angle);
            if (PerimeterSupport(expanded, cleaned, 2) < p.HoleMinSupport)
                continue;

            rects.Add(expanded);
        }
        return rects;
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

    private static DetectedBox RectToDetectedBox(RotatedRect rect, int imgWidth, int imgHeight, string type)
    {
        var pts = Cv2.BoxPoints(rect);
        var corners = pts.Select(f => new Point(
            Math.Clamp((int)Math.Round(f.X), 0, imgWidth - 1),
            Math.Clamp((int)Math.Round(f.Y), 0, imgHeight - 1))).ToList();
        return new DetectedBox { Corners = OrderCorners(corners), Type = type };
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
