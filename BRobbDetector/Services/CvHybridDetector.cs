using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace BRobbDetector.Services;

/// <summary>
/// Classical-CV quad detector — C# port of proto/cv_hybrid.py (edgefit family).
///
/// Pipeline: red HSV mask -> min-area denoise -> outer contours -> per contour
/// extract an edgefit quad (sector corners + adjacent-side TLS line
/// intersection) -> side+orthogonality admission filter -> keep-order IoU dedup.
///
/// Re-based on 2026-10-20 against the TRUE hand-drawn GT
/// (E:\Xray\CARS\data\Label@CARS_202609_raw, the red annotation burned into
/// Image@CARS_202609_annotated).  The previous pm_v5_2 numbers were measured
/// against Compare-V1@CARS_202609_annotated, a machine artifact (see
/// benchmark/gt_final.json), so they were re-measured:
///
///   C# cvhybrid, 500 imgs / 1121 GT boxes, point match:
///     before (hue<=10, hull->edgefit->mar, min_area 1500, dedup 0.75):
///         D=3 171/16.6  D=5 260/25.2  D=8 320/31.1  D=12 365/35.4
///     after  (hue<=5, edgefit only, min_area 900, dedup 0.30, ortho_tol 10):
///         D=3 547/62.8  D=5 592/67.9  D=8 598/68.6  D=12 601/69.0
///
/// Selected via DetectionMethod == "cvhybrid".
/// </summary>
public static class CvHybridDetector
{
    // red_mask thresholds (proto/common.py::red_mask)
    private const int SatMin = 80;
    private const int ValMin = 110;
    // Hue window of the hand-drawn annotation.  The annotation is pure red
    // (hue 0-4 for ~85% of its pixels) while the orange/red X-ray object
    // texture sits at hue 9-10; a wide window floods the mask with object
    // pixels.  Validated in proto/scope38_hue_sweep.py against the TRUE GT
    // (Label@CARS_202609_raw): hue<=5 lifts the point-match score from
    // 336 TP / 39.5 corr_f1 to 593 TP / 66.0 corr_f1 at D=5.
    private const int HueHigh = 5;

    // Detector config (proto/scope38_hue_sweep.py + proto/scope39_ablation.py,
    // measured against the TRUE GT Label@CARS_202609_raw).
    private const double DenoiseMinArea = 40.0;
    private const double MinArea = 900.0;
    private const double MinSide = 15.0;
    private const double MaxSide = 800.0;
    private const double MaxAspect = 12.0;
    // Orthogonality gate tolerance.  10 deg keeps the point-match score and
    // raises precision (625 vs 687 detections, corr_f1 67.9 vs 66.0) while
    // tightening the "adjacent edges are orthogonal" rule.
    private const double OrthoTol = 10.0;
    private const double BoundaryBand = 6.0;
    private const double DedupIou = 0.30;
    private const double FitBand = 6.0;
    private const int FitMinPts = 12;
    private const double FitJump = 20.0;
    // The hull / minAreaRect families are kept for reference but are not
    // emitted: the keep-order IoU dedup let their cruder quads displace the
    // accurate edgefit ones (union 553 TP / 57.4 vs edgefit only 593 / 66.0).
    private static readonly bool IncludeFallbackFamilies = false;
    private static readonly double[] ApproxEpsList = { 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18 };

    public static List<DetectedBox> Detect(Mat image)
    {
        var results = new List<DetectedBox>();
        if (image == null || image.Empty())
            return results;

        int W = image.Width, H = image.Height;
        using var mask = RedMask(image);
        using var denoised = Denoise(mask, DenoiseMinArea);
        var contours = OuterContours(denoised, MinArea);

        var hullQuads = new List<Point[]>();
        var edgeQuads = new List<Point[]>();
        var marQuads = new List<Point[]>();
        foreach (var c in contours)
        {
            AddPassed(edgeQuads, EdgefitCorners(c), W, H);
            if (!IncludeFallbackFamilies) continue;
            AddPassed(hullQuads, AdaptiveApprox(ConvexHull(c)), W, H);
            AddPassed(marQuads, MarQuad(c), W, H);
        }

        var edgeDedup = DedupKeepOrder(edgeQuads, DedupIou);
        List<Point[]> hy = edgeDedup;
        if (IncludeFallbackFamilies)
        {
            var union = new List<Point[]>(edgeDedup);
            union.AddRange(DedupKeepOrder(hullQuads, DedupIou));
            union.AddRange(DedupKeepOrder(marQuads, DedupIou));
            hy = DedupKeepOrder(union, DedupIou);
        }

        foreach (var q in hy)
        {
            results.Add(new DetectedBox { Corners = q.ToList() });
        }
        return results;
    }

    private static void AddPassed(List<Point[]> sink, Point2d[]? q, int W, int H)
    {
        var p = Pass(q, W, H);
        if (p != null) sink.Add(p);
    }

    // ---------------- mask + contours ----------------

    private static Mat RedMask(Mat image)
    {
        using var hsv = new Mat();
        Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);
        using var m1 = new Mat();
        using var m2 = new Mat();
        Cv2.InRange(hsv, new Scalar(0, SatMin, ValMin), new Scalar(HueHigh, 255, 255), m1);
        Cv2.InRange(hsv, new Scalar(170, SatMin, ValMin), new Scalar(180, 255, 255), m2);
        var mask = new Mat();
        Cv2.BitwiseOr(m1, m2, mask);
        return mask;
    }

    private static Mat Denoise(Mat mask, double minArea)
    {
        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);
        var denoised = mask.Clone();
        if (n <= 1) return denoised;
        for (int i = 1; i < n; i++)
        {
            double area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            if (area >= minArea) continue;
            using var cmp = new Mat();
            Cv2.Compare(labels, new Scalar(i), cmp, CmpTypes.EQ);
            using var notCmp = new Mat();
            Cv2.BitwiseNot(cmp, notCmp);
            Cv2.BitwiseAnd(denoised, notCmp, denoised);
        }
        return denoised;
    }

    private static List<Point[]> OuterContours(Mat mask, double minArea)
    {
        var result = new List<Point[]>();
        Cv2.FindContours(mask, out Point[][] contours, out HierarchyIndex[] hierarchy,
            RetrievalModes.CComp, ContourApproximationModes.ApproxNone);
        if (hierarchy == null) return result;
        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent >= 0) continue;
            if (Cv2.ContourArea(contours[i]) < minArea) continue;
            result.Add(contours[i]);
        }
        return result;
    }

    // ---------------- corner extractors ----------------

    private static Point[] ConvexHull(Point[] contour) => Cv2.ConvexHull(contour);

    private static Point2d[]? AdaptiveApprox(Point[] contour)
    {
        double peri = Cv2.ArcLength(contour, true);
        if (peri <= 0) return null;
        foreach (var eps in ApproxEpsList)
        {
            var approx = Cv2.ApproxPolyDP(contour, eps * peri, true);
            if (approx.Length == 4)
                return approx.Select(p => new Point2d(p.X, p.Y)).ToArray();
        }
        return null;
    }

    private static Point2d[]? MarQuad(Point[] contour)
    {
        var rect = Cv2.MinAreaRect(contour);
        if (rect.Size.Width < 1 || rect.Size.Height < 1) return null;
        var pts = Cv2.BoxPoints(rect);
        return pts.Select(p => new Point2d(p.X, p.Y)).ToArray();
    }

    private static Point2d[]? SectorCorners(Point[] contour)
    {
        var rect = Cv2.MinAreaRect(contour);
        if (rect.Size.Width < 1 || rect.Size.Height < 1) return null;
        double ang = rect.Angle * Math.PI / 180.0;
        var a = new Point2d(Math.Cos(ang), Math.Sin(ang));
        var b = new Point2d(-a.Y, a.X);
        var center = new Point2d(rect.Center.X, rect.Center.Y);
        var pts = contour.Select(p => new Point2d(p.X, p.Y)).ToArray();
        var corners = new Point2d[4];
        int k = 0;
        foreach (double su in new[] { 1.0, -1.0 })
            foreach (double sv in new[] { 1.0, -1.0 })
            {
                double best = double.NegativeInfinity;
                int bestIdx = 0;
                var dir = new Point2d(su * a.X + sv * b.X, su * a.Y + sv * b.Y);
                for (int i = 0; i < pts.Length; i++)
                {
                    double vx = pts[i].X - center.X, vy = pts[i].Y - center.Y;
                    double proj = vx * dir.X + vy * dir.Y;
                    if (proj > best) { best = proj; bestIdx = i; }
                }
                corners[k++] = pts[bestIdx];
            }
        return corners;
    }

    private static Point2d[] OrderQuad(Point2d[] pts)
    {
        double cx = pts.Average(p => p.X), cy = pts.Average(p => p.Y);
        return pts.OrderBy(p => Math.Atan2(p.Y - cy, p.X - cx)).ToArray();
    }

    private static (Point2d mu, Point2d dir) FitLineTls(List<Point2d> pts)
    {
        double mx = pts.Average(p => p.X), my = pts.Average(p => p.Y);
        var mu = new Point2d(mx, my);
        int n = pts.Count;
        using var src = new Mat(n, 2, MatType.CV_64FC1);
        for (int i = 0; i < n; i++)
        {
            src.Set(i, 0, pts[i].X - mx);
            src.Set(i, 1, pts[i].Y - my);
        }
        using var w = new Mat();
        using var u = new Mat();
        using var vt = new Mat();
        Cv2.SVDecomp(src, w, u, vt);
        double dx = vt.At<double>(0, 0), dy = vt.At<double>(0, 1);
        return (mu, new Point2d(dx, dy));
    }

    private static Point2d? LineIsect(Point2d m1, Point2d d1, Point2d m2, Point2d d2)
    {
        double det = d1.X * (-d2.Y) - (-d2.X) * d1.Y;
        if (Math.Abs(det) < 1e-9) return null;
        double rhs0 = m2.X - m1.X, rhs1 = m2.Y - m1.Y;
        double t = (rhs0 * (-d2.Y) - (-d2.X) * rhs1) / det;
        return new Point2d(m1.X + t * d1.X, m1.Y + t * d1.Y);
    }

    private static Point2d[]? EdgefitCorners(Point[] contour)
    {
        var sec = SectorCorners(contour);
        if (sec == null) return null;
        var o = OrderQuad(sec);
        var pts = contour.Select(p => new Point2d(p.X, p.Y)).ToArray();
        var lines = new (Point2d mu, Point2d dir)[4];
        for (int i = 0; i < 4; i++)
        {
            var a = o[i];
            var b = o[(i + 1) % 4];
            double dx = b.X - a.X, dy = b.Y - a.Y;
            double L = Math.Sqrt(dx * dx + dy * dy);
            if (L < 2) return null;
            double ux = dx / L, uy = dy / L;
            double nx = -uy, ny = ux;
            var sel = new List<Point2d>();
            for (int j = 0; j < pts.Length; j++)
            {
                double rx = pts[j].X - a.X, ry = pts[j].Y - a.Y;
                double along = rx * ux + ry * uy;
                double perp = rx * nx + ry * ny;
                if (along > -0.10 * L && along < 1.10 * L && Math.Abs(perp) < FitBand)
                    sel.Add(pts[j]);
            }
            if (sel.Count < FitMinPts) return null;
            lines[i] = FitLineTls(sel);
        }
        var refined = new Point2d[4];
        for (int i = 0; i < 4; i++)
        {
            var prev = lines[(i - 1 + 4) % 4];
            var cur = lines[i];
            var x = LineIsect(prev.mu, prev.dir, cur.mu, cur.dir);
            if (x == null) return null;
            double shiftX = x.Value.X - o[i].X, shiftY = x.Value.Y - o[i].Y;
            if (Math.Sqrt(shiftX * shiftX + shiftY * shiftY) > FitJump) return null;
            refined[i] = x.Value;
        }
        return refined;
    }

    // ---------------- admission filter + dedup ----------------

    private static bool SideOk(Point2d[] q)
    {
        double mn = double.MaxValue, mx = double.MinValue;
        for (int i = 0; i < 4; i++)
        {
            var a = q[i]; var b = q[(i + 1) % 4];
            double s = Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
            if (s < mn) mn = s;
            if (s > mx) mx = s;
        }
        if (mn < MinSide || mx > MaxSide) return false;
        if (mx / Math.Max(mn, 1e-6) > MaxAspect) return false;
        return true;
    }

    private static Point[]? Pass(Point2d[]? q, int W, int H)
    {
        if (q == null || q.Length != 4) return null;
        if (!SideOk(q)) return null;
        var p2f = q.Select(p => new Point2f((float)p.X, (float)p.Y)).ToArray();
        if (!GeometryValidation.IsOrthogonalQuad(p2f, W, H, OrthoTol, BoundaryBand)) return null;
        return q.Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();
    }

    private static double IouObb(Point[] a, Point[] b)
    {
        var ra = Cv2.BoundingRect(a);
        var rb = Cv2.BoundingRect(b);
        int cw = Math.Max(ra.Right, rb.Right) + 1;
        int ch = Math.Max(ra.Bottom, rb.Bottom) + 1;
        using var m1 = new Mat(ch, cw, MatType.CV_8UC1, Scalar.Black);
        using var m2 = new Mat(ch, cw, MatType.CV_8UC1, Scalar.Black);
        Cv2.FillPoly(m1, new[] { a }, new Scalar(255));
        Cv2.FillPoly(m2, new[] { b }, new Scalar(255));
        using var inter = new Mat();
        Cv2.BitwiseAnd(m1, m2, inter);
        using var uni = new Mat();
        Cv2.BitwiseOr(m1, m2, uni);
        double i = Cv2.CountNonZero(inter);
        double u = Cv2.CountNonZero(uni);
        return u <= 0 ? 0.0 : i / u;
    }

    private static List<Point[]> DedupKeepOrder(List<Point[]> quads, double iouThr)
    {
        var keep = new List<Point[]>();
        foreach (var r in quads)
        {
            bool overlaps = false;
            foreach (var k in keep)
            {
                if (IouObb(r, k) >= iouThr) { overlaps = true; break; }
            }
            if (!overlaps) keep.Add(r);
        }
        return keep;
    }
}