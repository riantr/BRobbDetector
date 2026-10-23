using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace BRobbDetector.Services;

/// <summary>
/// Hole-based box detector — C# port of proto/scope45_holes.py.
///
/// A hand-drawn red rectangle encloses a HOLE: the region inside the stroke.
/// findContours(RETR_CCOMP) reports that interior as a child contour of the
/// stroke's connected component, so the holes give the boxes directly - no
/// component splitting needed.  Two boxes that merely touch keep their
/// interiors separate.
///
/// The hole boundary is the INNER edge of the stroke while the GT side line
/// runs through the stroke CENTRE, so every hole side is refined by a
/// total-least-squares fit over the red band just outside the hole edge and
/// the four fitted lines are intersected.
///
/// Two refinements (pm_v10, 2026-10-23) attack the "leader arrow pulls a box
/// corner" failure mode:
///   * FitSideLine runs HoleFitPass robust passes: the band outside a hole side
///     can hold ANOTHER stroke, and those outliers drag the TLS fit onto the
///     wrong line, so the fit is repeated with the outliers removed.
///   * RefineSides re-fits the 4 sides on the red stroke in a SYMMETRIC band
///     around each side line and re-intersects them; the hole quad otherwise
///     inherits the orientation and offset error of the minAreaRect fit.
///
/// Measured on the TRUE hand-drawn GT (E:\Xray\CARS\data\Label@CARS_202609_raw,
/// 500 images / 1076 rule-consistent GT boxes, point match):
///   D=3 857 TP / 81.2 corr_f1, D=5 901 TP / 85.4, D=8 926 / 87.6, D=12 938 / 88.6
///   (benchmark/pm_v10_refine.json - bit-exact with proto/scope45_holes.py)
/// </summary>
public static class HoleDetector
{
    // ---- hole geometry config (proto/scope45_holes.py::HOLE_DEFAULT) ----
    private const double HoleMinSide = 14.0;     // min hole side (the box is slightly larger)
    private const double HoleFill = 0.55;        // min contourArea/rectArea for a hole to be rectangular
    private const double HoleMinCompArea = 60.0; // min area of the stroke component that owns the hole
    private const double HoleExpand = 2.7;       // fallback stroke half-width (measured median)
    private const double HoleBand = 3.0;         // band OUTSIDE the hole side that holds the stroke
    private const int HolePts = 12;              // min red points on a side to fit its line
    /// <summary>Extra robust passes of the side fit (outlier rejection).</summary>
    private const int HoleFitPass = 2;
    /// <summary>Residual (px) that counts as an outlier in a robust side fit.</summary>
    private const double HoleFitTol = 1.5;
    private const double HoleOvershoot = 0.06;   // how far past the hole corner to sample
    /// <summary>Min fraction of the quad perimeter that is red.</summary>
    public const double MinRedSupport = 0.5;
    private const double HoleAreaMin = 20.0;     // min contour area of a hole

    // ---- side refinement config (proto/scope45_holes.py::HOLE_DEFAULT) ----
    /// <summary>1 = re-fit the 4 sides on the red stroke after the hole quad.</summary>
    public const int HoleRefine = 1;
    /// <summary>Half width of the symmetric refine band (px).</summary>
    public const double HoleRefineBand = 3.0;
    /// <summary>Max 90th percentile residual of a refine band (px).</summary>
    public const double HoleRefineTol = 3.0;
    /// <summary>Min red points on a side of the refine band.</summary>
    public const int HoleRefinePts = 8;

    /// <summary>All hole contours of the red mask.</summary>
    /// <remarks>
    /// OpenCV's hierarchy is [next, previous, first_child, parent], so the
    /// siblings of a hole are chained through hier[i][0] - walking hier[i][2]
    /// would only ever yield the FIRST hole of a component.  (Getting this
    /// wrong cost 264 TP in the Python prototype.)
    ///
    /// OpenCvSharp's HierarchyIndex only exposes Next / Previous / Parent, but
    /// findContours always stores the first child of a contour at index
    /// (parent + 1); the chain then continues through Next until the parent
    /// changes.
    /// </remarks>
    public static List<Point[]> HolesOf(Mat mask)
    {
        var result = new List<Point[]>();
        Cv2.FindContours(mask, out Point[][] contours, out HierarchyIndex[] hierarchy,
            RetrievalModes.CComp, ContourApproximationModes.ApproxNone);
        if (hierarchy == null) return result;
        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent >= 0) continue;
            if (Cv2.ContourArea(contours[i]) < HoleMinCompArea) continue;
            int child = i + 1;
            while (child >= 0 && child < hierarchy.Length && hierarchy[child].Parent == i)
            {
                result.Add(contours[child]);
                child = hierarchy[child].Next;
            }
        }
        return result;
    }

    /// <summary>The quad of one hole, or null when the hole is not a box interior.</summary>
    public static Point2d[]? HoleQuad(Point[] hole, Mat mask)
    {
        double area = Cv2.ContourArea(hole);
        if (area < HoleAreaMin) return null;
        var rect = Cv2.MinAreaRect(hole);
        if (rect.Size.Width < HoleMinSide || rect.Size.Height < HoleMinSide) return null;
        if (area / Math.Max(rect.Size.Width * rect.Size.Height, 1e-6) < HoleFill) return null;

        var corners = OrderQuad(Cv2.BoxPoints(rect).Select(p => new Point2d(p.X, p.Y)).ToArray());
        var lines = new (Point2d mu, Point2d dir)[4];
        for (int i = 0; i < 4; i++)
        {
            if (!SideGeom(corners, i, out Point2d a, out Point2d u, out Point2d n, out double L))
                return null;
            var fit = FitSideLine(mask, a, u, n, L);
            if (fit == null)
            {
                lines[i] = (new Point2d(a.X + HoleExpand * n.X, a.Y + HoleExpand * n.Y), u);
            }
            else
            {
                var d = fit.Value.dir;
                if (d.X * u.X + d.Y * u.Y < 0) d = new Point2d(-d.X, -d.Y);
                lines[i] = (fit.Value.mu, d);
            }
        }
        var q = new Point2d[4];
        for (int i = 0; i < 4; i++)
        {
            var prev = lines[(i - 1 + 4) % 4];
            var cur = lines[i];
            var x = LineIsect(prev.mu, prev.dir, cur.mu, cur.dir);
            if (x == null) return null;
            q[i] = x.Value;
        }
        foreach (var p in q)
            if (double.IsNaN(p.X) || double.IsNaN(p.Y) || double.IsInfinity(p.X) || double.IsInfinity(p.Y))
                return null;
        return q;
    }

    /// <summary>
    /// Re-fit the 4 sides of a quad on the red stroke centred on each side.
    ///
    /// The hole quad is built from the hole's minAreaRect: its orientation and
    /// its side offsets inherit the error of the minAreaRect fit.  A second pass
    /// that fits each side on the red pixels in a symmetric band around the side
    /// line and re-intersects the sides removes most of that error.
    /// </summary>
    public static Point2d[]? RefineSides(Mat mask, Point2d[] q, double band,
                                          int minPts, double tol)
    {
        int H = mask.Height, W = mask.Width;
        var lines = new (Point2d mu, Point2d dir)[4];
        for (int i = 0; i < 4; i++)
        {
            var a = q[i];
            var b = q[(i + 1) % 4];
            double dx = b.X - a.X, dy = b.Y - a.Y;
            double L = Math.Sqrt(dx * dx + dy * dy);
            if (L < 1e-6) return null;
            double ux = dx / L, uy = dy / L;
            double nx = -uy, ny = ux;
            int nt = Math.Max((int)L, 8);
            int nOff = (int)Math.Ceiling(2.0 * band + 0.5);
            var pts = new List<Point2d>();
            for (int t = 0; t < nt; t++)
            {
                double tt = (double)t / (nt - 1);
                double bx = a.X + tt * dx;
                double by = a.Y + tt * dy;
                for (int s = 0; s < nOff; s++)
                {
                    double off = -band + s;
                    double px = bx + off * nx;
                    double py = by + off * ny;
                    int xi = (int)Math.Round(px);
                    int yi = (int)Math.Round(py);
                    if (xi < 0) xi = 0; else if (xi >= W) xi = W - 1;
                    if (yi < 0) yi = 0; else if (yi >= H) yi = H - 1;
                    if (mask.At<byte>(yi, xi) > 0)
                        pts.Add(new Point2d(px, py));
                }
            }
            if (pts.Count < minPts) return null;
            var fit = FitLineTls(pts);
            var mu = fit.mu;
            var dd = fit.dir;
            if (dd.X * ux + dd.Y * uy < 0) dd = new Point2d(-dd.X, -dd.Y);
            // reject the fit if the band is not a single stroke
            var resid = new List<double>(pts.Count);
            foreach (var p in pts)
            {
                double rx = p.X - mu.X, ry = p.Y - mu.Y;
                resid.Add(Math.Abs(rx * nx + ry * ny));
            }
            resid.Sort();
            if (Percentile90(resid) > tol) return null;
            lines[i] = (mu, dd);
        }
        var res = new Point2d[4];
        for (int i = 0; i < 4; i++)
        {
            var prev = lines[(i - 1 + 4) % 4];
            var cur = lines[i];
            var x = LineIsect(prev.mu, prev.dir, cur.mu, cur.dir);
            if (x == null) return null;
            res[i] = x.Value;
        }
        foreach (var p in res)
            if (double.IsNaN(p.X) || double.IsNaN(p.Y) || double.IsInfinity(p.X) || double.IsInfinity(p.Y))
                return null;
        return res;
    }

    /// <summary>numpy-compatible 90th percentile (linear interpolation).</summary>
    private static double Percentile90(List<double> sorted)
    {
        int n = sorted.Count;
        if (n == 0) return 0.0;
        double pos = 0.9 * (n - 1);
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        if (lo == hi) return sorted[lo];
        double frac = pos - lo;
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }

    /// <summary>Anchor, unit direction, OUTWARD normal and length of side i.</summary>
    private static bool SideGeom(Point2d[] pts, int i, out Point2d a, out Point2d u,
                                 out Point2d n, out double L)
    {
        a = pts[i];
        var b = pts[(i + 1) % 4];
        double dx = b.X - a.X, dy = b.Y - a.Y;
        L = Math.Sqrt(dx * dx + dy * dy);
        if (L < 1e-6) { u = n = new Point2d(); return false; }
        u = new Point2d(dx / L, dy / L);
        n = new Point2d(-u.Y, u.X);
        double cx = 0, cy = 0;
        foreach (var p in pts) { cx += p.X; cy += p.Y; }
        cx /= pts.Length; cy /= pts.Length;
        if ((cx - a.X) * n.X + (cy - a.Y) * n.Y > 0) n = new Point2d(-n.X, -n.Y);
        return true;
    }

    /// <summary>
    /// TLS line through the red stroke just OUTSIDE a hole side.
    /// The band can contain red pixels of ANOTHER stroke (a crossing stroke, a
    /// neighbouring box, red text).  Those outliers drag the total-least-squares
    /// fit onto the wrong line, so the fit is repeated with the outliers removed.
    /// </summary>
    private static (Point2d mu, Point2d dir)? FitSideLine(Mat mask, Point2d a, Point2d u,
                                                           Point2d n, double L)
    {
        int ns = Math.Max((int)(L * 1.5), 20);
        int nOff = (int)Math.Floor(HoleBand + 0.5) + 1;   // offsets 0..HoleBand
        int H = mask.Height, W = mask.Width;
        var pts = new List<Point2d>();
        for (int i = 0; i < ns; i++)
        {
            double t = -HoleOvershoot + (1.0 + 2.0 * HoleOvershoot) * i / (ns - 1);
            double bx = a.X + t * u.X * L;
            double by = a.Y + t * u.Y * L;
            for (int k = 0; k < nOff; k++)
            {
                double off = k;
                double px = bx + off * n.X;
                double py = by + off * n.Y;
                int xi = (int)Math.Round(px);
                int yi = (int)Math.Round(py);
                if (xi < 0) xi = 0; else if (xi >= W) xi = W - 1;
                if (yi < 0) yi = 0; else if (yi >= H) yi = H - 1;
                if (mask.At<byte>(yi, xi) > 0)
                    pts.Add(new Point2d(px, py));
            }
        }
        if (pts.Count < HolePts) return null;

        var fit = FitLineTls(pts);
        var mu = fit.mu;
        var d = fit.dir;
        if (d.X * u.X + d.Y * u.Y < 0) d = new Point2d(-d.X, -d.Y);

        for (int pass = 0; pass < HoleFitPass; pass++)
        {
            var nn = new Point2d(-d.Y, d.X);
            var kept = new List<Point2d>();
            foreach (var p in pts)
            {
                double rx = p.X - mu.X, ry = p.Y - mu.Y;
                if (Math.Abs(rx * nn.X + ry * nn.Y) <= HoleFitTol)
                    kept.Add(p);
            }
            if (kept.Count < HolePts) break;
            pts = kept;
            fit = FitLineTls(pts);
            mu = fit.mu;
            d = fit.dir;
            if (d.X * u.X + d.Y * u.Y < 0) d = new Point2d(-d.X, -d.Y);
        }
        return (mu, d);
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

    private static Point2d? LineIsect(Point2d p1, Point2d d1, Point2d p2, Point2d d2)
    {
        double det = d1.X * (-d2.Y) - (-d2.X) * d1.Y;
        if (Math.Abs(det) < 1e-9) return null;
        double rhs0 = p2.X - p1.X, rhs1 = p2.Y - p1.Y;
        double t = (rhs0 * (-d2.Y) - (-d2.X) * rhs1) / det;
        return new Point2d(p1.X + t * d1.X, p1.Y + t * d1.Y);
    }

    public static Point2d[] OrderQuad(Point2d[] pts)
    {
        double cx = pts.Average(p => p.X), cy = pts.Average(p => p.Y);
        return pts.OrderBy(p => Math.Atan2(p.Y - cy, p.X - cx)).ToArray();
    }

    /// <summary>Fraction of the quad perimeter that is red in the mask.</summary>
    public static double RedSupport(Mat mask, Point2d[] quad, int step = 2)
    {
        int H = mask.Height, W = mask.Width;
        int tot = 0, red = 0;
        for (int i = 0; i < 4; i++)
        {
            var a = quad[i];
            var b = quad[(i + 1) % 4];
            double L = Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
            int n = Math.Max((int)(L / step), 2);
            for (int k = 0; k < n; k++)
            {
                double t = (double)k / n;
                double px = a.X + t * (b.X - a.X);
                double py = a.Y + t * (b.Y - a.Y);
                int xi = (int)Math.Round(px);
                int yi = (int)Math.Round(py);
                if (xi < 0) xi = 0; else if (xi >= W) xi = W - 1;
                if (yi < 0) yi = 0; else if (yi >= H) yi = H - 1;
                if (mask.At<byte>(yi, xi) > 0) red++;
            }
            tot += n;
        }
        return (double)red / Math.Max(tot, 1);
    }

    /// <summary>Fraction of the quad perimeter that is red in the mask.</summary>
    public static double RedSupport(Mat mask, Point[] quad, int step = 2)
        => RedSupport(mask, quad.Select(p => new Point2d(p.X, p.Y)).ToArray(), step);
}