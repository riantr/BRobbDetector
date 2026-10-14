using System;
using OpenCvSharp;

namespace BRobbDetector.Services;

/// <summary>
/// Reusable geometric validation for detected quads.
///
/// Project rule: every detected target must have exactly four edges, and each edge
/// must be orthogonal (90 deg) to its adjacent edges. The rule is waived at a corner
/// iff that corner lies on the image boundary (the box may be clipped).
///
/// Production hole detection builds boxes from Cv2.MinAreaRect, which are already
/// perfect rectangles, so this is a no-op there. It is the shared post-grouping guard
/// for line/corner-based detectors (LSD, Hough) whose assembled quads may be skewed.
/// </summary>
public static class GeometryValidation
{
    public const double DefaultTolDeg = 15.0;
    public const double DefaultBoundaryBand = 6.0;

    public static double[] InteriorAngles(Point2f[] corners)
    {
        int n = corners.Length;
        var angles = new double[n];
        for (int i = 0; i < n; i++)
        {
            Point2f cur = corners[i];
            Point2f prev = corners[(i - 1 + n) % n];
            Point2f next = corners[(i + 1) % n];
            double v1x = prev.X - cur.X, v1y = prev.Y - cur.Y;
            double v2x = next.X - cur.X, v2y = next.Y - cur.Y;
            double n1 = Math.Sqrt(v1x * v1x + v1y * v1y);
            double n2 = Math.Sqrt(v2x * v2x + v2y * v2y);
            if (n1 < 1e-6 || n2 < 1e-6)
            {
                angles[i] = double.NaN;
                continue;
            }
            double cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
            cos = Math.Max(-1.0, Math.Min(1.0, cos));
            angles[i] = Math.Acos(cos) * 180.0 / Math.PI;
        }
        return angles;
    }

    public static bool CornerOnBoundary(Point2f p, int imgW, int imgH, double band)
    {
        return p.X <= band || p.X >= imgW - band || p.Y <= band || p.Y >= imgH - band;
    }

    public static bool IsOrthogonalQuad(Point2f[] corners, int imgW, int imgH,
        double tolDeg = DefaultTolDeg, double boundaryBand = DefaultBoundaryBand)
    {
        if (corners == null || corners.Length != 4)
            return false;

        double[] angles = InteriorAngles(corners);
        for (int i = 0; i < 4; i++)
        {
            if (CornerOnBoundary(corners[i], imgW, imgH, boundaryBand))
                continue; // waived: box may be clipped at the image edge
            double a = angles[i];
            if (double.IsNaN(a) || Math.Abs(a - 90.0) > tolDeg)
                return false;
        }
        return true; // if every corner was on the boundary the quad is waived
    }
}