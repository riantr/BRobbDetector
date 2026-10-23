using System;
using OpenCvSharp;

namespace BRobbDetector.Services;

/// <summary>
/// Reusable geometric validation for detected quads.
///
/// Project rule (2026-10-21): a detected rectangle is only valid when
///   1. it has exactly four edges,
///   2. every corner is orthogonal (90 deg) to its adjacent edges, and
///   3. NONE of its four vertices lies on the image edge.
///
/// Rule 3 replaced the former behaviour, which waived the orthogonality
/// check at a corner on the image boundary (the box may be clipped).  A
/// rectangle that touches the image edge is now rejected outright: the human
/// annotation was clipped by the image frame, so the target is not a valid
/// rectangle.
///
/// Production hole detection builds boxes from Cv2.MinAreaRect, which are already
/// perfect rectangles, so this is a no-op there. It is the shared post-grouping guard
/// for line/corner-based detectors (LSD, Hough) whose assembled quads may be skewed.
/// </summary>
public static class GeometryValidation
{
    public const double DefaultTolDeg = 15.0;

    /// <summary>Half width (px) of the border ring that counts as "the image
    /// edge". 0 = the outermost pixel ring (the literal rule).</summary>
    public const double DefaultEdgeBand = 0.0;

    /// <summary>Legacy waiver band, kept for API compatibility only: a corner
    /// within this band of the edge used to be exempt from the orthogonality
    /// check. The waiver is off - a clipped box is invalid under rule 3.</summary>
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

    /// <summary>True when the corner lies in the border ring of half width
    /// <paramref name="band"/>. The test is symmetric: the outermost pixel ring
    /// is x == 0 / x == imgW - 1 (and likewise for y); band widens the ring
    /// inwards on all four sides.</summary>
    public static bool CornerOnBoundary(Point2f p, int imgW, int imgH, double band)
    {
        return p.X <= band || p.X >= (imgW - 1 - band)
            || p.Y <= band || p.Y >= (imgH - 1 - band);
    }

    /// <summary>True when any vertex of the rectangle lies on the image edge.
    /// Used on the emitted (rounded) corners: the rule applies to the
    /// rectangle that is reported, not only to the float quad that entered
    /// the gate.</summary>
    public static bool QuadTouchesEdge(Point[] corners, int imgW, int imgH, double band)
    {
        if (corners == null) return false;
        foreach (var c in corners)
        {
            if (c.X <= band || c.X >= (imgW - 1 - band)
                || c.Y <= band || c.Y >= (imgH - 1 - band))
                return true;
        }
        return false;
    }

    public static bool IsOrthogonalQuad(Point2f[] corners, int imgW, int imgH,
        double tolDeg = DefaultTolDeg, double edgeBand = DefaultEdgeBand)
    {
        if (corners == null || corners.Length != 4)
            return false;

        double[] angles = InteriorAngles(corners);
        for (int i = 0; i < 4; i++)
        {
            // rule 3: a vertex on the image edge invalidates the rectangle
            if (CornerOnBoundary(corners[i], imgW, imgH, edgeBand))
                return false;
            double a = angles[i];
            if (double.IsNaN(a) || Math.Abs(a - 90.0) > tolDeg)
                return false;
        }
        return true;
    }
}