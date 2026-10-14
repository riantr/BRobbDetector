using System;
using OpenCvSharp;

namespace BRobbDetector.Services;

/// <summary>
/// Constant-correction box refinement with preserve-sup gate (LOOP-3).
/// Grows detected boxes by a fixed factor to recover near-miss detections,
/// reverting any correction that would drop a box below the unlabeled-real threshold.
/// </summary>
public static class BoxRefinement
{
    public const double WidthFactor = 1.172;   // exp(0.159)
    public const double HeightFactor = 1.187;  // exp(0.171)
    public const double DyShiftNormalized = -0.025;
    public const double SupThr = 0.30;
    public const double MinSide = 12.0;
    public const double MaxAspect = 8.0;

    public static RotatedRect ApplyConstantCorrection(RotatedRect rect)
    {
        float newW = (float)(rect.Size.Width * WidthFactor);
        float newH = (float)(rect.Size.Height * HeightFactor);
        float newY = (float)(rect.Center.Y + DyShiftNormalized * rect.Size.Height);
        return new RotatedRect(
            new Point2f(rect.Center.X, newY),
            new Size2f(newW, newH),
            rect.Angle);
    }

    public static double RedSupport(RotatedRect rect, Mat maskD)
    {
        var pts = Cv2.BoxPoints(rect);
        double totalSup = 0;
        for (int i = 0; i < 4; i++)
        {
            var p = pts[i];
            var q = pts[(i + 1) % 4];
            double dx = q.X - p.X, dy = q.Y - p.Y;
            double L = Math.Sqrt(dx * dx + dy * dy);
            int ns = Math.Max((int)L, 2);
            int count = 0;
            for (int t = 0; t < ns; t++)
            {
                double f = (double)t / ns;
                int x = Math.Clamp((int)Math.Round(p.X + f * dx), 0, maskD.Cols - 1);
                int y = Math.Clamp((int)Math.Round(p.Y + f * dy), 0, maskD.Rows - 1);
                if (maskD.At<byte>(y, x) > 0)
                    count++;
            }
            totalSup += (double)count / ns;
        }
        return totalSup / 4.0;
    }

    public static bool IsUnlabeledReal(RotatedRect rect, Mat maskD)
    {
        double w = rect.Size.Width, h = rect.Size.Height;
        double minS = Math.Min(w, h);
        double maxS = Math.Max(w, h);
        if (minS < MinSide || maxS / Math.Max(minS, 1e-6) > MaxAspect)
            return false;
        return RedSupport(rect, maskD) >= SupThr;
    }

    /// <summary>
    /// Apply constant correction, but revert if it would drop a box from
    /// unlabeled-real to non-unlabeled-real (preserve-sup gate).
    /// </summary>
    public static RotatedRect RefineWithPreserveSup(RotatedRect rect, Mat maskD)
    {
        var corrected = ApplyConstantCorrection(rect);
        bool wasReal = IsUnlabeledReal(rect, maskD);
        bool stillReal = IsUnlabeledReal(corrected, maskD);
        if (wasReal && !stillReal)
            return rect;
        return corrected;
    }
}