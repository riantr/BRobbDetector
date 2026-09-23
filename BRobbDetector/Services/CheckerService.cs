using System;
using System.IO;
using System.Threading.Tasks;
using BRobbDetector.Models;
using OpenCvSharp;

namespace BRobbDetector.Services;

public class CheckerService
{
    /// <summary>
    /// Computes the mean gray value of the border ring between the outer rect
    /// and the inner rect (inset by 2 px), excluding 2x2 px corners, matching
    /// the BRbboxChecker algorithm.
    /// </summary>
    public Task<double?> CalculateGrayMeanAsync(string imagePath, BBoxData box)
    {
        return Task.Run<double?>(() =>
        {
            try
            {
                using var image = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
                if (image.Empty())
                    return null;

                int width = image.Cols;
                int height = image.Rows;

                int x1 = Math.Clamp(box.X1, 0, width - 1);
                int y1 = Math.Clamp(box.Y1, 0, height - 1);
                int x2 = Math.Clamp(box.X2, 0, width - 1);
                int y2 = Math.Clamp(box.Y2, 0, height - 1);

                int roiW = x2 - x1 + 1;
                int roiH = y2 - y1 + 1;
                if (roiW <= 0 || roiH <= 0)
                    return null;

                var rect = new Rect(x1, y1, roiW, roiH);
                using var roi = new Mat(image, rect);

                using var ringMask = new Mat(roi.Size(), MatType.CV_8UC1, Scalar.All(255));

                int innerX1 = Math.Max(0, box.X1Inner - x1);
                int innerY1 = Math.Max(0, box.Y1Inner - y1);
                int innerX2 = Math.Min(roiW - 1, box.X2Inner - x1);
                int innerY2 = Math.Min(roiH - 1, box.Y2Inner - y1);

                if (innerX2 >= innerX1 && innerY2 >= innerY1)
                {
                    Cv2.Rectangle(ringMask, new Rect(innerX1, innerY1, innerX2 - innerX1 + 1, innerY2 - innerY1 + 1),
                        Scalar.Black, -1);
                }

                const int cornerSize = 2;
                Cv2.Rectangle(ringMask, new Rect(0, 0, cornerSize, cornerSize), Scalar.Black, -1);
                Cv2.Rectangle(ringMask, new Rect(roiW - cornerSize, 0, cornerSize, cornerSize), Scalar.Black, -1);
                Cv2.Rectangle(ringMask, new Rect(0, roiH - cornerSize, cornerSize, cornerSize), Scalar.Black, -1);
                Cv2.Rectangle(ringMask, new Rect(roiW - cornerSize, roiH - cornerSize, cornerSize, cornerSize), Scalar.Black, -1);

                var mean = Cv2.Mean(roi, ringMask);
                return mean.Val0;
            }
            catch (Exception)
            {
                return null;
            }
        });
    }
}