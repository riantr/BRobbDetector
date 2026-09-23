using OpenCvSharp;

namespace BRbboxDetector.Services;

public class ImageProcessor
{
    private readonly string _outputDir;
    
    public ImageProcessor(string? outputDir = null)
    {
        _outputDir = outputDir ?? "Result";
        Directory.CreateDirectory(_outputDir);
    }
    
    public Mat ConvertToGrayscale(Mat image)
    {
        if (image.Channels() == 1)
            return image.Clone();
            
        var gray = new Mat();
        Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        return gray;
    }
    
    public Mat GenerateMask(Mat grayImage, int minThreshold, int maxThreshold)
    {
        // 范围内的像素保持原值，范围外的设为白色(255) - 白底黑字效果
        var mask = new Mat(grayImage.Size(), MatType.CV_8UC1, Scalar.White);
        
        for (int y = 0; y < grayImage.Rows; y++)
        {
            for (int x = 0; x < grayImage.Cols; x++)
            {
                byte val = grayImage.At<byte>(y, x);
                if (val >= minThreshold && val <= maxThreshold)
                {
                    mask.Set<byte>(y, x, val); // 范围内保持原灰度值
                }
                else
                {
                    mask.Set<byte>(y, x, 255); // 范围外显示白色
                }
            }
        }
        
        return mask;
    }
    
    public Mat ApplyMaskToImage(Mat originalImage, Mat mask)
    {
        // mask中：范围内保持原灰度值，范围外为255（白色）
        var result = mask.Clone();
        return result;
    }
    
    public Mat RemoveIsolatedPoints(Mat image, int thresholdWidth, int thresholdHeight)
    {
        Mat gray;
        if (image.Channels() == 1)
        {
            gray = image.Clone();
        }
        else
        {
            gray = new Mat();
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        }
        
        var binary = new Mat();
        Cv2.Threshold(gray, binary, 127, 255, ThresholdTypes.BinaryInv);
        
        Point[][] contours;
        HierarchyIndex[] hierarchy;
        Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        
        var mask = new Mat(image.Size(), MatType.CV_8UC1, Scalar.Black);
        
        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);
            
            if (rect.Width <= thresholdWidth && rect.Height <= thresholdHeight)
            {
                Cv2.DrawContours(mask, new[] { contour }, -1, new Scalar(255), -1);
            }
        }
        
        var result = image.Clone();
        
        if (result.Channels() == 1)
        {
            for (int y = 0; y < result.Rows; y++)
            {
                for (int x = 0; x < result.Cols; x++)
                {
                    if (mask.At<byte>(y, x) == 255)
                    {
                        result.Set<byte>(y, x, 255);
                    }
                }
            }
        }
        else
        {
            for (int y = 0; y < result.Rows; y++)
            {
                for (int x = 0; x < result.Cols; x++)
                {
                    if (mask.At<byte>(y, x) == 255)
                    {
                        result.Set<Vec3b>(y, x, new Vec3b(255, 255, 255));
                    }
                }
            }
        }
        
        gray.Dispose();
        binary.Dispose();
        mask.Dispose();
        
        return result;
    }
    
    public Mat PreprocessImage(Mat image, int? minThreshold = null, int? maxThreshold = null, int? denoiseThreshold = null)
    {
        // Convert to grayscale first
        var result = ConvertToGrayscale(image);
        
        if (minThreshold.HasValue && maxThreshold.HasValue)
        {
            var mask = GenerateMask(result, minThreshold.Value, maxThreshold.Value);
            var masked = ApplyMaskToImage(result, mask);
            result.Dispose();
            result = masked;
            mask.Dispose();
        }
        
        if (denoiseThreshold.HasValue && denoiseThreshold.Value > 0)
        {
            var denoised = RemoveIsolatedPoints(result, denoiseThreshold.Value, denoiseThreshold.Value);
            result.Dispose();
            result = denoised;
        }
        
        return result;
    }
}
