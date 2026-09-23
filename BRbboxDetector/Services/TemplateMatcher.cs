using OpenCvSharp;
using BRbboxDetector.Models;
using OpenCvSharpRange = OpenCvSharp.Range;
using Interpolation = OpenCvSharp.InterpolationFlags;

namespace BRbboxDetector.Services;

public class TemplateMatcher : IDisposable
{
    private readonly string _templateDir;
    private readonly Dictionary<string, Mat> _templates = new();
    
    public TemplateMatcher(string templateDir)
    {
        _templateDir = templateDir;
        LoadTemplates();
    }
    
    private void LoadTemplates()
    {
        var supportedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        
        if (!Directory.Exists(_templateDir))
            return;
            
        var files = Directory.GetFiles(_templateDir);
        foreach (var file in files)
        {
            var ext = Path.GetExtension(file).ToLowerInvariant();
            if (!supportedExtensions.Contains(ext))
                continue;
                
            var templateName = Path.GetFileNameWithoutExtension(file);
            var templateImg = Cv2.ImRead(file, ImreadModes.Grayscale);
            
            if (!templateImg.Empty())
            {
                _templates[templateName] = templateImg;
            }
        }
    }
    
    public Mat CreateHollowTemplate(Mat templateImg, int borderWidth = 2)
    {
        var hollow = Mat.Zeros(templateImg.Size(), MatType.CV_8UC1);
        
        if (borderWidth <= templateImg.Rows)
            templateImg[new OpenCvSharpRange(0, borderWidth), new OpenCvSharpRange(0, templateImg.Cols)].CopyTo(hollow[new OpenCvSharpRange(0, borderWidth), new OpenCvSharpRange(0, templateImg.Cols)]);
        
        if (borderWidth <= templateImg.Rows)
            templateImg[new OpenCvSharpRange(templateImg.Rows - borderWidth, templateImg.Rows), new OpenCvSharpRange(0, templateImg.Cols)].CopyTo(hollow[new OpenCvSharpRange(templateImg.Rows - borderWidth, templateImg.Rows), new OpenCvSharpRange(0, templateImg.Cols)]);
        
        if (borderWidth <= templateImg.Cols)
            templateImg[new OpenCvSharpRange(0, templateImg.Rows), new OpenCvSharpRange(0, borderWidth)].CopyTo(hollow[new OpenCvSharpRange(0, templateImg.Rows), new OpenCvSharpRange(0, borderWidth)]);
        
        if (borderWidth <= templateImg.Cols)
            templateImg[new OpenCvSharpRange(0, templateImg.Rows), new OpenCvSharpRange(templateImg.Cols - borderWidth, templateImg.Cols)].CopyTo(hollow[new OpenCvSharpRange(0, templateImg.Rows), new OpenCvSharpRange(templateImg.Cols - borderWidth, templateImg.Cols)]);
        
        return hollow;
    }
    
    public MatchResult? DetectByHollowTemplate(Mat image, Mat hollowTemplate, double threshold = 0.5, double[]? scales = null)
    {
        scales ??= new double[] { 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 };
        
        MatchResult? bestMatch = null;
        double bestScore = -1;
        
        foreach (var scale in scales)
        {
            var scaledTemplate = new Mat();
            Cv2.Resize(hollowTemplate, scaledTemplate, new OpenCvSharp.Size(), scale, scale, InterpolationFlags.Linear);
            
            if (scaledTemplate.Rows > image.Rows || scaledTemplate.Cols > image.Cols)
            {
                scaledTemplate.Dispose();
                continue;
            }
            
            var result = new Mat();
            Cv2.MatchTemplate(image, scaledTemplate, result, TemplateMatchModes.CCoeffNormed);
            
            double maxVal;
            Point maxLoc;
            Cv2.MinMaxLoc(result, out _, out maxVal, out _, out maxLoc);
            
            if (maxVal > bestScore && maxVal >= threshold)
            {
                bestScore = maxVal;
                bestMatch = new MatchResult
                {
                    X = maxLoc.X,
                    Y = maxLoc.Y,
                    Width = scaledTemplate.Cols,
                    Height = scaledTemplate.Rows,
                    Score = maxVal,
                    Scale = scale,
                    Method = "hollow"
                };
            }
            
            scaledTemplate.Dispose();
            result.Dispose();
        }
        
        return bestMatch;
    }
    
    public MatchResult? DetectByNormalTemplate(Mat image, Mat template, double threshold = 0.5, double[]? scales = null)
    {
        scales ??= new double[] { 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 };
        
        MatchResult? bestMatch = null;
        double bestScore = -1;
        
        foreach (var scale in scales)
        {
            var scaledTemplate = new Mat();
            Cv2.Resize(template, scaledTemplate, new OpenCvSharp.Size(), scale, scale, InterpolationFlags.Linear);
            
            if (scaledTemplate.Rows > image.Rows || scaledTemplate.Cols > image.Cols)
            {
                scaledTemplate.Dispose();
                continue;
            }
            
            var result = new Mat();
            Cv2.MatchTemplate(image, scaledTemplate, result, TemplateMatchModes.CCoeffNormed);
            
            double maxVal;
            Point maxLoc;
            Cv2.MinMaxLoc(result, out _, out maxVal, out _, out maxLoc);
            
            if (maxVal > bestScore && maxVal >= threshold)
            {
                bestScore = maxVal;
                bestMatch = new MatchResult
                {
                    X = maxLoc.X,
                    Y = maxLoc.Y,
                    Width = scaledTemplate.Cols,
                    Height = scaledTemplate.Rows,
                    Score = maxVal,
                    Scale = scale,
                    Method = "normal"
                };
            }
            
            scaledTemplate.Dispose();
            result.Dispose();
        }
        
        return bestMatch;
    }
    
    public List<(int X1, int Y1, int X2, int Y2, double Score, string Method)> MatchSingleTemplate(
        Mat image, Mat template, double threshold = 0.8, bool useHollow = true)
    {
        if (template.Empty() || template.Width == 0 || template.Height == 0)
            return new List<(int, int, int, int, double, string)>();
        
        Mat imgGray;
        if (image.Channels() == 1)
        {
            imgGray = image.Clone();
        }
        else
        {
            imgGray = new Mat();
            Cv2.CvtColor(image, imgGray, ColorConversionCodes.BGR2GRAY);
        }
        
        Mat tmplGray;
        if (template.Channels() == 1)
        {
            tmplGray = template.Clone();
        }
        else
        {
            tmplGray = new Mat();
            Cv2.CvtColor(template, tmplGray, ColorConversionCodes.BGR2GRAY);
        }
        
        var detections = new List<MatchResult>();
        
        if (useHollow)
        {
            var hollowTemplate = CreateHollowTemplate(tmplGray);
            var hollowMatch = DetectByHollowTemplate(imgGray, hollowTemplate, threshold);
            if (hollowMatch != null)
                detections.Add(hollowMatch);
            hollowTemplate.Dispose();
        }
        
        var normalMatch = DetectByNormalTemplate(imgGray, tmplGray, threshold);
        if (normalMatch != null)
            detections.Add(normalMatch);
        
        var bboxes = new List<(int, int, int, int, double, string)>();
        
        if (detections.Count > 0)
        {
            var best = detections.OrderByDescending(d => d.Score).First();
            bboxes.Add((best.X, best.Y, best.X + best.Width, best.Y + best.Height, best.Score, best.Method));
        }
        
        imgGray.Dispose();
        tmplGray.Dispose();
        
        return bboxes;
    }
    
    public List<BboxInfo> MatchAllTemplates(Mat image, double threshold = 0.8, bool useHollow = true)
    {
        var results = new List<BboxInfo>();
        
        foreach (var kvp in _templates)
        {
            var templateImg = kvp.Value;
            var bboxes = MatchSingleTemplate(image, templateImg, threshold, useHollow);
            
            foreach (var bbox in bboxes)
            {
                var (x1, y1, x2, y2, score, method) = bbox;
                var pointsStr = string.Format("{0},{1};{2},{3}", x1, y1, x2, y2);
                results.Add(new BboxInfo
                {
                    Points = pointsStr,
                    Type = "bbox"
                });
            }
        }
        
        return results;
    }
    
    public void Dispose()
    {
        foreach (var template in _templates.Values)
        {
            template.Dispose();
        }
        _templates.Clear();
    }
}
