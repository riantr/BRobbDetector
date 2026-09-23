using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using OpenCvSharp;
using BRbboxDetector.Models;

namespace BRbboxDetector.Services;

public class BRbboxDetector : IDisposable
{
    private readonly ImageProcessor _processor;
    private readonly TemplateMatcher _matcher;
    public readonly string OutputDir;
    
    public BRbboxDetector(string templateDir = "template", string outputDir = "Result")
    {
        _processor = new ImageProcessor(outputDir);
        _matcher = new TemplateMatcher(templateDir);
        OutputDir = outputDir;
    }
    
    public BboxResult? Detect(string imagePath, PreprocessParams? preprocessParams = null, 
        double matchThreshold = 0.9, bool useHollow = false)
    {
        var image = Cv2.ImRead(imagePath);
        if (image.Empty())
            return null;
        
        if (preprocessParams != null)
        {
            image = _processor.PreprocessImage(
                image,
                preprocessParams.MinThreshold,
                preprocessParams.MaxThreshold,
                preprocessParams.DenoiseThreshold
            );
        }
        
        var results = _matcher.MatchAllTemplates(image, matchThreshold, useHollow);
        
        var result = new BboxResult
        {
            ImageName = Path.GetFileName(imagePath),
            Infos = results
        };
        
        image.Dispose();
        
        return result;
    }
    
    public List<string> DetectBatch(string inputDir, PreprocessParams? preprocessParams = null, 
        double matchThreshold = 0.9, bool useHollow = false, 
        Action<int, int, string>? progressCallback = null,
        CancellationToken cancellationToken = default)
    {
        var supportedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
        
        var files = Directory.GetFiles(inputDir)
            .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .ToList();
        
        var jsonPaths = new List<string>();
        var total = files.Count;
        
        for (int i = 0; i < files.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var file = files[i];
            var fileName = Path.GetFileName(file);
            
            progressCallback?.Invoke(i + 1, total, fileName);
            
            var result = Detect(file, preprocessParams, matchThreshold, useHollow);
            
            if (result != null)
            {
                var jsonPath = Path.Combine(OutputDir, Path.GetFileNameWithoutExtension(file) + ".json");
                var json = JsonSerializer.Serialize(result, new JsonSerializerOptions 
                { 
                    WriteIndented = true,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
                File.WriteAllText(jsonPath, json);
                jsonPaths.Add(jsonPath);
            }
        }
        
        return jsonPaths;
    }
    
    public void Dispose()
    {
        _matcher.Dispose();
    }
}
