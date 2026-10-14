using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace BRobbDetector.Services;

public class YoloObbDetector : IDisposable
{
    private readonly InferenceSession _session;
    private const int InputSize = 480;
    private const int NumChannels = 6;
    private const float ConfThreshold = 0.55f;
    private const float IoUThreshold = 0.45f;
    private const float PadValue = 114f;

    public YoloObbDetector(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"YOLO ONNX model not found: {modelPath}");
        _session = new InferenceSession(modelPath);
    }

    public static string ResolveDefaultModelPath()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Assets", "yolov8s_obb.onnx"),
            Path.Combine(AppContext.BaseDirectory, "yolov8s_obb.onnx"),
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c))
                return c;
        }
        return candidates[0];
    }

    public List<DetectedBox> Detect(Mat image)
    {
        var results = new List<DetectedBox>();
        if (image.Empty())
            return results;

        var (inputTensor, ratio, padLeft, padTop) = Preprocess(image);
        var rawOutput = RunInference(inputTensor);
        var detections = DecodeOutput(rawOutput, ratio, padLeft, padTop);
        var kept = NonMaxSuppression(detections, IoUThreshold);

        foreach (var det in kept)
        {
            results.Add(new DetectedBox
            {
                Corners = det.Corners.Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToList(),
                RuleName = "yolo",
                Type = string.Empty
            });
        }

        return results;
    }

    public List<DetectedBox> DetectFile(string imagePath)
    {
        using var image = Cv2.ImRead(imagePath);
        return Detect(image);
    }

    private (DenseTensor<float> tensor, float ratio, int padLeft, int padTop) Preprocess(Mat image)
    {
        int h = image.Height;
        int w = image.Width;

        float r = MathF.Min((float)InputSize / h, (float)InputSize / w);
        int newW = (int)Math.Round(w * r);
        int newH = (int)Math.Round(h * r);

        using var resized = new Mat();
        Cv2.Resize(image, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Linear);

        float dw = (InputSize - newW) / 2f;
        float dh = (InputSize - newH) / 2f;
        int top = (int)Math.Round(dh - 0.1);
        int bottom = (int)Math.Round(dh + 0.1);
        int left = (int)Math.Round(dw - 0.1);
        int right = (int)Math.Round(dw + 0.1);

        using var padded = new Mat();
        Cv2.CopyMakeBorder(resized, padded, top, bottom, left, right, BorderTypes.Constant,
            new Scalar(PadValue, PadValue, PadValue));

        using var rgb = new Mat();
        Cv2.CvtColor(padded, rgb, ColorConversionCodes.BGR2RGB);

        using var floatMat = new Mat();
        rgb.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

        var ch = new Mat[3];
        Cv2.Split(floatMat, out ch);

        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        for (int c = 0; c < 3; c++)
        {
            ch[c].GetArray(out float[] data);
            int offset = c * InputSize * InputSize;
            for (int i = 0; i < data.Length; i++)
                tensor.SetValue(offset + i, data[i]);
            ch[c].Dispose();
        }

        return (tensor, r, left, top);
    }

    private float[] RunInference(DenseTensor<float> inputTensor)
    {
        var inputName = _session.InputMetadata.First().Key;
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();
        return output.ToArray();
    }

    private List<YoloDetection> DecodeOutput(float[] rawOutput, float ratio, int padLeft, int padTop)
    {
        int numAnchors = rawOutput.Length / NumChannels;
        var detections = new List<YoloDetection>();

        for (int i = 0; i < numAnchors; i++)
        {
            float cx = rawOutput[0 * numAnchors + i];
            float cy = rawOutput[1 * numAnchors + i];
            float w = rawOutput[2 * numAnchors + i];
            float h = rawOutput[3 * numAnchors + i];
            float conf = rawOutput[4 * numAnchors + i];
            float angle = rawOutput[5 * numAnchors + i];

            if (conf < ConfThreshold)
                continue;

            float cosA = MathF.Cos(angle);
            float sinA = MathF.Sin(angle);
            float hw = w / 2f;
            float hh = h / 2f;

            var corners = new Point2f[4];
            corners[0] = new Point2f(cx + (-hw * cosA + hh * sinA), cy + (-hw * sinA - hh * cosA));
            corners[1] = new Point2f(cx + (hw * cosA + hh * sinA), cy + (hw * sinA - hh * cosA));
            corners[2] = new Point2f(cx + (hw * cosA - hh * sinA), cy + (hw * sinA + hh * cosA));
            corners[3] = new Point2f(cx + (-hw * cosA - hh * sinA), cy + (-hw * sinA + hh * cosA));

            for (int j = 0; j < 4; j++)
            {
                float x = (corners[j].X - padLeft) / ratio;
                float y = (corners[j].Y - padTop) / ratio;
                corners[j] = new Point2f(x, y);
            }

            float minX = corners.Min(c => c.X);
            float minY = corners.Min(c => c.Y);
            float maxX = corners.Max(c => c.X);
            float maxY = corners.Max(c => c.Y);

            detections.Add(new YoloDetection
            {
                Corners = corners,
                Confidence = conf,
                AABB = new Rect2f(minX, minY, maxX - minX, maxY - minY)
            });
        }

        return detections;
    }

    private static List<YoloDetection> NonMaxSuppression(List<YoloDetection> dets, float iouThr)
    {
        var sorted = dets.OrderByDescending(d => d.Confidence).ToList();
        var kept = new List<YoloDetection>();
        var suppressed = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (suppressed[i])
                continue;
            kept.Add(sorted[i]);

            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j])
                    continue;
                if (IoU(sorted[i].AABB, sorted[j].AABB) > iouThr)
                    suppressed[j] = true;
            }
        }

        return kept;
    }

    private static float IoU(Rect2f a, Rect2f b)
    {
        float x1 = Math.Max(a.X, b.X);
        float y1 = Math.Max(a.Y, b.Y);
        float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        if (x2 <= x1 || y2 <= y1)
            return 0f;

        float interArea = (x2 - x1) * (y2 - y1);
        float unionArea = a.Width * a.Height + b.Width * b.Height - interArea;
        return interArea / unionArea;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }

    private class YoloDetection
    {
        public Point2f[] Corners { get; set; } = new Point2f[4];
        public float Confidence { get; set; }
        public Rect2f AABB { get; set; }
    }
}