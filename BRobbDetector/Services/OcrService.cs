using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using BRobbDetector.Models;
using OpenCvSharp;

namespace BRobbDetector.Services;

public class OcrResult
{
    public string Text { get; set; } = string.Empty;
    public double Confidence { get; set; }
}

public class OcrService : IDisposable
{
    private Process? _process;
    private StreamWriter? _stdin;
    private StreamReader? _stdout;
    private bool _disposed;

    private static readonly string[] PythonCandidates = {
        "python", "python3", "py",
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "miniconda3", "python.exe"),
    };

    public bool IsRunning => _process != null && !_process.HasExited;

    public void Start(string? sidecarPath = null)
    {
        if (IsRunning)
            return;

        sidecarPath ??= FindSidecar();
        if (!File.Exists(sidecarPath))
            throw new FileNotFoundException($"OCR sidecar not found: {sidecarPath}");

        var python = FindPython();
        if (python == null)
            throw new InvalidOperationException("Python not found. Install Python or miniconda3.");

        var psi = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"\"{sidecarPath}\"",
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            CreateNoWindow = true
        };

        _process = new Process { StartInfo = psi };
        _process.Start();
        _stdin = _process.StandardInput;
        _stdout = _process.StandardOutput;

        var readyLine = _stdout.ReadLine();
        if (string.IsNullOrEmpty(readyLine))
            throw new InvalidOperationException("OCR sidecar failed to start.");
    }

    public OcrResult Recognize(string imagePath)
    {
        if (!IsRunning || _stdin == null || _stdout == null)
            throw new InvalidOperationException("OCR service is not running.");

        _stdin.WriteLine(imagePath);
        _stdin.Flush();

        var line = _stdout.ReadLine();
        if (string.IsNullOrEmpty(line))
            return new OcrResult();

        using var doc = JsonDocument.Parse(line);
        var root = doc.RootElement;
        return new OcrResult
        {
            Text = root.TryGetProperty("text", out var t) ? t.GetString() ?? "" : "",
            Confidence = root.TryGetProperty("conf", out var c) ? c.GetDouble() : 0.0
        };
    }

    public async Task<OcrResult> RecognizeAsync(string imagePath)
    {
        return await Task.Run(() => Recognize(imagePath));
    }

    private static string FindPython()
    {
        foreach (var candidate in PythonCandidates)
        {
            try
            {
                var psi = new ProcessStartInfo(candidate, "--version")
                {
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                };
                var p = Process.Start(psi);
                p?.WaitForExit(3000);
                if (p?.ExitCode == 0)
                    return candidate;
            }
            catch { }
        }
        return null!;
    }

    private static string FindSidecar()
    {
        var candidates = new List<string>();

        var baseDir = AppContext.BaseDirectory;
        for (int i = 0; i < 6; i++)
        {
            candidates.Add(Path.Combine(baseDir, "proto", "ocr_sidecar.py"));
            var parent = Directory.GetParent(baseDir);
            if (parent == null) break;
            baseDir = parent.FullName;
        }

        baseDir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 4; i++)
        {
            candidates.Add(Path.Combine(baseDir, "proto", "ocr_sidecar.py"));
            var parent = Directory.GetParent(baseDir);
            if (parent == null) break;
            baseDir = parent.FullName;
        }

        return candidates.FirstOrDefault(File.Exists)
            ?? Path.Combine(AppContext.BaseDirectory, "proto", "ocr_sidecar.py");
    }

    public static Rect CropForOcr(int imgW, int imgH, int minX, int minY, int maxX, int maxY,
        TextPosition pos, int extent)
    {
        int x, y, w, h;
        int margin = 4;

        switch (pos)
        {
            case TextPosition.Above:
                x = minX - margin;
                y = minY - extent;
                w = maxX - minX + 2 * margin;
                h = extent;
                break;
            case TextPosition.Below:
                x = minX - margin;
                y = maxY;
                w = maxX - minX + 2 * margin;
                h = extent;
                break;
            case TextPosition.Left:
                x = minX - extent;
                y = minY - margin;
                w = extent;
                h = maxY - minY + 2 * margin;
                break;
            case TextPosition.Right:
                x = maxX;
                y = minY - margin;
                w = extent;
                h = maxY - minY + 2 * margin;
                break;
            default: // Inside
                x = minX + margin;
                y = minY + margin;
                w = maxX - minX - 2 * margin;
                h = maxY - minY - 2 * margin;
                break;
        }

        x = Math.Clamp(x, 0, imgW - 1);
        y = Math.Clamp(y, 0, imgH - 1);
        w = Math.Clamp(w, 1, imgW - x);
        h = Math.Clamp(h, 1, imgH - y);
        return new Rect(x, y, w, h);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        try
        {
            if (_stdin != null && IsRunning)
            {
                _stdin.WriteLine("QUIT");
                _stdin.Flush();
                _process?.WaitForExit(5000);
            }
        }
        catch { }

        try { _process?.Kill(); } catch { }
        _process?.Dispose();
        _stdin?.Dispose();
        _stdout?.Dispose();
    }
}