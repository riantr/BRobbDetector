using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using BRobbDetector.Models;

namespace BRobbDetector.Services;

public class LabelJsonService
{
    private static readonly JsonSerializerOptions WriteOptions = new()
    {
        WriteIndented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    private static readonly JsonSerializerOptions ReadOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public LabelData? Load(string jsonPath)
    {
        if (!File.Exists(jsonPath))
            return null;

        var content = File.ReadAllText(jsonPath);
        return JsonSerializer.Deserialize<LabelData>(content, ReadOptions);
    }

    public void Save(string jsonPath, LabelData data)
    {
        var json = JsonSerializer.Serialize(data, WriteOptions);
        File.WriteAllText(jsonPath, json);
    }

    public List<BBoxData> LoadBoxesFromDir(string labelDir, string imageDir)
    {
        var result = new List<BBoxData>();
        if (!Directory.Exists(labelDir))
            return result;

        const int borderWidth = 2;

        foreach (var jsonFile in Directory.GetFiles(labelDir, "*.json"))
        {
            LabelData? data;
            try
            {
                data = Load(jsonFile);
            }
            catch
            {
                continue;
            }

            if (data == null)
                continue;

            var imageName = string.IsNullOrEmpty(data.ImgName)
                ? Path.GetFileNameWithoutExtension(jsonFile) + ".jpg"
                : data.ImgName;

            var seen = new HashSet<(int, int, int, int)>();
            int index = 1;

            foreach (var info in data.Infos)
            {
                var points = ParsePoints(info.Points);
                if (points.Count < 2)
                    continue;

                int x1, y1, x2, y2;
                if (points.Count >= 4)
                {
                    x1 = points.Min(p => p.X);
                    y1 = points.Min(p => p.Y);
                    x2 = points.Max(p => p.X);
                    y2 = points.Max(p => p.Y);
                }
                else
                {
                    x1 = points[0].X;
                    y1 = points[0].Y;
                    x2 = points[1].X;
                    y2 = points[1].Y;
                }

                var key = (x1, y1, x2, y2);
                if (!seen.Add(key))
                    continue;

                result.Add(new BBoxData
                {
                    ImageName = imageName,
                    BBoxIndex = index++,
                    X1 = x1,
                    Y1 = y1,
                    X2 = x2,
                    Y2 = y2,
                    X1Inner = x1 + borderWidth,
                    Y1Inner = y1 + borderWidth,
                    X2Inner = x2 - borderWidth,
                    Y2Inner = y2 - borderWidth,
                    Type = info.Type,
                    Points = info.Points
                });
            }
        }

        return result;
    }

    public static List<(int X, int Y)> ParsePoints(string points)
    {
        var result = new List<(int, int)>();
        if (string.IsNullOrWhiteSpace(points))
            return result;

        foreach (var part in points.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var coords = part.Split(',');
            if (coords.Length == 2 &&
                int.TryParse(coords[0].Trim(), out int x) &&
                int.TryParse(coords[1].Trim(), out int y))
            {
                result.Add((x, y));
            }
        }

        return result;
    }

    public void WriteResult(string outputPath, string imageName, IEnumerable<DetectedBox> boxes)
    {
        var data = new LabelData
        {
            ImgName = imageName,
            Infos = boxes.Select(b => new LabelInfo
            {
                Points = b.PointsText,
                Type = b.Type
            }).ToList()
        };

        Save(outputPath, data);
    }

    public void ApplyDeletions(string labelDir, List<BBoxData> boxes)
    {
        foreach (var group in boxes.GroupBy(b => b.ImageName))
        {
            var jsonFile = Path.Combine(labelDir, Path.GetFileNameWithoutExtension(group.Key) + ".json");
            if (!File.Exists(jsonFile))
                continue;

            var data = Load(jsonFile);
            if (data == null)
                continue;

            var deletedKeys = group.Where(b => b.MarkedForDelete)
                .Select(b => (b.X1, b.Y1, b.X2, b.Y2))
                .ToHashSet();

            data.Infos = data.Infos.Where(info =>
            {
                var points = ParsePoints(info.Points);
                if (points.Count < 2)
                    return true;

                int x1, y1, x2, y2;
                if (points.Count >= 4)
                {
                    x1 = points.Min(p => p.X);
                    y1 = points.Min(p => p.Y);
                    x2 = points.Max(p => p.X);
                    y2 = points.Max(p => p.Y);
                }
                else
                {
                    x1 = points[0].X;
                    y1 = points[0].Y;
                    x2 = points[1].X;
                    y2 = points[1].Y;
                }

                return !deletedKeys.Contains((x1, y1, x2, y2));
            }).ToList();

            Save(jsonFile, data);
        }
    }
}