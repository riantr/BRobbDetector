using BRbboxChecker.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace BRbboxChecker.Services;

public class JsonService : IJsonService
{
    public async Task<List<BBoxData>> LoadFromJsonAsync(string labelDir, string imageDir)
    {
        return await Task.Run(() =>
        {
            var result = new List<BBoxData>();
            
            if (!Directory.Exists(labelDir))
                return result;

            var jsonFiles = Directory.GetFiles(labelDir, "*.json");
            int borderWidth = 2;

            foreach (var jsonFile in jsonFiles)
            {
                try
                {
                    var jsonContent = File.ReadAllText(jsonFile);
                    var labelData = JsonSerializer.Deserialize<LabelData>(jsonContent);
                    
                    if (labelData == null) continue;

                    var imageName = string.IsNullOrEmpty(labelData.ImgName) 
                        ? Path.GetFileNameWithoutExtension(jsonFile) + ".jpg" 
                        : labelData.ImgName;

                    // 去重边界框
                    var uniqueBboxes = new List<(int x1, int y1, int x2, int y2)>();
                    var seen = new HashSet<(int, int, int, int)>();

                    foreach (var info in labelData.Infos.Where(i => i.Type == "bbox"))
                    {
                        var points = info.Points.Split(';');
                        if (points.Length < 2) continue;

                        var parts = points[0].Split(',');
                        var x1 = int.Parse(parts[0]);
                        var y1 = int.Parse(parts[1]);

                        parts = points[1].Split(',');
                        var x2 = int.Parse(parts[0]);
                        var y2 = int.Parse(parts[1]);

                        var key = (x1, y1, x2, y2);
                        if (!seen.Contains(key))
                        {
                            seen.Add(key);
                            uniqueBboxes.Add((x1, y1, x2, y2));
                        }
                    }

                    // 如果有重复，更新JSON文件
                    if (uniqueBboxes.Count < labelData.Infos.Count(i => i.Type == "bbox"))
                    {
                        _ = UpdateJsonFileAsync(labelDir, imageName, 
                            uniqueBboxes.Select(b => new BBoxData
                            {
                                X1 = b.x1, Y1 = b.y1, X2 = b.x2, Y2 = b.y2,
                                X1Inner = b.x1 + borderWidth, Y1Inner = b.y1 + borderWidth,
                                X2Inner = b.x2 - borderWidth, Y2Inner = b.y2 - borderWidth
                            }).ToList());
                    }

                    // 添加边界框数据（不计算灰度值）
                    int bboxIndex = 1;
                    foreach (var bbox in uniqueBboxes)
                    {
                        result.Add(new BBoxData
                        {
                            ImageName = imageName,
                            BBoxIndex = bboxIndex++,
                            X1 = bbox.x1,
                            Y1 = bbox.y1,
                            X2 = bbox.x2,
                            Y2 = bbox.y2,
                            X1Inner = bbox.x1 + borderWidth,
                            Y1Inner = bbox.y1 + borderWidth,
                            X2Inner = bbox.x2 - borderWidth,
                            Y2Inner = bbox.y2 - borderWidth,
                            GrayMean = null
                        });
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"读取JSON文件失败: {jsonFile}, {ex.Message}");
                }
            }

            return result;
        });
    }

    public async Task UpdateJsonFileAsync(string labelDir, string imageName, List<BBoxData> uniqueBboxes)
    {
        await Task.Run(() =>
        {
            var jsonFile = Path.Combine(labelDir, imageName.Replace(".jpg", ".json"));
            
            if (!File.Exists(jsonFile))
                return;

            try
            {
                var jsonContent = File.ReadAllText(jsonFile);
                var labelData = JsonSerializer.Deserialize<LabelData>(jsonContent);
                
                if (labelData == null) return;

                labelData.Infos = uniqueBboxes.Select(bbox => new LabelInfo
                {
                    Points = $"{bbox.X1},{bbox.Y1};{bbox.X2},{bbox.Y2}",
                    Type = "bbox"
                }).ToList();

                var options = new JsonSerializerOptions
                {
                    WriteIndented = true,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                };

                File.WriteAllText(jsonFile, JsonSerializer.Serialize(labelData, options));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"更新JSON文件失败: {jsonFile}, {ex.Message}");
            }
        });
    }
}

public class LabelData
{
    [JsonPropertyName("img_name")]
    public string ImgName { get; set; } = string.Empty;

    [JsonPropertyName("infos")]
    public List<LabelInfo> Infos { get; set; } = new();
}

public class LabelInfo
{
    [JsonPropertyName("points")]
    public string Points { get; set; } = string.Empty;

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;
}