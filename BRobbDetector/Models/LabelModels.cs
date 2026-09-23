using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BRobbDetector.Models;

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

    [JsonPropertyName("score")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public double Score { get; set; }
}