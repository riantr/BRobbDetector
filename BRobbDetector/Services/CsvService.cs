using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BRobbDetector.Models;

namespace BRobbDetector.Services;

public class CsvService
{
    private static readonly string[] Columns =
    {
        "Index", "ImageName", "BBoxIndex", "GrayMean",
        "X1", "Y1", "X2", "Y2",
        "X1Inner", "Y1Inner", "X2Inner", "Y2Inner", "Type"
    };

    public Task<List<BBoxData>> LoadFromCsvAsync(string csvPath)
    {
        return Task.Run(() =>
        {
            var records = new List<BBoxData>();
            if (!File.Exists(csvPath))
                return records;

            var lines = File.ReadAllLines(csvPath);
            if (lines.Length < 2)
                return records;

            var header = lines[0].Split(',');
            var map = new Dictionary<string, int>();
            for (int i = 0; i < header.Length; i++)
                map[header[i].Trim()] = i;

            foreach (var line in lines.Skip(1))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var fields = SplitCsvLine(line);
                string Get(string col) => map.TryGetValue(col, out int idx) && idx < fields.Length ? fields[idx] : string.Empty;

                records.Add(new BBoxData
                {

                    ImageName = Get("ImageName"),
                    BBoxIndex = ParseInt(Get("BBoxIndex")),
                    GrayMean = double.TryParse(Get("GrayMean"), NumberStyles.Float, CultureInfo.InvariantCulture, out var gm) ? gm : null,
                    X1 = ParseInt(Get("X1")),
                    Y1 = ParseInt(Get("Y1")),
                    X2 = ParseInt(Get("X2")),
                    Y2 = ParseInt(Get("Y2")),
                    X1Inner = ParseInt(Get("X1Inner")),
                    Y1Inner = ParseInt(Get("Y1Inner")),
                    X2Inner = ParseInt(Get("X2Inner")),
                    Y2Inner = ParseInt(Get("Y2Inner")),
                    Type = Get("Type")
                });
            }

            return records;
        });
    }

    public Task SaveToCsvAsync(string csvPath, List<BBoxData> data)
    {
        return Task.Run(() =>
        {
            var sb = new StringBuilder();
            sb.AppendLine(string.Join(",", Columns));

            for (int i = 0; i < data.Count; i++)
            {
                var item = data[i];
                sb.Append(i).Append(',');
                sb.Append(Escape(item.ImageName)).Append(',');
                sb.Append(item.BBoxIndex).Append(',');
                sb.Append(item.GrayMean?.ToString("0.##", CultureInfo.InvariantCulture) ?? string.Empty).Append(',');
                sb.Append(item.X1).Append(',');
                sb.Append(item.Y1).Append(',');
                sb.Append(item.X2).Append(',');
                sb.Append(item.Y2).Append(',');
                sb.Append(item.X1Inner).Append(',');
                sb.Append(item.Y1Inner).Append(',');
                sb.Append(item.X2Inner).Append(',');
                sb.Append(item.Y2Inner).Append(',');
                sb.Append(Escape(item.Type));
                sb.AppendLine();
            }

            File.WriteAllText(csvPath, sb.ToString());
        });
    }

    private static string Escape(string value)
    {
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }

    private static string[] SplitCsvLine(string line)
    {
        var fields = new List<string>();
        var sb = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    sb.Append(c);
                }
            }
            else if (c == '"')
            {
                inQuotes = true;
            }
            else if (c == ',')
            {
                fields.Add(sb.ToString());
                sb.Clear();
            }
            else
            {
                sb.Append(c);
            }
        }

        fields.Add(sb.ToString());
        return fields.ToArray();
    }

    private static int ParseInt(string value) => int.TryParse(value, out int v) ? v : 0;
}