using BRbboxChecker.Models;
using CsvHelper;
using CsvHelper.Configuration;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace BRbboxChecker.Services;

public class CsvService : ICsvService
{
    public async Task<List<BBoxData>> LoadFromCsvAsync(string csvPath)
    {
        return await Task.Run(async () =>
        {
            if (!File.Exists(csvPath))
                return new List<BBoxData>();

            using var reader = new StreamReader(csvPath);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            
            var records = new List<BBoxData>();
            await foreach (var record in csv.GetRecordsAsync<BBoxDataCsvRecord>())
            {
                records.Add(new BBoxData
                {
                    ImageName = record.ImageName,
                    BBoxIndex = record.BBoxIndex,
                    X1 = record.X1,
                    Y1 = record.Y1,
                    X2 = record.X2,
                    Y2 = record.Y2,
                    X1Inner = record.X1Inner,
                    Y1Inner = record.Y1Inner,
                    X2Inner = record.X2Inner,
                    Y2Inner = record.Y2Inner,
                    GrayMean = record.GrayMean
                });
            }
            
            return records;
        });
    }

    public async Task SaveToCsvAsync(string csvPath, List<BBoxData> data)
    {
        await Task.Run(async () =>
        {
            var records = data.Select((item, index) => new BBoxDataCsvRecord
            {
                Index = index,
                ImageName = item.ImageName,
                BBoxIndex = item.BBoxIndex,
                GrayMean = item.GrayMean,
                X1 = item.X1,
                Y1 = item.Y1,
                X2 = item.X2,
                Y2 = item.Y2,
                X1Inner = item.X1Inner,
                Y1Inner = item.Y1Inner,
                X2Inner = item.X2Inner,
                Y2Inner = item.Y2Inner
            });

            using var writer = new StreamWriter(csvPath);
            using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
            await csv.WriteRecordsAsync(records);
        });
    }

    public async Task<Dictionary<string, Dictionary<string, int>>> LoadMainListOrderAsync(string csvPath)
    {
        return await Task.Run(() =>
        {
            if (!File.Exists(csvPath))
                return new Dictionary<string, Dictionary<string, int>>();

            var result = new Dictionary<string, Dictionary<string, int>>();
            
            using var reader = new StreamReader(csvPath);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            
            int index = 0;
            foreach (var record in csv.GetRecords<BBoxDataCsvRecord>())
            {
                var key = $"{record.X1},{record.Y1},{record.X2},{record.Y2}";
                
                if (!result.ContainsKey(record.ImageName))
                {
                    result[record.ImageName] = new Dictionary<string, int>();
                }
                
                result[record.ImageName][key] = index;
                index++;
            }
            
            return result;
        });
    }
}

public class BBoxDataCsvRecord
{
    public int Index { get; set; }
    public string ImageName { get; set; } = string.Empty;
    public int BBoxIndex { get; set; }
    public double? GrayMean { get; set; }
    public int X1 { get; set; }
    public int Y1 { get; set; }
    public int X2 { get; set; }
    public int Y2 { get; set; }
    public int X1Inner { get; set; }
    public int Y1Inner { get; set; }
    public int X2Inner { get; set; }
    public int Y2Inner { get; set; }
}