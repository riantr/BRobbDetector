using BRbboxChecker.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace BRbboxChecker.Services;

public interface ICsvService
{
    Task<List<BBoxData>> LoadFromCsvAsync(string csvPath);
    Task SaveToCsvAsync(string csvPath, List<BBoxData> data);
    Task<Dictionary<string, Dictionary<string, int>>> LoadMainListOrderAsync(string csvPath);
}