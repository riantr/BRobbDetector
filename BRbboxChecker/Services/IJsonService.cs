using BRbboxChecker.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace BRbboxChecker.Services;

public interface IJsonService
{
    Task<List<BBoxData>> LoadFromJsonAsync(string labelDir, string imageDir);
    Task UpdateJsonFileAsync(string labelDir, string imageName, List<BBoxData> uniqueBboxes);
}