using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace BRobbDetector.Services;

public class CategoryResult
{
    public string Category { get; set; } = string.Empty;
    public double Confidence { get; set; }
}

public class CategoryNormalizer
{
    public static readonly string[] DefaultCategories = {
        "子弹", "工具", "枪支", "液体", "火种",
        "烟火制品", "管制器具", "锂电池", "锐器"
    };

    private static readonly Regex ConfidencePattern = new(@"0\.\d+", RegexOptions.Compiled);
    private static readonly Regex DotConfidencePattern = new(@"\.\d+", RegexOptions.Compiled);
    private static readonly Regex BareConfidencePattern = new(@"[\u4e00-\u9fff]\s+(\d{1,2})\s*$", RegexOptions.Compiled);
    private readonly List<string> _categories;

    public CategoryNormalizer(IEnumerable<string>? categories = null)
    {
        _categories = (categories ?? DefaultCategories)
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .OrderByDescending(c => c.Length)
            .ToList();
    }

    public CategoryResult Normalize(string ocrText)
    {
        var result = new CategoryResult();

        if (string.IsNullOrWhiteSpace(ocrText))
            return result;

        result.Confidence = ExtractConfidence(ocrText);
        var cleaned = CleanText(ocrText);

        foreach (var cat in _categories)
        {
            if (cleaned.Contains(cat))
            {
                result.Category = cat;
                return result;
            }
        }

        var bestMatch = BestFuzzyMatch(cleaned);
        if (bestMatch.score >= 0.5)
        {
            result.Category = bestMatch.category;
            return result;
        }

        return result;
    }

    public string NormalizeCategory(string ocrText) => Normalize(ocrText).Category;

    private static double ExtractConfidence(string text)
    {
        var match = ConfidencePattern.Match(text);
        if (match.Success && double.TryParse(match.Value, out var conf))
            return conf;

        match = DotConfidencePattern.Match(text);
        if (match.Success && double.TryParse("0" + match.Value, out conf))
            return conf;

        match = BareConfidencePattern.Match(text);
        if (match.Success && int.TryParse(match.Groups[1].Value, out var pct) && pct > 0 && pct < 100)
            return pct / 100.0;

        return 0.0;
    }

    private static string CleanText(string text)
    {
        var chinese = new StringBuilder();
        foreach (var ch in text)
        {
            if (ch >= 0x4E00 && ch <= 0x9FFF)
                chinese.Append(ch);
        }
        return chinese.ToString();
    }

    private (string category, double score) BestFuzzyMatch(string text)
    {
        string bestCat = "";
        double bestScore = 0;

        foreach (var cat in _categories)
        {
            double score = Similarity(text, cat);
            if (score > bestScore)
            {
                bestScore = score;
                bestCat = cat;
            }
        }
        return (bestCat, bestScore);
    }

    private static double Similarity(string a, string b)
    {
        if (a.Length == 0 || b.Length == 0)
            return 0;

        int maxLen = Math.Max(a.Length, b.Length);
        int dist = Levenshtein(a, b);
        return 1.0 - (double)dist / maxLen;
    }

    private static int Levenshtein(string a, string b)
    {
        int n = a.Length, m = b.Length;
        if (n == 0) return m;
        if (m == 0) return n;

        var prev = new int[m + 1];
        var curr = new int[m + 1];
        for (int j = 0; j <= m; j++)
            prev[j] = j;

        for (int i = 1; i <= n; i++)
        {
            curr[0] = i;
            for (int j = 1; j <= m; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                curr[j] = Math.Min(
                    Math.Min(curr[j - 1] + 1, prev[j] + 1),
                    prev[j - 1] + cost);
            }
            (prev, curr) = (curr, prev);
        }
        return prev[m];
    }
}
