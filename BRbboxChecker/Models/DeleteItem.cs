using System.Collections.Generic;

namespace BRbboxChecker.Models;

public class DeleteItem
{
    public string ImageName { get; set; } = string.Empty;
    public List<ItemWithPosition> ItemsWithPositions { get; set; } = new();
}

public class ItemWithPosition
{
    public int MainListIndex { get; set; }
    public BBoxData BBoxData { get; set; } = new();
}