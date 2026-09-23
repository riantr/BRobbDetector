using Avalonia;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using BRbboxChecker.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;
using System.Threading.Tasks;

namespace BRbboxChecker.Services;

public class ImageProcessingService : IImageProcessingService
{
    public async Task<Bitmap?> LoadImageAsync(string imagePath)
    {
        try
        {
            await Task.Run(() => { }); // 确保异步
            
            using var fileStream = File.OpenRead(imagePath);
            return new Bitmap(fileStream);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"加载图像失败: {ex.Message}");
            return null;
        }
    }

    public async Task<double?> CalculateGrayMeanAsync(string imagePath, int x1, int y1, int x2, int y2, 
                                                       int x1Inner, int y1Inner, int x2Inner, int y2Inner)
    {
        try
        {
            return await Task.Run(() =>
            {
                using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);
                var grayImage = image.CloneAs<L8>();
                
                // 提取外矩形区域
                var outerRoi = grayImage.Clone(x => x.Crop(new Rectangle(x1, y1, x2 - x1 + 1, y2 - y1 + 1)));
                
                // 创建内矩形掩码
                var mask = new bool[outerRoi.Height, outerRoi.Width];
                for (int y = 0; y < outerRoi.Height; y++)
                {
                    for (int x = 0; x < outerRoi.Width; x++)
                    {
                        mask[y, x] = true;
                    }
                }
                
                // 移除内矩形区域
                int innerY1 = y1Inner - y1;
                int innerY2 = y2Inner - y1;
                int innerX1 = x1Inner - x1;
                int innerX2 = x2Inner - x1;
                
                for (int y = innerY1; y <= innerY2 && y < outerRoi.Height; y++)
                {
                    for (int x = innerX1; x <= innerX2 && x < outerRoi.Width; x++)
                    {
                        mask[y, x] = false;
                    }
                }
                
                // 去除四角2x2区域
                int cornerSize = 2;
                for (int y = 0; y < cornerSize && y < outerRoi.Height; y++)
                {
                    for (int x = 0; x < cornerSize && x < outerRoi.Width; x++)
                    {
                        mask[y, x] = false; // 左上角
                    }
                    for (int x = Math.Max(0, outerRoi.Width - cornerSize); x < outerRoi.Width; x++)
                    {
                        mask[y, x] = false; // 右上角
                    }
                }
                
                for (int y = Math.Max(0, outerRoi.Height - cornerSize); y < outerRoi.Height; y++)
                {
                    for (int x = 0; x < cornerSize && x < outerRoi.Width; x++)
                    {
                        mask[y, x] = false; // 左下角
                    }
                    for (int x = Math.Max(0, outerRoi.Width - cornerSize); x < outerRoi.Width; x++)
                    {
                        mask[y, x] = false; // 右下角
                    }
                }
                
                // 计算边框区域灰度均值
                double sum = 0;
                int count = 0;
                
                for (int y = 0; y < outerRoi.Height; y++)
                {
                    for (int x = 0; x < outerRoi.Width; x++)
                    {
                        if (mask[y, x])
                        {
                            sum += outerRoi[x, y].PackedValue;
                            count++;
                        }
                    }
                }
                
                return count > 0 ? sum / count : (double?)null;
            });
        }
        catch (Exception ex)
        {
            Console.WriteLine($"计算灰度均值失败: {ex.Message}");
            return null;
        }
    }

    public Bitmap? DrawBoundingBox(Bitmap? bitmap, BBoxData bboxData, bool showTarget)
    {
        if (bitmap == null) return null;
        
        try
        {
            using var renderBitmap = new RenderTargetBitmap(
                new PixelSize(bitmap.PixelSize.Width, bitmap.PixelSize.Height),
                new Vector(96, 96));
            
            using (var ctx = renderBitmap.CreateDrawingContext())
            {
                // 绘制原图
                ctx.DrawImage(bitmap, new Rect(0, 0, bitmap.PixelSize.Width, bitmap.PixelSize.Height));
                
                if (showTarget)
                {
                    // 绘制外框（红色）
                    var outerPen = new Pen(Brushes.Red, 2);
                    var outerRect = new Rect(bboxData.X1, bboxData.Y1, 
                                           bboxData.X2 - bboxData.X1, bboxData.Y2 - bboxData.Y1);
                    ctx.DrawRectangle(outerPen, outerRect);
                    
                    // 绘制内框（绿色）
                    var innerPen = new Pen(Brushes.Green, 2);
                    var innerRect = new Rect(bboxData.X1Inner, bboxData.Y1Inner, 
                                           bboxData.X2Inner - bboxData.X1Inner, 
                                           bboxData.Y2Inner - bboxData.Y1Inner);
                    ctx.DrawRectangle(innerPen, innerRect);
                }
            }
            
            return renderBitmap;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"绘制边界框失败: {ex.Message}");
            return bitmap;
        }
    }
}