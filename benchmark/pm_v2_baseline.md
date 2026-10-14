# Point-Match Benchmark: pm_v2_baseline

**500 images, 746 GT boxes**

## Configuration
- Snap window: 20px
- Snap passes: 2
- YOLO conf: 0.55
- Edge sampling: full (0.0-1.0)
- Line fit: Huber (nearest red per profile)

## Results

| Method | D=3 | D=5 | D=8 | D=12 |
|--------|------|------|------|------|
| canny(contour) | 371 | 378 | 388 | 389 |
| canny+m8 | 371 | 379 | 390 | 397 |
| m8 | 93 | 176 | 235 | 284 |
| YOLO | 12 | 80 | 201 | 321 |
| YOLO+snap | 70 | 209 | 275 | 330 |
| combine+snap | 120 | 243 | 298 | 355 |

## Recall (%)

| Method | D=3 | D=5 | D=8 | D=12 |
|--------|------|------|------|------|
| canny(contour) | 49.7% | 50.7% | 52.0% | 52.1% |
| canny+m8 | 49.7% | 50.8% | 52.3% | 53.2% |
| m8 | 12.5% | 23.6% | 31.5% | 38.1% |
| YOLO | 1.6% | 10.7% | 26.9% | 43.0% |
| YOLO+snap | 9.4% | 28.0% | 36.9% | 44.2% |
| combine+snap | 16.1% | 32.6% | 39.9% | 47.6% |

## Key Findings

- **canny(contour) dominates at D≤8**: D=3 371 vs 120 (3.1×), D=5 378 vs 243 (1.6×), D=8 388 vs 298 (1.3×)
- **GT corners sit on the OUTER contour of the red stroke**, not centerline — averaging outer+inner/2 drops D=3 TP 35→17
- **approxPolyDP 4-vertex cap is the biggest bottleneck** (227 candidate contours → 132 quads → 103 matched); adaptive eps list [0.01..0.18] recovers boxes
- **canny+m8 hybrid (canny-priority dedup) only adds +1 TP** but inflates FP — contour method alone is the winner
