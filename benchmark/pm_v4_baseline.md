# Point-Match Benchmark: pm_v4_baseline

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
| hybrid(h+e+m) | 489 | 494 | 507 | 514 |
| hybrid(c+e+m) | 479 | 491 | 508 | 517 |
| hull+approx | 387 | 389 | 395 | 395 |
| edgefit | 172 | 302 | 355 | 393 |
| canny(contour) | 371 | 378 | 388 | 389 |
| m8 | 93 | 176 | 235 | 284 |
| YOLO | 12 | 80 | 201 | 321 |
| YOLO+snap | 70 | 209 | 275 | 330 |
| combine+snap | 120 | 243 | 298 | 355 |

## Recall (%)

| Method | D=3 | D=5 | D=8 | D=12 |
|--------|------|------|------|------|
| hybrid(h+e+m) | 65.5% | 66.2% | 68.0% | 68.9% |
| hybrid(c+e+m) | 64.2% | 65.8% | 68.1% | 69.3% |
| hull+approx | 51.9% | 52.1% | 52.9% | 52.9% |
| edgefit | 23.1% | 40.5% | 47.6% | 52.7% |
| canny(contour) | 49.7% | 50.7% | 52.0% | 52.1% |
| m8 | 12.5% | 23.6% | 31.5% | 38.1% |
| YOLO | 1.6% | 10.7% | 26.9% | 43.0% |
| YOLO+snap | 9.4% | 28.0% | 36.9% | 44.2% |
| combine+snap | 16.1% | 32.6% | 39.9% | 47.6% |

## Key Findings

- **canny(contour) dominates legacy at D≤8**: D=3 371 vs 120 (3.1×), D=5 378 vs 243 (1.6×), D=8 388 vs 298 (1.3×)
- **hull+approx beats contour** (convex hull fixes concave dents): D=3 387 vs 371 (+16), D=5 389 vs 378 (+11)
- **hybrid(h+e+m) is best** (hull-primary union): D=3 TP 489 (corr_f1 70.0), D=5 TP 494 (corr_f1 70.6)
- hybrid(c+e+m) [contour-primary] D=3 TP 479 — hull-primary is better
- 3-axis orthogonal scope: mask variants (close/dilate) are byte-identical to raw; hole-ring and fragment-join proposals are dead (see benchmark/cv_scope3.json)
- **Best at D=5: hybrid(h+e+m) (494 TP, corr_f1 70.6)**
