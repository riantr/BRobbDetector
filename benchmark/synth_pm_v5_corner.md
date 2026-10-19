# Synthetic benchmark: synth_pm_v5_corner

**Method**: pm_v5  
**Dataset**: 1000 images  
  - 50 unique X-ray backgrounds  
  - brokenness: [0.0, 0.05, 0.1, 0.2, 0.3]  
  - GT: minAreaRect of rendered red pixels (rebuilt per image)  
**Distance threshold**: D=5.0  

## Box-level metrics (D=5.0)

| TP | FP | FN | total_gt | total_det | recall | precision | f1 |
|----|----|----|----------|-----------|--------|-----------|----|
| 383 | 253 | 617 | 1000 | 636 | 38.3% | 60.2% | 46.8% |

## Per-brokenness recall (D=5.0)

| brokenness | n_gt | TP | recall | det | precision |
|------------|------|----|--------|-----|-----------|
| 0.00 | 200 | 121 | 60.5% | 200 | 60.5% |
| 0.05 | 200 | 100 | 50.0% | 172 | 58.1% |
| 0.10 | 200 | 82 | 41.0% | 133 | 61.6% |
| 0.20 | 200 | 50 | 25.0% | 82 | 61.0% |
| 0.30 | 200 | 30 | 15.0% | 49 | 61.2% |

## Multi-D performance

| D | TP | recall | precision | f1 |
|---|----|--------|-----------|----|
| 3 | 378 | 37.8% | 59.4% | 46.2% |
| 5 | 383 | 38.3% | 60.2% | 46.8% |
| 8 | 389 | 38.9% | 61.2% | 47.6% |
| 12 | 409 | 40.9% | 64.3% | 50.0% |
