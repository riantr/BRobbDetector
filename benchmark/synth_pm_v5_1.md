# Synthetic benchmark: synth_pm_v5_1

**Method**: pm_v5_1  
**Dataset**: 1000 images  
  - 50 unique X-ray backgrounds  
  - brokenness: [0.0, 0.05, 0.1, 0.2, 0.3]  
  - GT: minAreaRect of rendered red pixels (rebuilt per image)  
**Distance threshold**: D=5.0  

## Box-level metrics (D=5.0)

| TP | FP | FN | total_gt | total_det | recall | precision | f1 |
|----|----|----|----------|-----------|--------|-----------|----|
| 131 | 126 | 869 | 1000 | 257 | 13.1% | 51.0% | 20.8% |

## Per-brokenness recall (D=5.0)

| brokenness | n_gt | TP | recall | det | precision |
|------------|------|----|--------|-----|-----------|
| 0.00 | 200 | 121 | 60.5% | 200 | 60.5% |
| 0.05 | 200 | 10 | 5.0% | 57 | 17.5% |
| 0.10 | 200 | 0 | 0.0% | 0 | 0.0% |
| 0.20 | 200 | 0 | 0.0% | 0 | 0.0% |
| 0.30 | 200 | 0 | 0.0% | 0 | 0.0% |

## Multi-D performance

| D | TP | recall | precision | f1 |
|---|----|--------|-----------|----|
| 3 | 129 | 12.9% | 50.2% | 20.5% |
| 5 | 131 | 13.1% | 51.0% | 20.8% |
| 8 | 133 | 13.3% | 51.8% | 21.2% |
| 12 | 146 | 14.6% | 56.8% | 23.2% |
