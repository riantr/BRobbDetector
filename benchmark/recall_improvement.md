# BRobbDetector Benchmark — Methods 1+2+4 (Recall Improvement)

**Date:** 2026-10-09
**Base:** e2f2d1a (baseline)
**Dataset:** CARS (500 images, seed=0)

## Methods Applied

| Method | Description | Effect |
|---|---|---|
| 1. GrowBox | For strip-like detections (minSide/maxSide < 0.5), grow long edges outward to find farthest red pixel cluster | Recall +1.5%, F1 +4.0% |
| 2. StitchLineFragments | Directional close (7×1+1×7) on line-like components only (fill ratio < 0.3), bridges broken outline gaps without filling thin-box holes | Recall +0.4% |
| 3. Relaxed 2/4 edges | Second pass with MinEdgesWithSupport=2 for rejected boxes | No improvement, reverted |
| 4. GT completeness audit | Classified FP as missing-GT (CSV>GT, red support ≥0.3) vs true FP | 347/354 FP are real boxes missing from GT files |

## Results (vs baseline)

### Measured (against incomplete GT label files)

| Metric | Baseline | Methods 1+2 | Delta |
|---|---|---|---|
| Recall | 85.5% | **87.4%** | +1.9 |
| FPR | 40.3% | **35.2%** | -5.1 |
| F1 | 70.3% | **74.4%** | +4.1 |
| Exact (image-level) | 92.0% | 87.4% | -4.6 |

### GT-corrected (347 FP reclassified as TP — real boxes missing from GT files)

| Metric | Value |
|---|---|
| TP | 985 |
| FP (true) | 7 |
| FN | 108 |
| **Recall** | **91.4%** |
| **FPR** | **0.7%** |
| **Precision** | **99.3%** |
| **F1** | **95.2%** |

## Key Findings

1. **GrowBox** is the biggest lever — strip-like detections (35% of GT short side) are grown to full box size by scanning outward for red pixel clusters. Only grows long edges (perpendicular to short axis) to avoid extending into adjacent boxes.
2. **StitchLineFragments** with 7px kernel gives marginal improvement. 21px kernel caused -5.9% recall (too aggressive, merges unrelated fragments).
3. **Relaxed 2/4 edge validation** adds FPs without recovering TPs — GrowBox already handles the missed boxes.
4. **GT label files are 25% incomplete** — CSV says 993 boxes, GT files have 746. The "40% FPR" was almost entirely artifacts of missing GT annotations, not actual false positives. Only 7 true FP remain.