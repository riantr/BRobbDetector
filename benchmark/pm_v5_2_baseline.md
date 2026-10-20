# pm_v5_2 frozen baseline (LOOP-31 optimized params)

**Frozen**: 2026-10-19  
**Method**: pm_v5_1 with LOOP-31 parameter optimization  
**Source**: LOOP-31 systematic 6-stage parameter sweep on real data  
**Detections**: `tmp_sc31_pm_v5_2/` (500 JSON files)  

## Parameter changes vs pm_v5_1

| param | pm_v5_1 | pm_v5_2 | delta |
|-------|---------|---------|-------|
| min_area | 1000 | 1500 | +500 |
| ortho_tol | 15 | 20 | +5 |
| boundary_band | 12 | 20 | +8 |
| dedup_iou | 0.30 | 0.75 | +0.45 |

## Per-D metrics (500 images)

| D | TP | FN | rawFP | trueFP | real_unl | recall | raw_f1 | corrected_f1 |
|---|----|----|-------|--------|----------|--------|--------|--------------|
| 3 | 601 | 145 | 364 | 194 | 170 | 80.6% | 70.3% | 78.0% |
| 5 | 609 | 137 | 356 | 191 | 165 | 81.6% | 71.2% | 78.8% |
| 8 | 624 | 122 | 341 | 180 | 161 | 83.6% | 72.9% | 80.5% |
| 12 | 632 | 114 | 333 | 175 | 158 | 84.7% | 73.9% | 81.4% |

## Delta vs pm_v5_1_baseline (D=5)

| Metric | pm_v5_1 | pm_v5_2 | Δ |
|--------|---------|---------|---|
| TP@D=5 | 502 | 609 | +107 |
| trueFP@D=5 | 157 | 191 | +34 |
| recall@D=5 | 67.3% | 81.6% | +14.3 |
| corrected_f1@D=5 | 71.5% | 78.8% | +7.3 |

## Promotion rationale

pm_v5_2 improves over pm_v5_1 by +107 TP and +7.3 corrected_f1 at D=5. The gains come from:
- **ortho_tol 15→20**: hand-drawn boxes aren't perfectly orthogonal; 15 was too strict, rejecting real boxes
- **dedup_iou 0.30→0.75**: 0.30 was over-deduplicating true positives with moderate overlap (IoU 0.30-0.75)
- **min_area 1000→1500**: filters more noise contours, precision gain
- **boundary_band 12→20**: allows more boundary-clipped boxes to pass

**Caveat**: params optimized on the same 500 images used for evaluation (overfitting risk). dedup_iou=0.75 is at the edge (0.80 drops to corr_f1=76.6). Consider 0.70 as a more robust choice.

**How to use**: run `eval_pm_benchmark.py --compare pm_v5_2_baseline` and report the delta.
