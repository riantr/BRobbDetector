# pm_v5_1 frozen baseline (min_area=1000)

**Frozen**: 2026-10-15  
**Method**: pm_v5 hybrid(h+e+m) + admission filter, **min_area=1000** (vs pm_v5's 900)  
**Source**: LOOP-15c axis sweep refinement; LOOP-18 real validation confirmed  
**Detections**: `tmp_sc6_hy_o1000_ba/` (500 JSON files)  

## Per-D metrics (500 images)

| D | TP | FN | rawFP | trueFP | real_unl | recall | raw_f1 | corrected_f1 |
|---|----|----|-------|--------|----------|--------|--------|--------------|
| 3 | 490 | 256 | 305 | 162 | 143 | 65.7% | 63.6% | 70.1% |
| 5 | 502 | 244 | 293 | 157 | 136 | 67.3% | 65.2% | 71.5% |
| 8 | 517 | 229 | 278 | 146 | 132 | 69.3% | 67.1% | 73.4% |
| 12 | 524 | 222 | 271 | 141 | 130 | 70.2% | 68.0% | 74.3% |

## Delta vs pm_v5_baseline (D=5)

| Metric | pm_v5 | pm_v5_1 | Δ |
|--------|-------|---------|---|
| min_area | 900 | 1000 | +100 |
| TP@D=5 | 502 | 502 | +0 |
| trueFP@D=5 | 161 | 157 | -4 |
| recall@D=5 | 67.3% | 67.3% | +0.0 |
| corrected_f1@D=5 | 71.3% | 71.5% | +0.2 |

## Promotion rationale

pm_v5_1 is a clean precision improvement over pm_v5: identical TP, +0.2 corrected_f1, -4 trueFP. No TP is lost; the gain comes from filtering 4 more small-red-noise false positives. The improvement grows at relaxed thresholds (+0.6 corrected_f1 at D=12, +0.5 at D=8), confirming the precision gain is real and consistent.

**How to use**: When proposing a new detection method, run `eval_pm_benchmark.py --compare pm_v5_1_baseline` and report the delta. Both pm_v5 and pm_v5_1 are now frozen reference points.
