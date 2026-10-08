# BRobbDetector Benchmark — Baseline

**Date:** 2026-10-08
**Git:** e243060
**Dataset:** CARS (500 images, seed=0)
**Runtime:** ~161s

## Configuration

| Parameter | Value |
|---|---|
| Method | `hole` |
| Filter | `none` |
| ColorSpace | `hsvLabRgb` |
| CloseMode | `iter1` |
| IoU threshold | 0.3 |

## Results

### Image-level (count match vs CSV expected)

| Metric | Value |
|---|---|
| Exact | **92.0%** (460/500) |
| Over | 19 |
| Under | 21 |

### Box-level (IoU ≥ 0.3 vs GT label files)

| Metric | Value |
|---|---|
| TP | 638 |
| FP | 431 |
| FN | 108 |
| Total GT | 746 |
| Total Det | 1069 |
| CSV expected | 1055 |
| **Recall (检出率)** | **85.5%** |
| **Precision (精确率)** | **59.7%** |
| **FPR (误检率)** | **40.3%** |
| F1 | 70.3% |
| Avg IoU (matched) | 0.761 |

## Notes

- CSV expected counts (1055) differ from actual GT label file boxes (746) — 178/500 images have mismatched counts. Image-level "exact" compares against CSV counts; box-level metrics compare against actual GT geometry.
- Pipeline: Step0 (binarize HSV+Lab+RGB) → Step1 (RemoveText + LineOpen) → Step2 (ApplyClose iter1) → Step3 (hole detection + edge validation + NMS + refine). LineClose is in Step1Filter for visualization only, NOT in the detection path.

## Usage

```bash
# Run benchmark and compare against baseline
python proto/eval_benchmark.py 500 --save myrun --compare baseline

# Reuse existing output (skip detection)
python proto/eval_benchmark.py 500 --no-run --compare baseline
```