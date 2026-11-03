# loop2 corpus-scaling probe, arm C: 2,000 -> 8,000 training images

Date: 2026-11-03. Artifact: `benchmark/loop2_scaling_8k.json`.
Harness: `proto/cs2_scaling_probe.py --train-imgs 8000 --work tmp_loop2_scale_8k`.

**Decision: DL-DEAD** (pre-registered rule: TP@D=5 must grow >= 1.4x from the 2,000-image arm,
i.e. >= 703 TP; measured **692 TP = 1.378x**). This closes the "more labelled images on the CPU
budget" lever; it does NOT close DL as a direction, because the loss is corner precision, not detection.

## What was held fixed

Only the training-image count varies, exactly as in the 300 -> 2,000 probe:

`yolov8n-obb` | 8 epochs | imgsz 384 | batch 8 | workers 2 | conf 0.55 | CPU only |
`RandomState(7)`, val = first 100 shuffled benchmark ids | labels from
`Label@CARS_202609_raw`, boundary-validated true GT. The 300-image arm is a strict subset of
the 2,000-image arm, which is a strict subset of the 8,000-image arm (same `RandomState(11)`
extra pool); the 100 validation images are identical in all three.

## Arms

| arm | train imgs | labels (train+val) | train min | D5 TP | D5 trueFP | D5 corr_f1 | val D5 TP | val f1 |
|---|---|---|---|---|---|---|---|---|
| A: 300 | 300 | 889 | 6.0 | 234 | 638 | 24.0 | 49 | 25.5 |
| B: 2,000 | 2,000 | 4,710 | 61.8 | 502 | 436 | 49.9 | 108 | 53.7 |
| C: 8,000 | 8,000 | 18,313 | 257.5 | **692** | 298 | 67.0 | **152** | 73.3 |

## Arm C per-D (same `D2.score`, true GT)

| D | bench TP | bench trueFP | bench corr_f1 | val TP | val trueFP | val corr_f1 |
|---|---|---|---|---|---|---|
| 3 | 415 | 552 | 40.6 | 86 | 103 | 42.5 |
| 5 | 692 | 298 | 67.0 | 152 | 47 | 73.3 |
| 8 | 824 | 179 | 79.3 | 181 | 24 | 86.0 |
| 12 | 895 | 111 | 86.0 | 189 | 17 | 89.6 |

## Scaling

| interval | images added | delta TP@D5 | TP per 1,000 added |
|---|---|---|---|
| 300 -> 2,000 | 1,700 | +268 | **157.6** |
| 2,000 -> 8,000 | 6,000 | +190 | **31.7** |
| 300 -> 8,000 | 7,700 | +458 | 59.5 |

Absolute TP per 1,000 training images: **780 -> 251 -> 86.5**.

Ratios: 2,000 -> 8,000 = **1.378x** (the pre-registered decision); 300 -> 8,000 = 2.957x
(continuity only - it inherits the steep 300 -> 2,000 segment and must not be used as the
stage decision).

## Decision

- Pre-registered threshold: 2,000 -> 8,000 must reach **1.4x = 703 TP@D=5**.
- Measured: **692 TP = 1.378x -> DL-DEAD**.
- The marginal slope collapsed from **157.6** to **31.7 TP per 1,000 images (-80%)**: adding
  26.7x data bought 2.96x TP, and the last 6,000 images bought only 1.38x. This is saturation,
  not a linear continuation.
- **Val-split caveat:** the 100-image val split gives 152 vs 108 = **1.407x**, just above 1.4,
  while the 500-image benchmark (the pre-registered unit) gives 1.378x. The arm sits on the
  knee. The decision is not rounded across the threshold: 692 < 703 is DL-DEAD, and the val
  signal is recorded so the knee is visible rather than hidden.

## Comparison to the shipped detector

| D | pm_v10 (classical) | arm C (8,000 DL) |
|---|---|---|
| 5 | 901 TP / 134 trueFP / f85.4 | 692 / 298 / f67.0 |
| 8 | 926 / 113 / f87.6 | 824 / 179 / f79.3 |
| 12 | 938 / 103 / f88.6 | 895 / 111 / f86.0 |

The DL arm remains below pm_v10 at every D. The gap narrows sharply at D=12 (895 vs 938), which
confirms the earlier diagnosis: the residual is **corner precision**, not detection. More data
at this budget does not close it.

## Recommendation

1. **Do not fund a large CPU run** and do not budget a "~4,500 images" run: the measured
   marginal return at 2,000 -> 8,000 is 31.7 TP/1,000, one fifth of the 300 -> 2,000 rate.
2. **Keep pm_v10 as the shipped detector.** DL is not a replacement at this operating point.
3. If DL is revisited, it must be a **GPU + corner-precision/architecture** lever (higher
   resolution, OBB-aware loss, or a corner-refinement head), not more corpus at 384 px.
4. Re-open the data-scaling question only with a 2,000 -> 8,000 arm that clears **1.4x on the
   pre-registered 500-image benchmark with a confidence interval**, not on the val split alone.

## Provenance and hygiene

- Checkpoint: `tmp_loop2_scale_8k/train/weights/best.pt`, 6.37 MB, mtime **2026-11-03 15:41:02**;
  training wall-clock **257.5 min**.
- Corpus: 8,000 train images, 18,313 labels (train+val), 890 MB under gitignored
  `tmp_loop2_scale_8k/`.
- Disk: **13.6 GB** free before the build (build-only run), **10.45 GB** after scoring.
- Split consistency: 300 train + 100 val -> **889 labels, MATCH** with the anchor.
- Git: the only tracked source change is `proto/cs2_scaling_probe.py` (the `--ref-json`,
  `--build-only`, `--skip-build` and reference-decision additions). `benchmark/loop2_scaling_8k.json`
  and this file are new. No `data/`, GT, label or detector file was modified; `tmp_*` is ignored;
  nothing was committed or pushed.

## Reproduction

```
python proto/cs2_scaling_probe.py --train-imgs 8000 --work tmp_loop2_scale_8k \
  --out benchmark/loop2_scaling_8k.json --build-only
python proto/cs2_scaling_probe.py --train-imgs 8000 --work tmp_loop2_scale_8k \
  --out benchmark/loop2_scaling_8k.json --skip-build
```