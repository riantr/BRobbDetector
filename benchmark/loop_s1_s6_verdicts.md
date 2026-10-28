# Loop verdicts — S1 → S6 (2026-10-26)

Goal: evaluate six named 4-corner-recall candidates against the frozen pm_v10
baseline, in the fixed order, and ship only the ones that strictly beat it.

**Result: no detector change was adopted. S1–S5 are DEAD. S6 delivered a
separate GT-proposal artifact (data only, contingent on human confirmation).**

## Attribution anchor (Done-when #1)

`benchmark/loop_baseline_check.json` — `ok = true`, `dets = 1070`, `n_gt = 1076`:

```
parity: py=1070 cs=1070 mismatched images=0
C# (consistent GT) ngt=1076 D=5 TP=901 rec=83.7 prec=87.1 trueFP=134 unl=35 corr_f1=85.4
```

| D | TP | trueFP | real_unl | recall | corr_f1 | matches pm_v10_refine.json |
|---|---|---|---|---|---|---|
| 3 | 857 | 177 | 36 | 79.6 | 81.2 | OK |
| 5 | 901 | 134 | 35 | 83.7 | 85.4 | OK |
| 8 | 926 | 113 | 31 | 86.1 | 87.6 | OK |
| 12 | 938 | 103 | 29 | 87.2 | 88.6 | OK |

Every S1–S6 harness carries a `base` arm that reproduces this column exactly
(`base-arm Δ = 0` in every row below), so all deltas are attributable.

## Master verdict table (one row per item, all required fields)

| item | hypothesis | harness → result | base Δ | D=3 TP/trueFP/corr_f1 | D=5 TP/trueFP/corr_f1 | D=8 TP/trueFP/corr_f1 | D=12 TP/trueFP/corr_f1 | gain/loss @D=5 | CATE input | held-out seed 1 (ΔTP/ΔtrueFP/Δf1 @D=5) | held-out seed 2 (ΔTP/ΔtrueFP/Δf1 @D=5) | verdict | numeric reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | **Inward** cut-side completion: for a covering hole LARGER than the box, slide a side flagged "cut" (red continues ≥5 px past BOTH corners) TOWARD the quad centre onto the next parallel stroke, re-intersect | `proto/s1_inward_complete.py` → `benchmark/s1_inward.json` (variant `in060`) | 0 | 840/179/80.2 | 893/130/85.1 | 918/109/87.3 | 931/99/88.4 | **0 / 8** | n/a | −10 / −5 / −0.4 | −5 / −8 / ±0.0 | **DEAD** | 185 cut sides flagged → 181 replaced → 125 quads changed, yet **0 new TP and 8 destroyed**; ΔTP@D5 −8, Δcorr_f1 −0.3 |
| **S2** | Small/thin specialist second pass fused by **union** (min_area 300→120, min_side 12→6, max_aspect 12→20, hole_min_side 14→8, hole_fill 0.55→0.50, refine 1→0), dedup at the shipped IoU | `proto/s2_small_specialist.py` → `benchmark/s2_small_specialist.json` (variant `union_40`) | 0 | 857/728/64.4 | 901/685/67.7 | 926/664/69.5 | 938/654/70.3 | **0 / 0** (specialist alone 4 / 6) | `fit_pass` earns its TP mainly on min_side ≤ 50 (+0.011 TP/img, p=0.067); `refine` is −0.016/img on min_side ≤ 30 (n=47) | +3 / +602 / −17.2 | +6 / +568 / −16.4 | **DEAD** | Union adds **0 TP** and **+551 trueFP** (dets 1070→1640, corr_f1 67.7); the specialist alone nets −2 TP (4 gains / 6 losses) and floods the candidate set |
| **S3** | Support-priority dedup: when two quads overlap ≥ dedup_iou, keep the higher-red-support quad instead of the first-emitted; dedup_iou fixed at 0.40 | `proto/s3_support_dedup.py` → `benchmark/s3_support_dedup.json` (variant `sup`) | 0 | 858/176/81.3 | 901/134/85.4 | 926/113/87.6 | 938/103/88.6 | **0 / 0** | n/a (threshold deliberately not retuned) | +1 / +0 / ±0.0 | +0 / +0 / ±0.0 | **DEAD** | ΔTP@D5 **+0**, ΔtrueFP +0, dets unchanged at 1070 — on the shipped candidate set the ordering never changes the survivors; the +1 on seed 1 does not reproduce |
| **S4** | Robust in-band pixel **SELECTION**: per cross-section keep only the red-run core nearest the side line (nearest-run + ±1 px), discarding a second stroke ≥2 px away — selection, not re-weighting | `proto/s4_run_select.py` → `benchmark/s4_run_select.json` (variant `runs3`) | 0 | 855/174/81.2 | 901/129/85.6 | 923/111/87.5 | 936/100/88.6 | **4 / 4** | n/a | −1 / +1 / −0.1 | +0 / −3 / +0.1 | **DEAD** | TP@D5 **+0** (the gate is TP-primary), −2 @D3, −3 @D8; the 4 gains are exactly cancelled by 4 losses (churn); only trueFP −5 / corr_f1 +0.2, and that sign does not survive held-out |
| **S5** | YOLO-OBB as a **proposer** + geometric side refit (`_refine_sides` then nearest-run, re-enforce orthogonality, shipped gates) — not a raw union, not crop re-detection | `proto/s5_yolo_refine.py` → `benchmark/s5_yolo_refine.json` (variant `yunion`) | 0 | 857/177/81.2 | 901/134/85.4 | 926/113/87.6 | 938/103/88.6 | **0 / 0** | n/a | +0 / +1 / −0.1 | +1 / +3 / −0.1 | **DEAD** | 936 proposals (conf 0.55) → the gates accept only **311**; `yref` alone is 263 TP (**−638**, gains 3 / losses 641); the refined union adds **0 TP** (dets 1070→1071). The binding constraint is the validity gate, not the proposal source |
| **S6** | GT completion: derive tiered candidate GT additions from the `real_unl` detections at D=5 | `proto/s6_gt_completion.py` → `benchmark/s6_extended_gt.json` | 0 | — (data artifact) | — | — | — | n/a | n/a | n/a (no detector change) | n/a | **DELIVERED — not a detector WIN** | 35 candidates (tier A 23 / B 8 / C 4); against the extended GT the plain f1 goes 84.0 → **85.2 (A) / 85.6 (A+B) / 85.8 (A+B+C)**. Upper bound and circular until a human confirms each box; the 1076-box GT is untouched |

## Final summary (Done-when #7)

| item | ΔTP@D=5 | ΔtrueFP@D=5 | Δcorr_f1@D=5 | shipped detector changed? |
|---|---|---|---|---|
| S1 inward cut-side completion | **−8** | −4 | −0.3 | **no** |
| S2 small/thin specialist union | **+0** | +551 | −17.7 | **no** |
| S3 support-priority dedup | **+0** | +0 | ±0.0 | **no** |
| S4 nearest-run pixel selection | **+0** | −5 | +0.2 | **no** |
| S5 YOLO proposer + geometric refit | **+0** | +0 | ±0.0 | **no** |
| S6 GT completion | n/a — data artifact | n/a | n/a | **no** |
| **shipped detector** | **baseline 901 TP / 134 trueFP / 85.4** | — | — | **UNCHANGED** |

**No candidate strictly beat the frozen baseline at D=5, so nothing was shipped.**
`benchmark/pm_v10_refine.json`, `proto/scope45_holes.py`, `BRobbDetector/` and
`data/` are all unmodified; `git status` shows only new untracked artifacts.
Done-when #5 is therefore vacuously satisfied (no adopted item ⇒ no C# port, no
new frozen baseline, no README/AGENTS change).

## Supporting measurement — stratified CATE (Done-when #2)

`benchmark/dml_cate_strata.json`, produced by `proto/dml_cate_strata.py` from the
**existing** `benchmark/dml_panel_real.json` (500 images × 16 cells) with **no new
detection runs**. Estimator: exact within-image paired contrasts (complete block),
SE over images, normal-approximation p. Strata: `gt_min_side_mean ≤ 30` and `≤ 50`
(the measured 47% / 23% miss bands), `gt_aspect_mean ≥ 6` and `≥ 2.5`, `n_gt ≥ 3`
(dense clusters), `n_holes > median` (local red complexity).

- **`hole_min_side` 8→14: θ = 0.000 TP/image in EVERY stratum** — its "pure FP
  filter, TP effect exactly 0" property is stratum-invariant; there is no hidden
  small-box TP cost.
- `hole_fit_pass` 0→2: aggregate +0.004; concentrated at min_side ≤ 50 (+0.011,
  p=0.067) and in dense images (+0.014, p=0.169).
- `hole_refine` 0→1: aggregate +0.004, but **−0.016 on min_side ≤ 30** (n=47,
  p=0.47) vs +0.009 on the larger stratum — a sign flip in the smallest boxes.
- `min_red_support` 0.5→0.6: negative in every stratum (worst in small and dense).
- **No sign flip with |θ| ≥ 1 TP/image anywhere** — the panel holds no hidden
  large conditional effect, which is why S2's union gained exactly 0 TP.

(The `gt_aspect_mean ≥ 6` bin is degenerate at image level — only 2 images — because
the panel stores a per-image mean; `≥ 2.5` is the usable aspect bin, 241 images.)

## Plateau note (goal stop-if clause)

S2, S3 and S4 are three consecutive items with **|ΔTP@D=5| ≤ 2**. Per the stop-if
clause no further variants were added to those items; the sequence continued to S5
and S6 as the goal's Scope requires.

## Mechanistic conclusion

S1 (inward) and P4 (outward) are now **symmetric dead ends**: flagging a "cut" side
and sliding it in either direction never converts a miss into a hit (outward gains
0 / losses 15; inward gains 0 / losses 8). The cut-side signal detects *that* a
stroke crosses, but not *which* parallel stroke is the box's own.

S5 explains why the highest-recall proposal source does not help: the boxes YOLO
finds that pm_v10 misses are exactly the ones that fail the shipped validity gates
(red support, edge rule, orthogonality) — only 311 of 936 proposals pass, and the 3
unique gains are absorbed by dedup. **The gate is the binding constraint on those
boxes, not the proposal source.**

S4 is the only TP-neutral precision gain (trueFP −5, corr_f1 +0.2) and it fails the
TP-primary gate; it is recorded as the one lever measured to trade in the right
direction without a TP cost.

## Artifacts

`benchmark/loop_baseline_check.json`, `benchmark/dml_cate_strata.json`,
`benchmark/s1_inward.json`, `benchmark/s2_small_specialist.json`,
`benchmark/s3_support_dedup.json`, `benchmark/s4_run_select.json`,
`benchmark/s5_yolo_refine.json`, `benchmark/s6_extended_gt.json`,
`benchmark/s_heldout.json`.

Harnesses: `proto/s0_baseline_check.py`, `proto/dml_cate_strata.py`,
`proto/s1_inward_complete.py`, `proto/s2_small_specialist.py`,
`proto/s3_support_dedup.py`, `proto/s4_run_select.py`, `proto/s5_yolo_refine.py`,
`proto/s6_gt_completion.py`, `proto/s_heldout_check.py`.

**No detector code, no GT file and no frozen baseline was modified. Nothing was
committed or pushed.**