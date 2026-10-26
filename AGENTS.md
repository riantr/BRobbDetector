# BRobbDetector

## Build & Run

```bash
# Build (Debug)
dotnet build BRobbDetector/BRobbDetector.csproj -c Debug

# Build (Release)
dotnet build BRobbDetector/BRobbDetector.csproj -c Release

# Run GUI
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug

# CLI detection
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug -- --cli <inputDir> <outputDir> [--method cvhybrid]

# Self-tests
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug -- --selftest-ortho
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug -- --selftest-refine

# Checker
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Debug -- --check <imageDir> <labelDir>
```

## Tech Stack
- .NET 10, Avalonia 11.2.8 (MVVM), OpenCvSharp5 5.0.0.20261018 (OpenCV 5 native)
- Python RapidOCR sidecar for OCR (`proto/ocr_sidecar.py`)

## Ground Truth
- **Benchmark GT = `E:\Xray\CARS\data\Label@CARS_202609_raw`** (`proto/common.py::LAB`).
  This is the hand-drawn annotation set: its geometry is literally the red
  rectangle burned into `Image@CARS_202609_annotated`, the detector's input
  (97.6% of raw boxes have >=0.8 red-line support in the input image; the
  machine-artifact candidates Compare-V1 / Compare / detector output score
  only ~49%). Evidence: `benchmark/gt_final.json`, `proto/gt_final.py`.
- The raw label files carry a **UTF-8 BOM** — read them with `encoding="utf-8-sig"`.
- `Label@CARS_202609_annotated` holds **detector output**, not GT — never read it as ground truth.
- `Compare-V1@CARS_202609_annotated` / `Compare@CARS_202609_annotated` are machine
  artifacts, not GT (the pre-2026-10-20 baselines were measured against them and
  are therefore invalid).
- Current C# baseline (`cvhybrid` = **holes + edgefit + refine**, 500 imgs):
  **D=3 857 TP / corr_f1 81.2, D=5 901 / 85.4, D=8 926 / 87.6, D=12 938 / 88.6**
  (`benchmark/pm_v10_refine.json`). 1070 detections, trueFP 134, real_unl 35 at D=5.
  Scored against the **rule-consistent GT** (1076 boxes: the 45 hand-annotated
  boxes that the image edge clips are not valid rectangles either, so they are
  not valid targets — `proto/scope78_parity.py`).
  Previous baseline (pm_v9, `benchmark/pm_v9_edge_rule.json`): D=3 842 / 79.4,
  **D=5 899 / 84.7**, D=8 922 / 86.7, D=12 937 / 87.9; 1102 detections,
  trueFP 149, real_unl 54 at D=5.  The pm_v10 change is a strict win at every
  D (D=3 +15 TP / -27 trueFP, D=5 +2 / -15, D=8 +4 / -16, D=12 +1 / -15).
  Previous baseline (pm_v8, no edge rule, raw GT 1121 boxes,
  `benchmark/pm_v8_csharp_holes.json`): D=3 859 / 77.5, **D=5 926 / 83.5**,
  D=8 960 / 86.4, D=12 978 / 88.0; 1148 detections, trueFP 172, real_unl 50.
  Previous baseline (edgefit only, `benchmark/pm_v7_csharp_rawgt.json`):
  D=3 579 / 66.2, D=5 602 / 68.8, D=8 605 / 69.1, D=12 607 / 69.4.
  Config: hue<=4, sat>=100, val>=140, hole family (min_side **14**, fill 0.55,
  band 3, pts 12, expand 2.7, **fit_pass 2 / fit_tol 1.5**, **refine 1 /
  refine_band 3.0 / refine_tol 3.0 / refine_pts 8**) + edgefit family
  (min_area 300, min_side 12, fit_band 8), ortho_tol 10, edge_band 0,
  dedup_iou 0.40, min_red_support 0.5
  (`proto/scope45_holes.py`, ported to `BRobbDetector/Services/HoleDetector.cs`).
  The C# port is **bit-exact** with the Python prototype (all 1070 boxes, same order).
- **pm_v10 tuning (2026-10-23)** — three changes that each attack the
  "leader arrow pulls a box corner" failure mode:
  1. `hole_min_side` 8 -> 14: the hole-level profile showed FP holes have
     p25 min_side 17.2 vs p50 52.0 for TP holes (`benchmark/real_hole_features.json`).
  2. `hole_fit_pass=2` (`hole_fit_tol=1.5`): robust TLS on each side — the band
     outside a hole side can hold another stroke, which drags the fit.
  3. `hole_refine=1, hole_refine_band=3.0, hole_refine_tol=3.0`: re-fit the 4
     sides on the red stroke in a SYMMETRIC band around the side line and
     re-intersect them — the hole quad inherits the minAreaRect error otherwise.
  Swept on real data with `proto/sweep_real.py` (~200 variants, 6 s each).
- **Held-out validation of pm_v10** — `proto/sweep_real.py --seed N` draws a
  *different* 500-image sample from the same pool (56832 images; seed 0 is the
  tuning set).  pm_v10 beats pm_v9 on both held-out samples, so the win is not
  overfitted:
  | sample | pm_v9 D=3 / D=5 | pm_v10 D=3 / D=5 |
  |---|---|---|
  | seed 1 (1184 GT) | 878 tp/256 fp, f75.8 / 946/190, f81.6 | **890/213, f77.8** / **950/156, f83.0** |
  | seed 2 (1150 GT) | 864/222, f77.3 / 914/178, f81.5 | **865/200, f78.1** / 911/161, f82.0 |
  `min_red_support=0.6` was **NOT** shipped: on held-out data it is a wash
  (seed 1: 950/156 -> 950/153; seed 2: 911/161 -> 910/154, i.e. −1 TP), and
  on the pm_v9 config it *loses* TP (946 -> 939), so the small gain it shows on
  the tuning set is an interaction with pm_v10, not a general effect.
  Results: `benchmark/sweep_real_holdout_s{1,2}.json`.
- **Emulation benchmark** (`data/Image@emul4`, `data/Label@emul4`, 5000 synthetic
  images / 9982 GT boxes) — the aggregate metrics transfer to the real data
  (emul D=5 rec 84.2 / prec 87.4 vs real 83.6 / 85.8) and it CONFIRMED pm_v10:
  D=5 TP 2773 -> 2807, FP 400 -> 359; D=3 TP 2097 -> 2544, FP 1076 -> 622.
  **But the red-support thresholds do NOT transfer**: emulation says
  `min_red_support` 0.5->0.7 is +115 TP / -256 FP, on real data it is
  monotonically harmful (D=5 TP 899 -> 885).  The reason is in the support
  distribution of the emitted detections (`proto/support_profile.py`,
  `benchmark/support_profile_{real,emul}.json`): real TP support p50 0.996 /
  real FP support p50 0.90 (the FPs are arrow-corrupted *real* annotations, so
  they are just as red as the TPs and the threshold cannot separate them),
  whereas emul FP support p50 is only 0.671.  Use the emulation for structural
  params (fits, min_side, refine) and the real data for red thresholds.
  Harness: `proto/sweep_emulation.py`; results `benchmark/sweep_emul4_r3*.json`.
  Real-data harness: `proto/sweep_real.py`; results `benchmark/sweep_real_r*.json`.
- **Causal decomposition of pm_v10 (DoubleML, 2026-10-23)** — a 2^4 complete-block
  factorial panel (every image run through all 16 configs of `hole_min_side`
  8/14 x `hole_fit_pass` 0/2 x `hole_refine` 0/1 x `min_red_support` .5/.6)
  estimated with DoubleML PLR (RF nuisance, cluster-robust SE on the image),
  cross-checked against exact paired design contrasts.  Results:
  `benchmark/dml_panel_{real,emul}.json`, `benchmark/dml_{real,emul}.json`.
  Key causal facts (500 real images, DML single-treatment effect):
  1. `hole_min_side=14` is a **pure FP filter** — the paired effect on TP@D=5 is
     exactly 0.0000 (se 0.0000); it removes arrow-corrupted FP holes and
     recovers zero true positives.  FP@D=5 -7.6 (p=.015).
  2. `hole_fit_pass=2` and `hole_refine=1` are the **TP gainers** and they
     interact: TP@D=3 +7.1 (p=.026) and +10.6 (p=.065) alone but +16 together
     (sub-additive, interaction -3.5).  At D=5 the gain appears only when BOTH
     are on (conditional effect of fit_pass +1.5 with refine=0, +2.5 with
     refine=1) — they are partly substitutes, you need both.
  3. `min_red_support=0.6` is a genuine **trade-off whose TP cost is
     conditional**: FP@D=5 -6.7 (p=.0019) but the TP@D=5 effect in the four
     joint strata of (fit_pass, refine) is -4.0 (both off), -1.0 (one on),
     -1.0 (the other on), **0.0 (both on)**; its FP benefit shrinks from -6 FP
     (geometry off) to -4 FP (geometry on).  This causally explains "0.6 helps
     under pm_v10 but hurts under pm_v9" and why the held-out check rejected
     it: it is an **interaction**, not a general effect.  The red threshold is
     only safe once the geometry is fixed.
  4. The FP effects are **sub-additive** (the knobs kill overlapping FPs): the
     four DML main effects sum to -22.3 FP@D=5 while the global pm_v9->pm_v10
     contrast is only -15.  For TP the sum matched the global contrast at D=3
     (+15.7 vs +15).
- **Box-level DoubleML (real, 1076 GT boxes, `proto/dml_boxlevel.py`)** — binary
  outcome = detected@D=5, controls = image-level covariates; results
  `benchmark/dml_boxlevel_{panel,real}.json`.  What causally drives a miss:
  `box_sup_min` (min per-side red coverage) +0.494/unit (p<1e-5, +0.147/SD) is
  dominant, `box_min_side` +0.0093/px (p=0.0003, +0.144/SD) second,
  `hole_fill` +0.382 (p=0.0002); `hole_ms` **-0.0039/px** (p=0.0027) — a
  *larger* covering hole means a merged/multi-box region, so less likely
  detected.  `box_area` and `box_edge_dist` carry no independent signal.
- **Real vs emulation, causally (same 2^4 panel, 2500 emulation images):**
  `min_red_support` .5->.6 is **-1.0 TP on the real data (p=.43) but +23.6 TP
  on the emulation (p=7e-8)** — opposite signs in *every* stratum (with the
  geometry off: -4 real vs +27 emul; with it fixed: 0 real vs +21 emul),
  confirming that the red threshold does not transfer.  The structural knobs
  keep their sign (refine: +2.0 real / +22.5 emul at D=5; at D=3 refine is
  +209.5 emul vs +10.6 real) but not their magnitude (fit_pass: +1.8 real /
  -2.4 emul).  **Gate red thresholds on the real 500-image set only.**
- **Rectangle validity rule (2026-10-21):** a rectangle is valid only when it has
  exactly 4 edges, every corner is orthogonal within `ortho_tol`, **and no vertex
  lies on the image edge** (`edge_band` = 0 = the outermost pixel ring). The old
  behaviour (waive the orthogonality check at a boundary corner) is gone. The check
  runs on the float quad in the gate *and* on the emitted (rounded) corners.
  Single source of truth: `proto/quad_ortho.py::is_orthogonal_quad` +
  `BRobbDetector/Services/GeometryValidation.cs::IsOrthogonalQuad`.
  Band sweep (post-hoc, 500 imgs): band 0 -> corr_f1 84.7, band 1 -> 84.6,
  band 2 -> 84.5, band 6 -> 84.4, no rule -> 83.5 (`proto/scope76_edge_consistent.py`).
- **Ceiling:** the remaining 177 FN at D=5 are mostly *not drawn in red*.
  GT boxes with red-line support >=0.6: 707 boxes, 699 detected (98.9%).
  FN GTs have a median min-side red support of 0.23 vs 0.85 for TPs.
  Realistic remaining headroom is ~61 boxes, not 177 (`proto/scope73_ceiling.py`).
- The detector's input has the human annotation burned in as red rectangles, so the
  red mask must be narrow: the annotation is hue 0-4, the orange X-ray object texture
  is hue 9-10. A wide hue window floods the mask with object pixels and was the
  single largest error source.
- **Legacy / external algorithm collections: all dead (LOOP-43 2026-10-23,
  LOOP-44 2026-10-26).** Every algorithm from `D:\src\java\algo\Segmentation`
  (Multi-Otsu, Otsu, Mixture-model, SIOX, RATS, Watershed, Blob_Labeler,
  Densitometry, RetFM-J) and both OptiNav plugins (`Find_Ridges.java` = a
  Steger-style ridge detector; `GCNR_.java` = the generalized contrast-to-noise
  ratio, Rodriguez-Molares et al., IEEE TUFFC 67(4):745-759, 2020) was ported
  and measured on the real 500-image set.  Every mask-level transfer is
  catastrophic (D=5 TP 901 -> 0..451 with FP 169 -> 1103..1768; the ridge map
  floods: 132,490 px / in-band precision 0.031 / 1228 holes vs 3,383 / 0.591 /
  19.9 for the shipped hue mask), and the graded gate scores are no better than
  the binary red support (Densitometry AUC 0.56, gCNR 0.751 vs **0.721** for
  `_red_support`, but with no better operating point on the corrected metric).
  Structural conclusions: Find Ridges is dead as a mask, as a support-gate
  source (-6 TP) and as a side refit (-9..-11 TP); SIOX-as-quad-source adds
  nothing (union byte-identical to pm_v10); Blob_Labeler 4-connected labelling
  is dead (-10 TP) and its "edge correction" is a no-op; gCNR is dead as a gate.
  The one non-obvious measurement worth keeping: on the **redness channel
  `R - max(G,B)`** the ridge argmax IS a stroke-centre locator (within 1 px of
  the GT line in 81.6% of cross-sections, bias +0.04 px), yet refining the side
  lines on it still loses TP because the shipped unweighted TLS fit plus the
  ortho gate is already better.  Root cause for all of it: separating the red
  annotation from the orange X-ray texture is a COLOUR problem, already solved
  by the hue<=4 window; a grey-level or single-channel shape filter cannot do it.
  Harnesses: `proto/seg_transfer_eval.py`, `proto/optinav_transfer_eval.py`;
  results `benchmark/seg_transfer_*.json`, `benchmark/optinav_*.json`.
- **OpenCV hierarchy gotcha:** hole siblings are chained through `hier[i][0]`
  (next), not `hier[i][2]` (first child). Walking first-child only yields the first
  hole of a component and cost 264 TP in the prototype. OpenCvSharp's
  `HierarchyIndex` has no `FirstChild` — the first child is always at index
  `parent + 1`.

## Project Structure
- `BRobbDetector/` — C# project (main)
  - `Services/HoleDetector.cs` — hole family (the accurate one), port of `proto/scope45_holes.py`
  - `Services/CvHybridDetector.cs` — pm_v10 pipeline (holes + edgefit + side refine + ortho_edge + edge rule + dedup)
  - `Services/GeometryValidation.cs` — the validity rule (4 edges, right angles, no vertex on the image edge)
- `proto/` — Python prototype + OCR sidecar
  - `sweep_real.py` — real-data parameter sweep (500 imgs, ~6 s/variant) — the gate for any change
  - `sweep_emulation.py` — emulation-dataset sweep (5000 imgs, ~50 s/variant) — the fast proxy for structural params
  - `dml_panel.py` — 2^4 complete-block factorial panel builder (real/emul) for the DoubleML decomposition
  - `dml_analyze.py` — DoubleML PLR analysis of a panel (single/multi/stratified, design contrasts, blocked OLS, CATE)
  - `dml_boxlevel.py` — box-level DoubleML: what causally drives a missed GT box
  - `dml_patch_conditional.py` — back-fill the reverse/joint conditional effects into an existing `benchmark/dml_*.json` (no need to re-run the slow DML fits)
- `data/` — test images and labels