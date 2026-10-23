# BRobbDetector

Detects hand-drawn red annotation boxes (rotated rectangles / OBB) in X-ray security images using a classical-CV pipeline. Built with .NET 10 + Avalonia MVVM, with a Python prototype/benchmark harness.

## Algorithm (cvhybrid, pm_v10 — holes + edgefit + side refine)

A classical-CV pipeline operating on the HSV red mask, with **two** quad families:

```
red_mask → denoise ─┬→ holes  → hole_quads  ─→ side_refine ─┐
                    │                                          ├→ admission_filter → ortho_enforce → dedup
                    └→ outer_contours → edgefit_quads ────────┘
```

| Stage | Operation | Key params |
|-------|-----------|-----------|
| 1. `red_mask` | HSV threshold (S≥100, V≥140, H≤4) → binary mask | smin=100, vmin=140, hmax=4 |
| 2. `denoise` | Drop connected components < min_area | denoise_min_area=40 |
| 3. `holes` | `findContours` RETR_CCOMP → the child contours of every stroke component | hole_min_comp_area=60 |
| 4. `hole_quad` | minAreaRect of the hole + per-side TLS fit on the red band just outside the hole edge → intersect the 4 lines | hole_min_side=14, hole_fill=0.55, hole_band=3, hole_pts=12, hole_expand=2.7, **hole_fit_pass=2, hole_fit_tol=1.5** |
| 4b. `side_refine` | Re-fit the 4 sides on the red stroke in a **symmetric** band around each side line → re-intersect | **hole_refine=1, hole_refine_band=3.0, hole_refine_tol=3.0, hole_refine_pts=8** |
| 5. `outer_contours` | findContours RETR_CCOMP + area filter (edgefit family) | min_area=300 |
| 6. `edgefit` | sector partition + TLS line fit → line intersection | fit_band=8, fit_min_pts=12, fit_jump=20 |
| 7. `admission_filter` | Side + orthogonality check + **no vertex on the image edge** (on the **float** quad) | ortho_tol=10, edge_band=0, min_side=12, max_aspect=12 |
| 8. `ortho_enforce` | `ortho_edge`: keep edge midpoints, average edge directions, re-intersect | — |
| 8b. `edge_rule` | The **emitted** (rounded) corners are re-checked: a vertex on the image edge invalidates the rectangle | edge_band=0 |
| 9. `red_support` | Hole family only: ≥ 0.5 of the perimeter must be red | min_red_support=0.5 |
| 10. `dedup` | Single keep-order IoU dedup, holes first (they win ties) | dedup_iou=0.40 |

### Stage 1 — the hue window is the critical parameter

The detector's input image has the human annotation burned in as red rectangles.
The annotation strokes are pure red (hue 0-4 for ~85% of their pixels), while the
orange/red X-ray object texture sits at hue 9-10. A wide hue window floods the mask
with object pixels — this was the single largest error source. Against the true GT,
hue≤10 scores D=5 335 TP / 42.6 corr_f1 and hue≤5 scores 592 / 67.9. Narrowing to
hue≤4 ties at D=5 but gives the best D=3 score (555 / 63.7 vs 547 / 62.8); combined
with the other re-tuned axes it reaches D=5 601 / 68.7.

### Stage 3/4 — the hole family is what carries the accuracy

A hand-drawn red rectangle encloses a **hole**: the region inside the stroke. The
interior of a box is background, so `findContours` with `RETR_CCOMP` reports it as a
child contour of the stroke's component. Two boxes that merely touch keep their
interiors separate, so the holes give the boxes directly — no component splitting
needed (the old edgefit family could only emit **one** quad per red blob, which is
what caused the "merged" failure mode below).

The hole boundary is the *inner* edge of the stroke while the GT side line runs
through the stroke *centre*, so every hole side is refined by a total-least-squares
fit over the red band just outside the hole edge and the four fitted lines are
intersected.

Two refinements were added in 2026-10-23 (pm_v10) and both attack the same
failure mode — the annotation's **leader arrow pulls a box corner**:

* **`hole_fit_pass=2`** — the band outside a hole side can hold *another* stroke
  (a crossing stroke, a neighbouring box, red text), and those outliers drag the
  total-least-squares fit onto the wrong line. The fit is repeated with the
  outliers (residual > 1.5 px) removed.
* **`hole_refine=1, hole_refine_band=3.0`** — the hole quad is built from the
  hole's `minAreaRect`, so its orientation and side offsets inherit that fit's
  error. A second pass re-fits each side on the red pixels in a *symmetric*
  band centred on the side line (the earlier band was one-sided) and
  re-intersects the four lines. A band whose 90th-percentile residual exceeds
  3 px is not a single stroke → the side is rejected and the hole quad is kept.

> **OpenCV hierarchy gotcha:** the siblings of a hole are chained through
> `hier[i][0]` (next). Walking `hier[i][2]` (first child) would only ever yield the
> *first* hole of a component — getting this wrong cost **264 TP** in the prototype.

### Stage 4 — edgefit stays as a second family

`edgefit` (sector partition + total-least-squares line fit → adjacent-side line
intersection) is kept as a second family because it recovers boxes whose stroke is
broken. The `hull` / `minAreaRect` families are **not** emitted: the keep-order IoU
dedup let their cruder quads displace the accurate ones (union D=5 553 TP / 57.4
corr_f1 against 593 / 66.0 for edgefit alone).

## Benchmark (500 real images, 1076 valid GT boxes)

Point-match metric: a detection is TP only if all 4 corners, after optimal Hungarian 4-to-4 assignment, are each within D pixels of GT.

GT = `Label@CARS_202609_raw` — the hand-drawn annotation, whose geometry is the red
rectangle burned into the input image (`benchmark/gt_final.json`). The older
`Compare-V1@CARS_202609_annotated` baseline below was measured against a machine
artifact and is **not comparable**.

**The GT is filtered with the same validity rule as the detector** (2026-10-21): a
rectangle with a vertex on the image edge is not a valid rectangle, so the 45
hand-annotated boxes that the image frame clips are not valid targets either
(1121 → 1076 boxes). Scoring against the unfiltered 1121 would penalise the
detector for correctly dropping them.

| D | TP | FN | rawFP | trueFP | recall | corrected_f1 |
|---|----|----|-------|--------|--------|--------------|
| 3 | 857 | 219 | 191 | 177 | 79.6% | 81.2 |
| 5 | **901** | 175 | 169 | 134 | 83.7% | **85.4** |
| 8 | 926 | 150 | 148 | 113 | 86.1% | 87.6 |
| 12 | 938 | 138 | 138 | 103 | 87.2% | 88.6 |

C# `cvhybrid` (holes + edgefit + side refine, hue≤4 red mask, min_area 300,
min_side 12, sat≥100, val≥140, dedup 0.40, ortho_tol 10, edge_band 0,
fit_band 8): `benchmark/pm_v10_refine.json`. Previous baseline (pm_v9,
`benchmark/pm_v9_edge_rule.json`): D=3 842 / 79.4, D=5 899 / 84.7, D=8 922 / 86.7,
D=12 937 / 87.9 — the pm_v10 change is a strict win at every D.

**The C# port is bit-exact with the Python prototype**: all 1070 boxes of the
500-image benchmark have identical coordinates *and* order.

### Held-out validation (is the win real, or fitted to the 500 images?)

`proto/sweep_real.py --seed N` draws a *different* 500-image sample from the same
pool of 56832 annotated images (seed 0 is the tuning set). pm_v10 beats pm_v9 on
both held-out samples:

| sample | pm_v9 D=3 / D=5 | pm_v10 D=3 / D=5 |
|---|---|---|
| seed 1 (1184 GT) | 878 tp / 256 fp, f 75.8 / 946 / 190, f 81.6 | **890 / 213, f 77.8** / **950 / 156, f 83.0** |
| seed 2 (1150 GT) | 864 / 222, f 77.3 / 914 / 178, f 81.5 | **865 / 200, f 78.1** / 911 / 161, f 82.0 |

`min_red_support=0.6` looked like a free extra win on the tuning set (901 TP /
130 FP vs 901 / 134) but it does **not** survive the held-out check (seed 1:
950 → 950 TP, 156 → 153 FP; seed 2: 911 → **910** TP, 161 → 154 FP) and on the
pm_v9 config it *loses* TP (946 → 939). It is therefore **not shipped** — the
gain was an interaction with pm_v10, not a general effect.

### The emulation benchmark (5000 synthetic images)

`data/Image@emul4` / `data/Label@emul4` is a 5000-image synthetic set with
**9982** complete GT boxes (generated by `proto/gen_emulation.py`, with the same
annotation furniture — crossing strokes, leader arrows, red text — as the real
data). It is ~10x faster to score than the real set and its aggregate metrics
track the real ones closely (emulation D=5 recall 84.2 / precision 87.4 vs.
real 83.7 / 87.1).

It **confirmed** the pm_v10 tuning:

| variant | D=3 TP/FP | D=5 TP/FP | D=8 TP/FP |
|---------|-----------|-----------|-----------|
| base | 2097/1076 | 2773/400 | 2973/200 |
| **pm_v10 (WINNER)** | **2544/622** | **2807/359** | 3010/156 |

But it does **not** transfer for the red-support threshold: the emulation says
`min_red_support` 0.5 → 0.7 is worth +115 TP / −256 FP, while on the real data
it is monotonically harmful (D=5 TP 899 → 885). The reason is visible in the
red-support distribution of the emitted detections
(`proto/support_profile.py` → `benchmark/support_profile_{real,emul}.json`):

| | real | emulation |
|---|---|---|
| TP support, p50 | **0.996** | 0.925 |
| FP support, p50 | **0.90** | 0.671 |
| FP support, p90 | **1.0** | 0.833 |

On the real data the false positives are *just as red as the true positives* —
they are arrow-corrupted **real** annotations, so a red-support threshold cannot
separate them and every point of threshold costs true positives. The
emulation's false positives are much less red, so the threshold separates them
cleanly there. **Use the emulation for structural parameters (fits, min_side,
refine) and the real 500-image set as the gate for everything else.**

### What each knob actually does — DoubleML decomposition (2026-10-23)

The pm_v9 → pm_v10 delta was three changes at once (`hole_min_side`,
`hole_fit_pass`, `hole_refine`), plus a fourth candidate that was rejected
(`min_red_support` 0.6). A **2⁴ complete-block
factorial panel** separates them: every image is run through all 16
configurations, so each knob is estimated against a *paired* baseline
(`proto/dml_panel.py` → `benchmark/dml_panel_{real,emul}.json`), and the
effects are estimated with **DoubleML PLR** (random-forest nuisance models,
cluster-robust SEs clustered on the image, 17 image-level covariates)
cross-checked against the exact design-based (paired) contrasts and a blocked
OLS with image fixed effects (`proto/dml_analyze.py` →
`benchmark/dml_{real,emul}.json`).

Main effects on the real 500-image benchmark (Δ TP/FP over the whole sample,
DoubleML single-treatment estimates, clustered SE):

| outcome | `hole_min_side` 8→14 | `hole_fit_pass` 0→2 | `hole_refine` 0→1 | `min_red_support` .5→.6 |
|---|---|---|---|---|
| TP @D=3 | −0.9 (p=.41) | **+7.1 (p=.026)** | +10.6 (p=.065) | −1.1 (p=.26) |
| FP @D=3 | −6.5 (p=.025) | −7.2 (p=.044) | −12.1 (p=.066) | **−7.7 (p=.0008)** |
| TP @D=5 | +0.5 (p=.32) | +1.8 (p=.23) | +2.0 (p=.53) | −1.0 (p=.43) |
| FP @D=5 | **−7.6 (p=.015)** | −4.1 (p=.12) | −3.9 (p=.39) | **−6.7 (p=.0019)** |

Four conclusions that the flat before/after table could not give:

1. **`hole_min_side=14` is a pure false-positive filter.** The paired design
   effect on TP@D=5 is **exactly 0.0000** (se 0.0000) — it removes arrow-
   corrupted FP holes and recovers *zero* true positives. At D=3 it even costs
   1 TP. The "leader arrow pulls a box corner" story was only half right.
2. **`hole_fit_pass=2` and `hole_refine=1` are the TP gainers, and they
   interact.** At D=3: fit_pass +7.1, refine +10.6, but together only +16
   (sub-additive, interaction −3.5). At D=5 the gain only materialises when
   *both* are on (conditional effect of fit_pass: +1.5 with refine=0, +2.5
   with refine=1). Robust fitting and symmetric re-fitting are partly
   substitutes — you need both, and one alone is not enough.
3. **`min_red_support=0.6` is a real trade-off whose TP cost is conditional.**
   It is the second most effective FP killer (−6.7 FP @D=5, p=.0019) but it
   costs TP *only when the geometry is not fixed*. In the four joint strata of
   (`fit_pass`, `refine`) the TP@D=5 effect of the threshold is:

   | `fit_pass` | `refine` | ΔTP@D=5 | ΔFP@D=5 |
   |---|---|---|---|
   | off | off | **−4.0** | −6.0 |
   | off | on | −1.0 | −6.0 |
   | on | off | −1.0 | −9.5 |
   | on | on | **0.0** (exactly 0.0000) | −4.0 |

   The TP cost vanishes completely once both geometry knobs are on, and its FP
   benefit shrinks at the same time (−6 FP → −4 FP). This causally
   explains the earlier puzzle ("0.6 helps under pm_v10 but hurts under
   pm_v9") and why the held-out check rejected it — it was an **interaction**,
   not a general effect. Correct reading: *the red threshold is only safe once
   the geometry is fixed.*
4. **The FP effects are sub-additive** — the knobs kill overlapping FPs. The
   four DML main effects sum to −22.3 FP @D=5 while the global pm_v9→pm_v10
   contrast is only −15. For TP the sum matched the global contrast almost
   exactly at D=3 (+15.7 vs +15).

**Box level** (`proto/dml_boxlevel.py` → `benchmark/dml_boxlevel_real.json`):
one row per GT box (1076), binary outcome = detected at D=5, controls =
image-level covariates. What causally drives a miss:

| feature | effect on P(detected) | per 1 SD | p |
|---|---|---|---|
| `box_sup_min` (min per-side red coverage) | **+0.494** | +0.147 | 1.4e−5 |
| `box_sup_mean` | +0.500 | +0.079 | 0.0013 |
| `box_min_side` | **+0.0093**/px | +0.144 | 0.0003 |
| `hole_fill` | +0.382 | +0.056 | 0.0002 |
| `hole_ms` (covering-hole size) | **−0.0039**/px | −0.059 | 0.0027 |
| `box_area` | +4.1e−6/px² | +0.015 | 0.16 |
| `box_edge_dist` | ~0 | +0.001 | 0.94 |

Red coverage dominates; size is the second factor. A *larger* covering hole
makes a box **less** likely to be found (it has swallowed a neighbouring
annotation). Area and edge distance carry no independent signal.

**Real vs emulation (the transfer hypothesis, tested causally).** The same
2⁴ panel on 2500 emulation images (`benchmark/dml_panel_emul.json`), analysed
with the same DoubleML estimator (the exact paired design effects agree to
within 0.5 TP):

| knob, D=5 | real ΔTP | emulation ΔTP | real ΔFP | emulation ΔFP |
|---|---|---|---|---|
| `min_red_support` .5→.6 | −1.0 (p=.43) | **+23.6** (p=7e−8) | −6.7 (p=.002) | **−44.6** (p=3e−19) |
| `hole_refine` 0→1 | +2.0 (p=.53) | **+22.5** (p=.0034) | −3.9 (p=.39) | −12.9 (p=.099) |
| `hole_fit_pass` 0→2 | +1.8 (p=.23) | −2.4 (p=.33) | −4.1 (p=.12) | +2.8 (p=.34) |
| `hole_min_side` 8→14 | +0.5 (p=.32) | −0.3 (p=.33) | **−7.6** (p=.015) | −1.9 (p=.11) |

The red-support threshold has the **opposite sign** on the two datasets — on
the emulation it is worth +23.6 TP with a p-value of 7e−8, on the real data it
is −1.0 and insignificant. The emulation's FPs are much less red than its TPs;
the real data's are just as red. The sign flip holds in *every* stratum: with
the geometry off the threshold is worth +27 TP on the emulation and −4 on the
real data; with the geometry fixed it is +21 on the emulation and exactly 0 on
the real data. The structural knobs keep their sign (refine helps on both) but
not their magnitude: at D=3 refine is worth **+209 TP** on the emulation
(p=3e−39) vs +10.6 on the real data. **Only the real benchmark can gate a red
threshold** — and this is now a causal statement, not an observation.

Reproduce:

```bash
python proto/dml_panel.py --dataset real --n 500 --seed 0 \
    --out benchmark/dml_panel_real.json          # ~2 min
python proto/dml_analyze.py --panel benchmark/dml_panel_real.json \
    --outcomes tp5,fp5,tp3,fp3 --cate-outcomes tp5 \
    --out benchmark/dml_real.json                # ~12 min
python proto/dml_boxlevel.py --n 500 --out benchmark/dml_boxlevel_real.json
python proto/dml_panel.py --dataset emul --n 2500 \
    --out benchmark/dml_panel_emul.json
python proto/dml_analyze.py --panel benchmark/dml_panel_emul.json \
    --outcomes tp5,fp5,tp3,fp3 --cate-outcomes tp5 \
    --out benchmark/dml_emul.json
```

### The edge rule (2026-10-21)

A rectangle is valid only when it has exactly 4 edges, every corner is orthogonal
within `ortho_tol`, and **no vertex lies on the image edge**. The previous behaviour
— *waive* the orthogonality check at a corner on the border — is gone: a box that
the image frame clips is not a valid rectangle.

| edge band | corr_f1 @D=5 | note |
|-----------|--------------|------|
| 0 (outermost pixel ring) | **84.7** | the literal rule, shipped |
| 1 | 84.6 | |
| 2 | 84.5 | |
| 6 (old `boundary_band`) | 84.4 | |
| no rule | 83.5 | |

The rule is a **win on the valid targets**: recall 82.6 → 83.6, precision 84.3 →
85.8, trueFP 172 → 149, and 46 fewer detections. Against the *unfiltered* GT it
looks like a loss (TP 926 → 899), but 45 of those 26 "lost" matches are boxes the
new rule declares invalid.

Single source of truth: `proto/quad_ortho.py::is_orthogonal_quad` +
`BRobbDetector/Services/GeometryValidation.cs::IsOrthogonalQuad`. The check runs
on the float quad in the gate **and** on the emitted (rounded) corners, so the
rectangle that is reported never touches the image frame.

### Remaining failure modes — the ceiling is annotation quality, not the algorithm

175 FN at D=5 (of the 1076 valid targets), measured on pm_v9; pm_v10 removes 2 of
them. Stage trace (`proto/scope71_trace.py`):
`far`=133 (the quad was built and passed every gate but sits >5 px off),
`quad_none`=22, `support`=11, `gate`=8, `no_hole`=3.

But the decisive finding (`proto/scope73_ceiling.py`) is that **the remaining misses
are mostly not drawn in red at all**:

| GT red-line support | boxes | detected | rate |
|----------------------|-------|----------|------|
| `sup_min >= 0.6` (well drawn) | 707 | 699 | **98.9%** |
| `sup_min >= 0.3` | 896 | 835 | 93.2% |
| all | 1076 | 899 | 83.6% |

FN GTs have a median min-side red support of **0.23** vs **0.85** for TPs; 91% of
the misses sit below 0.5. A box with `mean_sup < 0.6` misses ~80% of the time, one
with `mean_sup >= 0.8` only 4%. Example: image
`20260910_154730_772` has 8 GT boxes, 0 TP, mean_min_sup 0.14 — the image has no
red annotation at all (only printed red text "液体 0.91"). **Realistic remaining
headroom ≈ 61 boxes**, not 177.

Misses also concentrate in small boxes: short side 20-30 → 47% miss rate, 30-50 →
23%, 50-100 → 9%; aspect ≥ 6 → 38%. 407 of 499 images have 0 FN; 18 images with
≥3 FNs cover 92 of the FNs.

### Dead ends (do not re-explore)

| axis | result |
|------|--------|
| nested line-candidate rules (4800 swept) | max **+2 TP** with 129 extra FP — the red strokes simply are not in the mask |
| red-mask threshold sweep (sat×val×hue) | flat within ±1-2 TP; keep sat=100, val=140, hue=4 |
| global line / Radon detector | FP explosion |
| contour-fit v2 hole quad | 751 vs 922 TP |
| shared-line hole grouping | 116 wrong merges vs 1 right |
| quality merge, ortho-tol relaxation | no TP win |
| hole quad mode v2, hole group mode `replace`/`extra`, hole hull/convexity gate | clear regression (D=5 726 / 749 TP) |
| `min_red_support` ≥ 0.58 on the real data | monotonically losing TP (899 → 855 at 0.8) |
| `hole_fit_tol` alone, `hole_pts`, `hole_expand`, `dedup_iou`, `hole_min_comp_area` | no effect |

> Robust side-fit passes (`hole_fit_pass`) and `hole_refine` were listed here as
> dead ends until 2026-10-23; re-swept on the real 500-image set they *are* wins
> and are shipped in pm_v10 (see Stage 4b above). The earlier sweep had them
> mis-parameterised (one-sided refine band, `fit_pass` without a tuned `fit_tol`).

### Prototype / production parity

`proto/scope45_holes.py` (`P`, `HOLE_DEFAULT`, `PIPE_DEFAULT`) mirrors
`BRobbDetector/Services/HoleDetector.cs` and
`BRobbDetector/Services/CvHybridDetector.cs` one-for-one. Change one, change the
other — the C# port currently reproduces the Python output bit-exactly.

### Historical baseline (invalid GT — Compare-V1, 746 boxes)

| D | TP | FN | rawFP | trueFP | recall | corrected_f1 |
|---|----|----|-------|--------|--------|--------------|
| 3 | 601 | 145 | 364 | 194 | 80.6% | 78.0 |
| 5 | 609 | 137 | 356 | 191 | 81.6% | 78.8 |
| 8 | 624 | 122 | 341 | 180 | 83.6% | 80.5 |
| 12 | 632 | 114 | 333 | 175 | 84.7% | 81.4 |

Frozen baseline: `benchmark/pm_v5_2_baseline.json`. C# port updated and Release build passes.

## Build & Run

```bash
# Build (Release)
dotnet build BRobbDetector/BRobbDetector.csproj -c Release

# Run GUI
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Release

# CLI detection
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Release -- --cli <inputDir> <outputDir> [--ocr]

# Checker
dotnet run --project BRobbDetector/BRobbDetector.csproj -c Release -- --check <imageDir> <labelDir>
```

## Tech Stack

- .NET 10, Avalonia 11.2.8 (MVVM), OpenCvSharp5 5.0.0.20261018 (OpenCV 5 native)
- Python prototype + benchmark harness (`proto/`)
- RapidOCR sidecar for OCR (`proto/ocr_sidecar.py`)

## Project Structure

```
BRobbDetector/          C# project (main app)
  Services/             CvHybridDetector.cs (cvhybrid / pm_v9 pipeline), HoleDetector.cs
                        (hole family), GeometryValidation.cs (rectangle validity rule),
                        ColorBoxDetector, YoloObbDetector
  ViewModels/           MVVM view models
  Views/                Avalonia UI
proto/                  Python prototype + eval harness
  common.py             IMG / LAB (GT path) / red_mask / iou_obb
  cv_hybrid.py          5 CV families + hybrids
  scope45_holes.py      ★ production hole detector + tunable pipeline (HoleDetector.cs port)
  sweep_real.py         ★ real-data parameter sweep (500 imgs, ~6 s/variant) — the gate
  sweep_emulation.py    ★ emulation-dataset sweep (5000 imgs, ~50 s/variant)
  gen_emulation.py      ★ synthetic dataset generator (annotation furniture)
  eval_emulation.py     ★ emulation benchmark (detect_one / load_gt / score_view)
  hole_feature_profile.py  ★ hole-level TP/FP feature profile (min_side, fill, hull, ...)
  eval_pointmatch.py    Point-match metric + GT-corrected eval
  eval_pm_benchmark.py  500-image real benchmark
  eval_det_dir.py       Evaluate any detection dir against the true GT
  gt_final.py           Ground-truth arbitration evidence bundle
  scope40_retune.py     One-axis-at-a-time re-tune on the TRUE GT
  scope41_small_targets.py  Small-target / FN-reason analysis
  scope42_combo.py      Combination check of the marginal winners
  scope73_ceiling.py    ★ ceiling analysis: how much of the GT is actually drawn
  scope76_edge_consistent.py  ★ edge rule: detection filter vs. consistent GT
  scope78_parity.py     ★ C#/Python parity + benchmark (--run-csharp for one command)
benchmark/             Frozen baselines + per-LOOP results
data/                  Test images and labels
  Image@emul4/         5000 synthetic images (untracked - do not git add)
  Label@emul4/         Their complete GT (9982 boxes)
```