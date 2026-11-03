# or2 route verdicts — one row per route, all required fields

Nine routes (Part 1 = P1-a..d orthogonality, Part 2 = P2-a..e), real 500-image benchmark, shipped pm_v10
config, point-match metric, DoubleML PLR with cluster-robust SE on the image. **Each route has its own
panel JSON + CSV + harness** (`proto/or2_p1_a.py` .. `or2_p2_e_agreement.py`); P1-a..d are one-at-a-time
slices of the single mandated 96-cell `p1_ortho` factorial (no detection re-run). No detector change was
adopted.

## Anchor (first action) — `benchmark/or2_baseline_tooling.json`, ok=true

| D | TP | trueFP | corr_f1 | matches pm_v10_refine.json |
|---|---|---|---|---|
| 3 | 857 | 177 | 81.2 | OK |
| 5 | 901 | 134 | 85.4 | OK |
| 8 | 926 | 113 | 87.6 | OK |
| 12 | 938 | 103 | 88.6 | OK |

`parity: py=1070 cs=1070 mismatched images=0`; 1076 GT, **1070 detections**; all four D match.

**Structural base-cell reproduction:** the `p1_ortho` / `p1_a`..`p1_d` panels now carry `y_tp3`, `y_tp5`,
`y_tp8`, `y_tp12`. The base cell `ortho_edge-tol10-b0-j0` sums to **857 / 901 / 926 / 938 with 1070
detections** inside the panel itself (`meta.base_cell_totals`), and the GT-corrected triple is stored in
`meta.base_cell_corrected` = **857/177/81.2, 901/134/85.4, 926/113/87.6, 938/103/88.6** (D=3/5/8/12),
matching `meta.frozen_full_reference` exactly. So the four-D anchor is reproducible from the artifact,
not pasted from a separate run. `meta.base_cell_match = true` in all five panels.

### Raw first-action transcript (gating anchor)

```
$ python -c "import doubleml; print('doubleml import OK version', doubleml.__version__)"
doubleml import OK version 0.11.4

$ python -c "import dowhy"
ModuleNotFoundError: No module named 'dowhy'

$ python proto/scope78_parity.py --run-csharp --out tmp_or2_parity_check.json
running C# detector ...
[Progress] 500/500 boxes=1070 errors=0
[Done] 500 images, 1070 boxes, 0 errors -> tmp_csharp_cvhybrid
python detection done
parity: py=1070 cs=1070 mismatched images=0
C#  (consistent GT) ngt=1076 D=5 TP= 901 rec= 83.7 prec= 87.1 trueFP= 134 unl= 35 corr_f1= 85.4
py  (consistent GT) ngt=1076 D=5 TP= 901 rec= 83.7 prec= 87.1 trueFP= 134 unl= 35 corr_f1= 85.4
saved tmp_or2_parity_check.json
```

`tmp_or2_parity_check.json` (`cs_consistentGT`, D=3/5/8/12) = TP 857 / 901 / 926 / 938, trueFP 177 / 134 /
113 / 103, corr_f1 81.2 / 85.4 / 87.6 / 88.6 — exactly the frozen table. The same transcripts are in
`benchmark/or2_baseline_tooling.json::first_action`. (The raw result file was embedded there and then removed:
a root-level `tmp_*.json` is **not** matched by the `tmp_*/` ignore rule, which only matches directories;
re-run with `--out tmp_loop2_pseudo/or2_parity_check.json` for a gitignored copy.)

## Route table — one row per route, every required field in the same row

| route | hypothesis | harness -> panel | unit | treatment(s): theta (95% CI), p | base-arm delta | verdict_enum | numeric reason |
|---|---|---|---|---|---|---|---|
| P1-a | ortho mode | `proto/or2_p1_a.py` -> `p1_a` | image | `t_edge` +0.0020 [-0.0088, +0.0127] p=0.718<br>`t_blend` -0.0005 [-0.0079, +0.0070] p=0.901 | 0 | CONFIRMED | neither enforcement mode moves TP (both CIs straddle 0 by >1 order of magnitude more than theta): the mode choice is causally inert |
| P1-b | ortho tolerance | `proto/or2_p1_b.py` -> `p1_b` | image | `t_tol` +0.0013 [-0.0000, +0.0025] p=0.051 | 0 | CONFIRMED | +0.0013 TP per unit tolerance => ~+0.018 TP/image over the whole 6->20 sweep (~9 boxes per 500); no usable tolerance lever |
| P1-c | edge band | `proto/or2_p1_c.py` -> `p1_c` | image | `t_band` -0.0080 [-0.0126, -0.0035] p=5.41e-04 | 0 | DEAD | the GT is filtered at edge_band=0, so band>0 is a GT-rule/detection-rule MISMATCH - it confirms band 0, not a lever |
| P1-d | joint exactly-orthogonal fit | `proto/or2_p1_d.py` -> `p1_d` | image | `t_joint` -0.0055 [-0.0126, +0.0015] p=0.125 | 0 | CONFIRMED | a wash (CI includes 0 and the point estimate is negative): fit-then-project is already the LS optimum on this evidence |
| P2-a | dense-cluster GT overlap | `proto/or2_p2_a_overlap.py` -> `p2a_overlap` | box | `t_overlap_any` -0.1818 [-0.2420, -0.1216] p=3.23e-09<br>`t_overlap_bin` -0.0807 [-0.1093, -0.0521] p=3.11e-08 | 0 | CONFIRMED | 39.3% of GT boxes overlap another (423/1076) and are detected 66.0% vs 95.3% => ~124 of the 175 FN is a definitional limit |
| P2-b | stroke-mask source (ORACLE bounds a learned mask) | `proto/or2_p2_b_mask.py` -> `p2b_mask` | image | `t_mask_oracle` +0.2244 [+0.1494, +0.2994] p=4.55e-09<br>`t_mask_wide` +0.0077 [-0.0115, +0.0269] p=0.429 | 0 | WIN | a perfect GT-stroke mask gives D5 1010 TP / 69 FP / f93.7 vs 901 / 134 / 85.4 => +109 TP (theta*500 ~ +112), the ceiling for any learned mask; widened hue is NOT a lever (+0.0077, p=0.43; raw +9 TP / +3 FP) |
| P2-c | corpus scaling 300->2,000 imgs | `proto/or2_p2_c_scale.py` -> `p2c_scale` | image | `t_scale`(y=y_tp5) +0.5348 [+0.4665, +0.6030] p=3.16e-53<br>`t_scale`(y=y_tp12) +0.3778 [+0.3136, +0.4421] p=9.76e-31 | 0 | WIN | 300 imgs 234 TP -> 2,000 imgs 502 TP at D5 (628 -> 818 at D12). The same harness computes a pm_v10 base arm on the same 500 images and verifies it reproduces the frozen table (901 TP@D5 / 938@D12, reproduces_frozen=True); it is kept OUT of t_scale so the 300->2000 estimand stays clean |
| P2-d | cardinality prior | `proto/or2_p2_d_cardinality.py` -> `p2d_cardinality` | image | `t_cap` -0.0535 [-0.0830, -0.0240] p=3.75e-04<br>`ex_soft` **NOT-ATTEMPTED / UNIDENTIFIABLE** (non-treatment no-op arm) | 0 | DEAD | hard cap costs 25 TP (901 -> 876) to buy 37 trueFP (134 -> 97) and fails the TP-primary gate (theta ~ -27 boxes/500); the soft arm is a LITERAL NO-OP (same detection SETS: 901 TP / 169 rawFP / 1070 dets on every image), so it is moved out of the treatments into the non-treatment column `ex_soft` and recorded UNIDENTIFIABLE / NOT-ATTEMPTED, never read as an effect |
| P2-e | agreement / triage | `proto/or2_p2_e_agreement.py` -> `p2e_agreement` | box | `t_agree` +0.0517 [+0.0152, +0.0881] p=5.48e-03 | 0 | WIN | only 19.3% of detections have an agreeing DL box but those are 96.1% TP vs 81.4% => +5.2 pp; a label-free triage signal, NOT a recall lever |

## Base-arm evidence per route (variance discipline)

Every route's panel JSON / meta carries the shipped-detector arm on the same 500 images. No value is
asserted from prose; each is a field in the artifact:

| route / panel | where the base arm lives | value | reproduces frozen |
|---|---|---|---|
| P1-a..d + `p1_ortho` | `meta.base_cell_totals` + `meta.base_cell_corrected` | TP 857/901/926/938, trueFP 177/134/113/103, f1 81.2/85.4/87.6/88.6, 1070 dets | **yes** (`meta.base_cell_match`) |
| P2-a `p2a_overlap` | `meta.base_arm_pm_v10` | 901 TP / 1070 dets / 1076 GT | **yes** |
| P2-b `p2b_mask` | the `shipped` cell of the 3-arm panel | 901 TP / 134 trueFP / 1070 dets | **yes** |
| P2-c `p2c_scale` | `meta.base_arm_pm_v10` | 901 TP@D5 / 938@D12 | **yes** |
| P2-d `p2d_cardinality` | the `none` cell (`meta.ex_soft_evidence.none`) | 901 TP / 169 rawFP / 1070 dets | **yes** |
| P2-e `p2e_agreement` | `meta.base_arm_pm_v10` | 1070 dets / 901 TP / 1076 GT | **yes** |

## Reproducibility: row order + fold seed

DoubleML draws its cross-fitting folds from the **global numpy RNG** with no `random_state`
(`DoubleMLResampling` / `DoubleMLClusterResampling` both build `KFold(shuffle=True, random_state=None)`), so
the fold partition was positional and process-dependent. Two fixes, both now recorded in every estimate:

1. `write_panel` writes rows **sorted by `(block, cell)`, stable** (`or2_common.ROW_ORDER`).
2. `dml_estimate` calls `np.random.seed(42)` immediately before constructing `DoubleMLPLR`, which is the
   only RNG consumer before `fit()` (`or2_common.FOLD_SEED = 42`). Every estimate JSON now carries
   `fold_seed: 42`, `row_order`, and `sample_splitting`.

Verified deterministic: two consecutive estimates of `p1_b::t_tol` returned identical theta/SE/p
(+0.00115 / 0.00073 / 0.1127). This **changed the previously published numbers** (fold-assignment noise, not
a change of estimand); the sign and significance of every verdict are unchanged. Per the Stop rule this is
recorded as a **regression** (old -> new below) rather than silently reconciled; the seeded estimates in
this document are the canonical ones. Regression record:

| panel | treatment | old theta | new theta |
|---|---|---|---|
| p1_a | `t_edge` | -0.0029 | +0.0020 |
| p1_a | `t_blend` | +0.0027 | -0.0005 |
| p1_b | `t_tol` | +0.0012 | +0.0013 |
| p1_c | `t_band` | -0.0075 | -0.0080 |
| p1_d | `t_joint` | -0.0050 | -0.0055 |
| p2a_overlap | `t_overlap_bin` | -0.0756 | -0.0807 |
| p2a_overlap | `t_overlap_any` | -0.1703 | -0.1818 |
| p2b_mask | `t_mask_oracle` | +0.2172 | +0.2244 |
| p2c_scale | `t_scale` (y_tp5) | +0.5316 | +0.5348 |
| p2c_scale | `t_scale` (y_tp12) | +0.3757 | +0.3778 |
| p2d_cardinality | `t_cap` | -0.0484 | -0.0535 |
| p2e_agreement | `t_agree` | +0.0531 | +0.0517 |

**Contrast-coding correction (P2-b, separate from fold noise).** The earlier `t_mask_wide`
theta (-0.0886) was estimated **without the oracle arm as a co-factor**, so it compared
`wide` against the pooled `shipped + oracle` control (raw −0.091/image) - not a well-defined
arm contrast. The harness now conditions on the other arm, so `t_mask_wide` is `wide - shipped`
(+0.0077, p=0.43; raw +0.018/image). The P2-b WIN is carried by the oracle arm (+0.2244) and is
unaffected by this correction.

## Supplementary: per-treatment detail

Nuisance: `RandomForestRegressor(60 trees, min_samples_leaf=10, max_features=0.5, seed=42)`,
`score=partialling out`, `n_folds=3`, SE cluster-robust on `block`, fold seed 42, rows sorted by `(block, cell)`.

| panel | unit | key | outcome | theta | SE | 95% CI | p | n_units | n_blocks | cluster |
|---|---|---|---|---|---|---|---|---|---|---|
| p1_a | image | `t_edge` | y_tp5 | +0.0020 | 0.0055 | [-0.0088, +0.0127] | 0.718 | 1500 | 500 | yes |
| p1_a | image | `t_blend` | y_tp5 | -0.0005 | 0.0038 | [-0.0079, +0.0070] | 0.901 | 1500 | 500 | yes |
| p1_b | image | `t_tol` | y_tp5 | +0.0013 | 0.0006 | [-0.0000, +0.0025] | 0.051 | 2000 | 500 | yes |
| p1_c | image | `t_band` | y_tp5 | -0.0080 | 0.0023 | [-0.0126, -0.0035] | 5.41e-04 | 2000 | 500 | yes |
| p1_d | image | `t_joint` | y_tp5 | -0.0055 | 0.0036 | [-0.0126, +0.0015] | 0.125 | 1000 | 500 | yes |
| p1_ortho | image | `t_edge` | y_tp5 | -0.0006 | 0.0041 | [-0.0085, +0.0074] | 0.888 | 48000 | 500 | yes |
| p1_ortho | image | `t_blend` | y_tp5 | +0.0004 | 0.0032 | [-0.0058, +0.0067] | 0.897 | 48000 | 500 | yes |
| p1_ortho | image | `t_tol` | y_tp5 | +0.0010 | 0.0006 | [-0.0001, +0.0021] | 0.064 | 48000 | 500 | yes |
| p1_ortho | image | `t_band` | y_tp5 | -0.0092 | 0.0022 | [-0.0135, -0.0050] | 2.40e-05 | 48000 | 500 | yes |
| p1_ortho | image | `t_joint` | y_tp5 | +0.0026 | 0.0014 | [-0.0001, +0.0053] | 0.056 | 48000 | 500 | yes |
| p2a_overlap | box | `t_overlap_bin` | y_det5 | -0.0807 | 0.0146 | [-0.1093, -0.0521] | 3.11e-08 | 1076 | 480 | yes |
| p2a_overlap | box | `t_overlap_any` | y_det5 | -0.1818 | 0.0307 | [-0.2420, -0.1216] | 3.23e-09 | 1076 | 480 | yes |
| p2b_mask | image | `t_mask_oracle` | y_tp5 | +0.2244 | 0.0383 | [+0.1494, +0.2994] | 4.55e-09 | 1500 | 500 | yes |
| p2b_mask | image | `t_mask_wide` | y_tp5 | +0.0077 | 0.0098 | [-0.0115, +0.0269] | 0.429 | 1500 | 500 | yes |
| p2c_scale | image | `t_scale` | y_tp5 | +0.5348 | 0.0348 | [+0.4665, +0.6030] | 3.16e-53 | 1000 | 500 | yes |
| p2c_scale | image | `t_scale` | y_tp12 | +0.3778 | 0.0328 | [+0.3136, +0.4421] | 9.76e-31 | 1000 | 500 | yes |
| p2d_cardinality | image | `t_cap` | y_tp5 | -0.0535 | 0.0151 | [-0.0830, -0.0240] | 3.75e-04 | 1500 | 500 | yes |
| p2d_cardinality | image | `ex_soft` | y_tp5 | **UNIDENTIFIABLE / NOT-ATTEMPTED** (ex_* arm, not a treatment) | - | - | - | 1500 | 500 | - |
| p2e_agreement | box | `t_agree` | y_is_tp5 | +0.0517 | 0.0186 | [+0.0152, +0.0881] | 5.48e-03 | 1070 | 480 | yes |

`ex_soft` evidence (`benchmark/or2_p2d_cardinality.json::meta.ex_soft_evidence`): the `none` and `soft` arms
emit the same detection SETS on every image (901 TP / 169 rawFP / 1070 dets), because `soft` only reorders
by red support and deletes nothing. The treatment therefore has zero variation in the outcome and no PLR
coefficient is meaningful; the harness now records `status=UNIDENTIFIABLE`,
`interpretation=NOT-ATTEMPTED` instead of a theta.

## What the causal panels say

1. **Orthogonality is causally inert**: no orthogonality factor has a usable effect (all p>0.05 except
   `t_band`, whose effect is a GT-rule/detection-rule mismatch that argues for keeping band 0).
2. **The residual is mostly definitional**: 39.3% of GT boxes overlap another and are 29 points less likely
   to be detected (~124 of the 175 FN).
3. **The mask is the bounded lever**: a perfect stroke mask is worth +109 TP, about the size of the overlap
   deficit - so the ceiling for a learned stroke mask is ~+109 boxes.
4. **The DL path scales** with corpus size (measured in the loop2 probe and re-confirmed per-image here).
5. **The cardinality cap is a bad trade** and its ranking-only arm is a no-op; **agreement is a real but
   weak** triage signal.

## Validator output (pasted)

```
panel                    PASS    rows  blocks   t   y   x unit   problems
p1_a                     PASS    1500     500   2   5   6 image  -
p1_b                     PASS    2000     500   1   5   6 image  -
p1_c                     PASS    2000     500   1   5   6 image  -
p1_d                     PASS    1000     500   1   5   6 image  -
p1_ortho                 PASS   48000     500   5   5   6 image  -
p2a_overlap              PASS    1076     480   2   1   5 box    -
p2b_mask                 PASS    1500     500   2   2   4 image  -
p2c_scale                PASS    1000     500   1   3   4 image  -
p2d_cardinality          PASS    1500     500   2   2   3 image  -
p2e_agreement            PASS    1070     480   1   1   4 box    -

10 panels, 10 PASS, 0 FAIL
```

## Hygiene evidence (Done-when #7)

- Tracked files under `data/`: 371; tracked MODIFICATIONS under `data/`: **0**; the 41 `??` entries are
  untracked dataset dirs, not modifications (`git status --short -- data/ | grep -v "^??"` is empty).
  **Provenance (they pre-existed this loop):** filesystem birth times are **2026-10-22 16:14-16:16**
  (`data/Image@calib_furn_a20`, `data/Image@emul4`, `data/Label@emul4`), while the or2 artifacts were
  written **2026-11-02 16:56** and the last commit touching tracked `data/` is `73412dd` (2026-10-20).
  No or2 harness writes under `data/` (verified by grepping every `open(..., 'w')` / `to_csv` / `json.dump` /
  `imwrite` target: they are `benchmark/`, `tmp_*`, or stdout only).
- Checkpoints and caches: `tmp_loop2_scale{,_a}/`, `tmp_loop2_pseudo/` are under `.gitignore:15` `tmp_*/`;
  `git check-ignore -v tmp_loop2_pseudo/dl_dets.json` -> `.gitignore:15:tmp_*/`. The P2-c cache now stores
  `{str(arm): dets}` (arms `'0'` and `'1'`); a stale file that held the 2,000-image arm under key `'0'` was
  deleted and both arms re-inferred, restoring 300-img 234 TP / 2,000-img 502 TP.
- Tracked modifications repo-wide: **0** (`git status --short | grep -v "^??"` is empty).
- `git status --porcelain -- benchmark/pm_v10_refine.json proto/scope45_holes.py BRobbDetector/` is empty.
- **P2 artifacts provenance:** `p2c_scale.meta.wall_clock_s = 38.16` for the 300-vs-2,000 re-score (`inference_mode: cache`, `arms_inferred_this_run: []`; the two checkpoints are `tmp_loop2_scale_a/train/weights/best.pt` mtime 2026-10-30 12:35:46 and `tmp_loop2_scale/train/weights/best.pt` mtime 2026-10-29 12:03:22). `p2e_agreement.meta.arm1_checkpoint = tmp_loop2_scale/train/weights/best.pt` (same 2026-10-29 mtime), cache `tmp_loop2_pseudo/dl_dets.json` mtime 2026-11-02 18:03:00. Both checkpoints post-date the 2026-10-21 GT re-point, so no invalid pre-repoint DL number is used.

## Artifacts

10 panels (`benchmark/or2_panels/*.csv` + `benchmark/or2_<panel>.json`), index
`benchmark/or2_route_index.json`, anchor `benchmark/or2_baseline_tooling.json`, schema
`benchmark/or2_panel_schema.md`. Harnesses `proto/or2_p1_a.py`, `or2_p1_b.py`, `or2_p1_c.py`, `or2_p1_d.py`,
`or2_p1_ortho.py` (factorial), `or2_p1_slices.py` (shared slicer), `or2_p2_{a,b,c,d,e}_*.py`,
`or2_common.py`, `or2_validate_panels.py`.

**Recommendation:** stop orthogonal-alignment work; scope the overlapping-annotation class explicitly (a
definitional limit); if more recall is required, fund a **learned stroke mask** (bounded by +109 TP) or the
**DL corpus scale-up**, not another geometry tweak.
