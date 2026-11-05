# All-cause analysis — causal-graph factors vs four-corner recall

Tool: **`riantr/moonbit_doubleML@0.164.62`** (latest available). Fresh
pm_v10 instrumentation on the real 500 images (480 with n_gt > 0); outcome =
per-image four-corner recall @ D=5. Pooled recall = **901/1076 = 83.7%**
(per-image mean 91.9% because single-box images dominate).

Each factor is estimated by PLR **adjusting for all other factors in its
set**; the ranking column is `theta × sd(factor) × 100` = recall percentage
points per 1 SD.

## Full model — top five

| rank | factor | theta / unit | t | p | pp per SD |
|---|---|---|---|---|---|
| 1 | **S10_dedup_rate** | +0.38770 |   2.94 | 0.00326 | **+15.72** |
| 2 | **Ovl** | -0.44397 |  -3.20 | 0.00138 | **-14.46** |
| 3 | **W** | +0.16582 |   3.90 | 9.7e-05 | **+11.42** |
| 4 | **S4_quads** | -0.00580 |  -4.10 | 4.15e-05 | **-10.18** |
| 5 | **S5_edgefit_share** | -1.64132 |  -2.04 | 0.0408 | **-9.05** |

## Upstream (pre-gate) model — top five

This excludes the post-outcome dedup rate and the gate/dedup mediators,
so it is the actionable ranking.

| rank | factor | theta / unit | t | p | pp per SD |
|---|---|---|---|---|---|
| 1 | **Ngt** | -0.04268 |  -2.58 | 0.01 | **-8.30** |
| 2 | **Ovl** | -0.20251 |  -3.98 | 6.93e-05 | **-6.60** |
| 3 | **S1_mask** | -0.90648 |  -0.32 | 0.749 | **-1.08** |
| 4 | **W** | -0.00808 |  -0.27 | 0.788 | **-0.56** |
| 5 | **S4_quads** | +0.00026 |   0.14 | 0.892 | **+0.46** |

## Full model — all retained factors

| rank | factor | theta / unit | t | p | pp per SD |
|---|---|---|---|---|---|
| 1 | **S10_dedup_rate** | +0.38770 |   2.94 | 0.00326 | **+15.72** |
| 2 | **Ovl** | -0.44397 |  -3.20 | 0.00138 | **-14.46** |
| 3 | **W** | +0.16582 |   3.90 | 9.7e-05 | **+11.42** |
| 4 | **S4_quads** | -0.00580 |  -4.10 | 4.15e-05 | **-10.18** |
| 5 | **S5_edgefit_share** | -1.64132 |  -2.04 | 0.0408 | **-9.05** |
| 6 | **S1_mask** | +7.01997 |   5.86 | 4.6e-09 | **+8.38** |
| 7 | **S7_side_rate** | -4.72453 |  -1.34 | 0.18 | **-4.47** |
| 8 | **Ngt** | +0.01845 |   1.67 | 0.0951 | **+3.59** |
| 9 | **S2_denoise** | -2.11201 |  -1.21 | 0.226 | **-2.84** |
| 10 | **S9_support_rate** | +2.63073 |   0.75 | 0.453 | **+0.66** |
| 11 | **S8_ortho_rate** | -0.10022 |  -0.09 | 0.926 | **-0.33** |

## Upstream model — all factors

| rank | factor | theta / unit | t | p | pp per SD |
|---|---|---|---|---|---|
| 1 | **Ngt** | -0.04268 |  -2.58 | 0.01 | **-8.30** |
| 2 | **Ovl** | -0.20251 |  -3.98 | 6.93e-05 | **-6.60** |
| 3 | **S1_mask** | -0.90648 |  -0.32 | 0.749 | **-1.08** |
| 4 | **W** | -0.00808 |  -0.27 | 0.788 | **-0.56** |
| 5 | **S4_quads** | +0.00026 |   0.14 | 0.892 | **+0.46** |
| 6 | **S2_denoise** | -0.31495 |  -0.52 | 0.603 | **-0.42** |
| 7 | **S5_edgefit_share** | +0.00709 |   0.03 | 0.98 | **+0.04** |

## Reading

- **Full model:** `S10_dedup_rate`, `Ovl`, `W`, `S4_quads`, `S5_edgefit_share`
  lead. `S10_dedup_rate` is **post-outcome / reverse-causal** (more detected
  boxes create more duplicates to remove); it is not a lever.
- **Upstream model (total effect, no gate/dedup mediators):** `Ngt` (−8.30
  pp/SD, p=0.010) and `Ovl` (−6.60, p=6.9e-05) dominate; `S1_mask`, `W`,
  `S4_quads` are not significant. So the robust total-effect associations are
  **more GT boxes per image and more GT overlap lower recall**.
- **Actionable top five (full model, excluding the post-outcome dedup rate):**
  `Ovl` (−14.5), `W` (+11.4), `S4_quads` (−10.2), `S5_edgefit_share` (−9.0),
  `S1_mask` (+8.4). `W`, `S4`, `S5` and `S1` are **direct effects conditional on
  the gate**; their upstream total effects are not significant.
- **Sign flips are mediation, not noise:** `Ngt` is +3.6 (ns) in the full model
  but −8.3 (p=0.010) upstream; `W` is +11.4 (p=9.7e-05) full but −0.6 (ns)
  upstream. Do not average the two models; the upstream model is the total
  effect and the full model is the direct effect.
- `S8_ortho_rate` ≈ 0 and p≈0.9 — the orthogonality factor is inert, exactly as
  OR2 measured. `S2_denoise` is not significant, matching LOOP-28.
- `S10_dedup_rate` is positive only because it is a post-outcome rate; do not
  read it as "dedup helps recall".

## Caveats

- Conditional associations, not interventions: stage features are post-treatment
  outputs and the graph's `U` (image difficulty) is unobserved.
- The six-stage panel's significance is estimator-sensitive (LOOP-45: 2/6
  replicated under the MoonBit linear PLR), so do not over-read individual p-values.
- The first pass found an exact dependency `S4_quads = S5_edgefit + S6_refine`;
  this version uses non-redundant rates and drops VIF>10 factors.
- Read the top five as a screening order for the next measured lever, not as
  independent causal effects; the LOOP-45 sign/significance test remains the gate.
