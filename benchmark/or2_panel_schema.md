# or2 panel schema (DoWhy / DoubleML ingest format)

Every route in this loop emits **one panel** with this identical schema, so a
DoWhy or DoubleML analysis can consume any of them without special-casing.

## Columns

| column | meaning | rule |
|---|---|---|
| `unit` | the unit of analysis | exactly one value per panel: `image` or `box` |
| `block` | the cluster key | always the **image id**; required for cluster-robust SE |
| `cell` | which config / arm produced the row | free text, one value per arm |
| `t_*` | treatment column(s) | manipulable, **≥2 levels, ≥10 units per level** |
| `y_*` | outcome column(s) | per unit; TP counts for image panels, 0/1 for box panels |
| `x_*` | covariates | **pre-treatment only** — never derived from an outcome |

Continuous exposures that cannot satisfy the ≥10-units-per-level rule are stored
with an `ex_` prefix (e.g. `ex_overlap_deg`) and are **not** treatments; they stay
in the CSV for DoWhy use as continuous parents. A degenerate assigned arm that is
a literal no-op is likewise stored as `ex_*` (e.g. `ex_soft`) and recorded in
`not_attempted` instead of being estimated as a `t_*`.

All panels pass `proto/or2_validate_panels.py`, which enforces the table above
mechanically and additionally flags any `x_*` whose correlation with a `y_*` is
> 0.9999 as leakage.

## Estimand per unit level

- **`unit = image`** (P1, P2-b, P2-c, P2-d): the estimand is the **per-image
  change in true positives** at a fixed tolerance D, in units of TP per image.
  Multiply θ by 500 to read it as a box count on the 500-image benchmark.
- **`unit = box`** (P2-a, P2-e): the estimand is a **per-unit probability
  change** (a percentage-point effect on the probability that this box is
  detected / that this detection is a TP). It is *not* directly comparable to an
  image-level θ, and it must not be multiplied by 500.

## DoWhy spec fields (mandatory in every `benchmark/or2_<panel>.json`)

Each panel JSON carries `causal_spec` with:

- `unit` — the analysis unit
- `treatment` — the primary `t_*` column
- `outcome` — the primary `y_*` column
- `common_causes` — the `x_*` covariates plus any co-assigned factors
- `instruments` — claimed instruments, or `[]` (none claimed in this loop)
- `mediators` — claimed mediators, or `[]` (none claimed in this loop)
- `covariates_pre_treatment` + `covariate_note` — per-panel statement that every
  `x_*` is measured before the arm assignment and none is derived from a `y_*`
- `assumption` — one sentence stating the identification assumption
- `dowhy_usage` — a ready-to-paste `dowhy.CausalModel(data=..., treatment=...,
  outcome=..., common_causes=[...])` call

`dowhy` is **not installed** in this environment and must not be installed; these
fields are written so that a DoWhy run elsewhere works verbatim. Every panel also
carries `dml_estimates`, keyed by treatment, holding theta / SE / 95% CI / p /
nuisance model / n_units / n_blocks / a `cluster_robust_se` flag.

## Estimation settings (identical across panels)

DoubleML PLR, `score='partialling out'`, `n_folds=3`, nuisance
`RandomForestRegressor(60 trees, min_samples_leaf=10, max_features=0.5, seed=42)`,
SE **cluster-robust on `block`**. The nuisance was deliberately kept small so the
48,000-row P1 factorial is estimable on CPU; the exact settings are recorded inside
every estimate.

**Reproducibility:** rows are written **sorted by `(block, cell)` (stable)** and
`dml_estimate` seeds numpy with `FOLD_SEED = 42` immediately before constructing
`DoubleMLPLR`, because DoubleML's `KFold(shuffle=True, random_state=None)` otherwise
draws folds from the process RNG. Every estimate records `fold_seed`, `row_order`
and `sample_splitting`; re-estimating a panel now yields identical theta/SE/p.

**Degenerate arms:** an assigned arm whose levels are a literal no-op is moved into a
non-treatment `ex_*` column and recorded in `not_attempted` as
`status = UNIDENTIFIABLE`, `interpretation = NOT-ATTEMPTED` with its evidence, and
**no theta** (currently `p2d_cardinality::ex_soft`). A number there would be a nuisance
artifact, not an effect.

## Panels

| panel | route | unit | treatments | outcomes | rows | blocks |
|---|---|---|---|---|---|---|
| `p1_a` | **P1-a** ortho mode (slice) | image | `t_edge`, `t_blend` | `y_tp3`, `y_tp5`, `y_tp8`, `y_tp12`, `y_fp5` | 1500 | 500 |
| `p1_b` | **P1-b** ortho tolerance (slice) | image | `t_tol` | `y_tp3`, `y_tp5`, `y_tp8`, `y_tp12`, `y_fp5` | 2000 | 500 |
| `p1_c` | **P1-c** edge band (slice) | image | `t_band` | `y_tp3`, `y_tp5`, `y_tp8`, `y_tp12`, `y_fp5` | 2000 | 500 |
| `p1_d` | **P1-d** joint fit (slice) | image | `t_joint` | `y_tp3`, `y_tp5`, `y_tp8`, `y_tp12`, `y_fp5` | 1000 | 500 |
| `p1_ortho` | P1-a..d COMBINED factorial | image | `t_edge`, `t_blend`, `t_tol`, `t_band`, `t_joint` | `y_tp3`, `y_tp5`, `y_tp8`, `y_tp12`, `y_fp5` | 48000 | 500 |
| `p2a_overlap` | P2-a | box | `t_overlap_bin`, `t_overlap_any` | `y_det5` | 1076 | 480 |
| `p2b_mask` | P2-b | image | `t_mask_oracle`, `t_mask_wide` | `y_tp3`, `y_tp5` | 1500 | 500 |
| `p2c_scale` | P2-c | image | `t_scale` | `y_tp5`, `y_tp12`, `y_fp5` | 1000 | 500 |
| `p2d_cardinality` | P2-d | image | `t_cap` (+ `ex_soft`, a non-treatment no-op arm) | `y_tp5`, `y_fp5` | 1500 | 500 |
| `p2e_agreement` | P2-e | box | `t_agree` | `y_is_tp5` | 1070 | 480 |

`p1_a`..`p1_d` are one-at-a-time SLICES of the single mandated 96-cell `p1_ortho` factorial
(each holds the other three factors at the shipped setting); no detection was re-run to make them.

CSVs live in `benchmark/or2_panels/`; the matching JSON (schema + causal spec + estimates) in
`benchmark/or2_<panel>.json`; the route->panel->harness map is `benchmark/or2_route_index.json`.

## Validation run (pasted - Done-when #2)

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
