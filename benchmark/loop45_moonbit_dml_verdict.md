# LOOP-45 — MoonBit DoubleML replication of the causal claims

Tool: `riantr/moonbit_doubleML@0.164.62` (moon 0.1.20261029). Panels reused unchanged; PLR with linear learners, n_folds=3, seed=42, cluster-robust on the image block. Acceptance: **sign + p<0.05 status**.

Frozen anchor: 1076 rule-consistent GT, 1070 detections, D=3/5/8/12 = 857/901/926/938.

| claim | n | blocks | theta (MoonBit) | theta (Python) | p (MoonBit) | p (Python) | sign | sig<.05 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| P1-a:edge | 1500 | 500 | +0.0020 | +0.0020 | 0.658 | 0.718 | OK | OK | **REPLICATED** |
| P1-a:blend | 1500 | 500 | +0.0020 | -0.0005 | 0.565 | 0.901 | MISMATCH | OK | **NOT-REPLICATED** |
| P1-b:tol | 2000 | 500 | +0.0011 | +0.0013 | 0.0824 | 0.051 | OK | OK | **REPLICATED** |
| P1-c:band | 2000 | 500 | -0.0092 | -0.0080 | 1.67e-05 | 0.000541 | OK | OK | **REPLICATED** |
| P1-d:joint | 1000 | 500 | +0.0000 | -0.0055 | 0.688 | 0.125 | MISMATCH | OK | **NOT-REPLICATED** |
| P2-a:any | 1076 | 480 | -0.2073 | -0.1818 | 1.66e-10 | 3.23e-09 | OK | OK | **REPLICATED** |
| P2-a:bin | 1076 | 480 | -0.0924 | -0.0807 | 2.47e-09 | 3.11e-08 | OK | OK | **REPLICATED** |
| P2-b:oracle | 1500 | 500 | +0.2183 | +0.2244 | 2.09e-09 | 4.55e-09 | OK | OK | **REPLICATED** |
| P2-b:wide | 1500 | 500 | +0.0181 | +0.0077 | 0.0589 | 0.429 | OK | OK | **REPLICATED** |
| P2-c:y5 | 1000 | 500 | +0.5360 | +0.5348 | 0 | 3.16e-53 | OK | OK | **REPLICATED** |
| P2-c:y12 | 1000 | 500 | +0.3801 | +0.3778 | 0 | 9.76e-31 | OK | OK | **REPLICATED** |
| P2-d:cap | 1500 | 500 | -0.0500 | -0.0535 | 0.000397 | 0.000375 | OK | OK | **REPLICATED** |
| P2-e:agree | 1070 | 480 | +0.0761 | +0.0517 | 0.000841 | 0.00548 | OK | OK | **REPLICATED** |
| stage1:red_mask | 500 | 500 | -2.8087 | -1.5714 | 0.061 | 0.385 | OK | OK | **REPLICATED** |
| stage2:denoise | 500 | 500 | -0.2338 | -1.7923 | 0.187 | 0.00458 | OK | MISMATCH | **NOT-REPLICATED** |
| stage3:outer_contours | 500 | 500 | -0.4347 | +0.4941 | 0.169 | 0.0906 | MISMATCH | OK | **NOT-REPLICATED** |
| stage4:base_quads | 500 | 500 | +0.1843 | +0.2826 | 0.095 | 0.175 | OK | OK | **REPLICATED** |
| stage5:admission | 500 | 500 | +1.8806 | +1.5988 | 1.29e-05 | 0.0924 | OK | MISMATCH | **NOT-REPLICATED** |
| stage6:dedup | 500 | 500 | -1.6247 | +1.0949 | 0.000327 | 0.371 | MISMATCH | MISMATCH | **NOT-REPLICATED** |

## Summary

- claims: **19**, replicated: **13**, sign matches: **15**, significance matches: **16**.
- wall clock: 26.2 s.

A NOT-REPLICATED claim is not automatically refuted: the MoonBit port uses linear nuisance learners while the Python panels used RandomForest PLR, so the point estimates and p-values can differ. The sign/significance test is the pre-registered robustness check.

## Interpretation

- **OR2 claims are robust (11/13).** The two misses are both null-effect claims
  (`P1-a:blend` original theta -0.0005 vs MoonBit +0.0020; `P1-d:joint` original
  -0.0055 vs MoonBit +0.0000). The sign of a zero effect is arbitrary, so the
  substantive conclusion — orthogonality factors are causally inert — is
  unchanged. The OR2 effects that mattered (overlap, oracle mask, corpus scale,
  cap, agreement) all replicate with the same sign and p<0.05 status.
- **The 6-stage observational panel is NOT robust (only stage1 and stage4
  replicate).** `stage2` (the original "only significant" claim, p=0.0046)
  loses significance under the MoonBit linear-learner PLR (p=0.187);
  `stage3` and `stage6` flip sign; `stage5` becomes significant in the opposite
  direction from the original point estimate. This is an estimator-sensitivity
  result: those stage features are collinear nested counts, and the original
  RandomForest PLR's stage-level significance does not survive a different
  nuisance learner. Do not cite the 6-stage significance claims as robust.
- **The three proposed recall levers are NOT-REPLICATED.** See
  `benchmark/loop45_levers_verdict.md`: `card` is a literal no-op, `side3` loses
  TP on two of three seeds, and `local` floods raw FPs (169 -> 518 on seed 0)
  for no significant TP gain.
- Limitation: the MoonBit port's default PLR learner is linear regression while
  the Python panels used RandomForest nuisances. The acceptance rule was
  deliberately sign + p<0.05, not numeric equality; the theta/CI deltas above
  are descriptive only.

## Refutation (placebo treatment permutation)

Each replicated claim was re-estimated with the treatment permuted 9 times (seed fixed). Empirical p = (1 + #{|theta_perm| >= |theta_obs|}) / (9 + 1); the resolution is 0.10.

| claim | theta observed | placebo range | p_placebo | status |
|---|---|---|---|---|
| P1-a:edge | +0.0020 | [-0.0662, +0.0914] | 1.00 | WEAK |
| P1-a:blend | - | - | - | NOT-ATTEMPTED |
| P1-b:tol | +0.0011 | [-0.0058, +0.0059] | 0.90 | WEAK |
| P1-c:band | -0.0092 | [-0.0147, +0.0319] | 0.50 | WEAK |
| P1-d:joint | - | - | - | NOT-ATTEMPTED |
| P2-a:any | -0.2073 | [-0.0222, +0.0265] | 0.10 | PASS |
| P2-a:bin | -0.0924 | [-0.0119, +0.0114] | 0.10 | PASS |
| P2-b:oracle | +0.2183 | [-0.0746, +0.0412] | 0.10 | PASS |
| P2-b:wide | +0.0181 | [-0.0450, +0.0386] | 0.90 | WEAK |
| P2-c:y5 | +0.5360 | [-0.0624, +0.0321] | 0.10 | PASS |
| P2-c:y12 | +0.3801 | [-0.0551, +0.1047] | 0.10 | PASS |
| P2-d:cap | -0.0500 | [-0.0565, +0.0317] | 0.20 | WEAK |
| P2-e:agree | +0.0761 | [-0.0019, +0.0301] | 0.10 | PASS |
| stage1:red_mask | -2.8087 | [-3.2546, +1.5996] | 0.20 | WEAK |
| stage2:denoise | - | - | - | NOT-ATTEMPTED |
| stage3:outer_contours | - | - | - | NOT-ATTEMPTED |
| stage4:base_quads | +0.1843 | [-0.0039, +0.0228] | 0.10 | PASS |
| stage5:admission | - | - | - | NOT-ATTEMPTED |
| stage6:dedup | - | - | - | NOT-ATTEMPTED |
