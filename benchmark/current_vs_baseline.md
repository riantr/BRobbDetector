# Current best vs baseline (2026-11-05)

Current best = shipped C# `cvhybrid` = **pm_v10**. Baseline = frozen `benchmark/pm_v10_refine.json`.
All rows use the rule-consistent **1076 GT** and point-match (all 4 corners within D); trueFP is GT-corrected.

## 1. Identity check — current best vs frozen baseline

| view | GT | D=3 (TP/trueFP/recall/F1) | D=5 | D=8 | D=12 |
|---|---|---|---|---|---|
| frozen pm_v10 | 1076 | 857 / 177 / 79.6 / 81.2 | 901 / 134 / 83.7 / 85.4 | 926 / 113 / 86.1 / 87.6 | 938 / 103 / 87.2 / 88.6 |
| live C# cvhybrid | 1076 | 857 / 177 / 79.6 / 81.2 | 901 / 134 / 83.7 / 85.4 | 926 / 113 / 86.1 / 87.6 | 938 / 103 / 87.2 / 88.6 |

**Identical at every D; 1070 detections; parity `py=1070 cs=1070 mismatched=0` — no regression.**

## 2. Current best vs the previous baseline (pm_v9)

| model | GT | D=3 | D=5 | D=8 | D=12 |
|---|---|---|---|---|---|
| pm_v9 | 1076 | 842 / 204 / 78.3 / 79.4 | 899 / 149 / 83.6 / 84.7 | 922 / 129 / 85.7 / 86.7 | 937 / 118 / 87.1 / 87.9 |
| pm_v10 (current best) | 1076 | 857 / 177 / 79.6 / 81.2 | 901 / 134 / 83.7 / 85.4 | 926 / 113 / 86.1 / 87.6 | 938 / 103 / 87.2 / 88.6 |

Delta (pm_v10 − pm_v9): TP **+15 / +2 / +4 / +1**; trueFP **−27 / −15 / −16 / −15**. pm_v10 wins at every D.

## 3. LOOP-46 candidate C (hysteresis mask) — NOT ADOPTED

| model | GT | D=3 | D=5 | D=8 | D=12 |
|---|---|---|---|---|---|
| pm_v10 | 1076 | 857 / 177 / 79.6 / 81.2 | 901 / 134 / 83.7 / 85.4 | 926 / 113 / 86.1 / 87.6 | 938 / 103 / 87.2 / 88.6 |
| LOOP-46 C, seed 0 | 1076 | 870 / 157 / 80.9 / 82.7 | 910 / 120 / 84.6 / 86.4 | 931 / 103 / 86.5 / 88.2 | 944 / 93 / 87.7 / 89.4 |

Seed 0 delta: TP **+13 / +9 / +5 / +6**; trueFP **−20 / −14 / −10 / −10**; theta +0.0180 p=0.0194.
But held-out: seed 1 950→950 TP (p=0.998), seed 2 911→917 TP (p=0.0825). **Verdict NOT-REPLICATED; not adopted.**

## 4. Historical lineage (denominators differ)

| model | GT | D=3 | D=5 | D=8 | D=12 |
|---|---|---|---|---|---|
| pm_v7 (raw GT) | 1121 | 579 / 49 / 51.7 / 66.2 | 602 / 27 / 53.7 / 68.8 | 605 / 24 / 54.0 / 69.1 | 607 / 22 / 54.1 / 69.4 |
| pm_v8 (raw GT) | 1121 | 859 / 237 / 76.6 / 77.5 | 926 / 172 / 82.6 / 83.5 | 960 / 140 / 85.6 / 86.4 | 978 / 124 / 87.2 / 88.0 |
| pm_v9 (rule-consistent) | 1076 | 842 / 204 / 78.3 / 79.4 | 899 / 149 / 83.6 / 84.7 | 922 / 129 / 85.7 / 86.7 | 937 / 118 / 87.1 / 87.9 |
| pm_v10 (rule-consistent) | 1076 | 857 / 177 / 79.6 / 81.2 | 901 / 134 / 83.7 / 85.4 | 926 / 113 / 86.1 / 87.6 | 938 / 103 / 87.2 / 88.6 |

pm_v7/v8 are scored on the **raw 1121-box GT**; pm_v9/v10 on the **rule-consistent 1076 GT** — do not compare across denominators.

## Conclusion

- The shipped implementation is **bit-identical to the frozen baseline** (1070 dets, 857/901/926/938, parity 0).
- The current best **remains pm_v10**; it beats the previous baseline pm_v9 at every D.
- The only candidate with a tuning-set gain (LOOP-46 C) fails the held-out check and is **not adopted**.
- No code change was made by LOOP-45/46; the baseline is still the production detector.
