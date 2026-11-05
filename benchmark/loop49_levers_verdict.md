# LOOP-49 — dense-cluster second-box proposal

## Phase 0 — overlapping-FN census (124 rows)

| bucket | count |
|---|---|
| no_proposal | 98 |
| gate_side | 0 |
| gate_ortho | 0 |
| gate_support | 0 |
| dedup | 0 |
| corner_gt5 | 26 |

Second box proposed: **D5 = 0, D12 = 26**; no-proposal **100.0% at D5 (100%), 79.0% at D12**.

Live anchor: **1070 dets**, parity `py=1070 cs=1070 mismatched=0`, D=3/5/8/12 = **857/901/926/938**.

Stop interpretation: **D5 operational reading: 0/124 proposed = 100% no-proposal > 80% -> the overlap class is definitional. D12 "ever proposed" reading: 26/124 = 21% -> 79.0% no-proposal, just under 80%, so the lever was run; it added 10/7/20 quads but 0 TP and trueFP rose, so both readings close the line.**

## Phase 1 — the only eligible lever (orientation-aware concave-corner split)

| seed | added quads | theta | p | D5 TP shipped -> lever | D5 rawFP | trueFP | trueFP/TP |
|---|---|---|---|---|---|---|---|
| 0 | 10 | +0.0000 | 0.724 | 901 -> 901 | 169 -> 175 | 134 -> 135 | None |
| 1 | 7 | -0.0000 | 0.642 | 950 -> 950 | 184 -> 189 | 156 -> 160 | None |
| 2 | 20 | +0.0000 | 0.678 | 911 -> 911 | 186 -> 200 | 161 -> 170 | None |

**Verdict: DEAD.** seed 0: 901 -> 901 TP (0 gain), trueFP 134 -> 135, added 10 split quads, theta 0.0000, p=0.724; seed 1: 950 -> 950, trueFP 156 -> 160; seed 2: 911 -> 911, trueFP 161 -> 170. The seed-0 Stop-if (TP gain 0 while trueFP rises) is satisfied. DEAD.

Stop: **seed-0 TP gain is 0 while trueFP rises**.

Base arm (seed 0): **901 TP / 169 rawFP / 1070 dets**; matches the frozen anchor.

## All levers

| lever | name | target bucket | status | reason |
|---|---|---|---|---|
| dense-cluster split | orientation-aware concave-corner second-box proposal | no_proposal | **DEAD** | seed 0: 901 -> 901 TP (0 gain), trueFP 134 -> 135, added 10 split quads, theta 0.0000, p=0.724; seed 1: 950 -> 950, trueFP 156 -> 160; seed 2: 911 -> 911, trueFP 161 -> 170. The seed-0 Stop-if (TP gain 0 while trueFP rises) is satisfied. DEAD. |
| A | overlap-aware dedup | dedup | **NOT-ATTEMPTED** | not the dominant bucket; census dedup = 0 |
| B | cross-family agreement rescue | gate_side / gate_support | **NOT-ATTEMPTED** | not the dominant bucket; census gate_side / gate_support = 0 |
| D | ridge-centre rescue of rejects | gate_support / gate_side | **NOT-ATTEMPTED** | not the dominant bucket; census gate_support / gate_side = 0 |
| F | corner snap (5-8 px band) | corner_gt5 | **NOT-ATTEMPTED** | not the dominant bucket; census corner_gt5 = 26 |

## Reading

- The overlap class is **definitional**: 0/124 second boxes are proposed at D=5; the split adds quads but recovers **0 TP**.
- The orientation-aware concave selection produced 10/7/20 quads on seeds 0/1/2, all false or neutral; trueFP rises, so the seed-0 Stop-if fires.
- Do not re-propose fragment/hole merging, pairwise overlap merge, cut-side sliding, joint re-fitting, or per-image config selection; the overlap second-box proposal is now measured too.
- Next actionable association after overlap: **candidate clutter** (`S4_quads` -10.2 pp/SD, `S5_edgefit_share` -9.0) via a hole-first / edgefit-fallback test.

Artifacts: `benchmark/loop49_overlap_census.json`, `benchmark/loop49_levers.json`, `benchmark/loop49_panels/seed{0,1,2}_split.csv`, `proto/loop49_overlap_census.py`, `proto/loop49_levers.py`. No detector/GT/data/C# changes.
