# LOOP-46 — lever C (hysteresis red mask)

Phase 0 was revised by the user (2026-11-05) from **min-side** to **mean-side support >= 0.6**: **33 eligible FN** (no_proposal 21, corner_gt5 11, gate_ortho 1). The dominant bucket is `no_proposal` = 21, so lever C is the only eligible lever.

| seed | status | theta | p | D5 TP shipped -> lever | D5 trueFP | trueFP/TP |
|---|---|---|---|---|---|---|
| 0 | DONE | +0.0180 | 0.0194 | 901 -> 910 | 134 -> 120 | -1.556 |
| 1 | DONE | -0.0000 | 0.998 | 950 -> 950 | 156 -> 157 | None |
| 2 | DONE | +0.0120 | 0.0825 | 911 -> 917 | 161 -> 163 | 0.333 |

**Verdict: NOT-REPLICATED.** seed 0: +9 TP (901->910), trueFP 134->120, theta +0.0180 p=0.0194 (passes); seed 1: 0 TP (950->950), theta -0.0000 p=0.998 (fails); seed 2: +6 TP (911->917), theta +0.0120 p=0.0825 (fails p<0.05). The tuning-set win does not survive the held-out check -> NOT-REPLICATED.

Base arm (seed 0): **901 TP / 169 rawFP / 1070 dets** — matches the frozen anchor.

### Seed 0 per-D (corrected TP / trueFP)

| D | shipped | hyst | delta TP | delta trueFP |
|---|---|---|---|---|
| 3 | 857 / 177 | 870 / 157 | +13 | -20 |
| 5 | 901 / 134 | 910 / 120 | +9 | -14 |
| 8 | 926 / 113 | 931 / 103 | +5 | -10 |
| 12 | 938 / 103 | 944 / 93 | +6 | -10 |

## All six levers (routing rule: only the dominant bucket may run)

| lever | name | target bucket | status | reason |
|---|---|---|---|---|
| C | hysteresis red mask | no_proposal | **NOT-REPLICATED** | seed 0: +9 TP (901->910), trueFP 134->120, theta +0.0180 p=0.0194 (passes); seed 1: 0 TP (950->950), theta -0.0000 p=0.998 (fails); seed 2: +6 TP (911->917), theta +0.0120 p=0.0825 (fails p<0.05). The tuning-set win does not survive the held-out check -> NOT-REPLICATED. |
| A | overlap-aware dedup | dedup | **NOT-ATTEMPTED** | census dedup bucket = 0; not the dominant bucket |
| B | cross-family agreement rescue | gate_side / gate_support | **NOT-ATTEMPTED** | census gate_side=0, gate_support=0; not the dominant bucket |
| D | ridge-centre rescue of rejects | gate_support / gate_side | **NOT-ATTEMPTED** | census gate_support=0, gate_side=0; not the dominant bucket |
| E | cardinality-constrained ranking | no_proposal | **NOT-ATTEMPTED** | same bucket as C; only one lever from the dominant bucket was run (no stacked hybrid) |
| F | corner snap (5-8 px band) | corner_gt5 | **NOT-ATTEMPTED** | census corner_gt5=11; not the dominant bucket |

**Conclusion:** the hysteresis mask is a tuning-set win (+9 TP / -14 trueFP at D5; +13 at D3) but does not survive held-out seeds 1/2 (flat / +6 TP p=0.082). It is not adopted. The other five levers are ineligible under the pre-registered routing rule; do not run them on this headroom.
