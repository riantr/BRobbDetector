# LOOP-50 — hole-first / edgefit-fallback

Target: candidate clutter (`S4_quads` -10.2 pp/SD, `S5_edgefit_share` -9.0).
One lever: run holes first; edgefit only when the hole family emits nothing.

| seed | status | theta | p | D5 TP shipped -> lever | D5 rawFP | trueFP | trueFP/TP |
|---|---|---|---|---|---|---|---|
| 0 | DONE | -0.0100 | 0.0246 | 901 -> 896 | 169 -> 163 | 134 -> 129 | None |
| 1 | DONE | -0.0080 | 0.0447 | 950 -> 946 | 184 -> 177 | 156 -> 151 | None |
| 2 | DONE | -0.0040 | 0.156 | 911 -> 909 | 186 -> 175 | 161 -> 151 | None |

**Verdict: DEAD.** seed 0: 901 -> 896 TP (-5), trueFP 134 -> 129, theta -0.0100, p=0.0246; seed 1: 950 -> 946 TP (-4), trueFP 156 -> 151, theta -0.0080, p=0.0447; seed 2: 911 -> 909 TP (-2), trueFP 161 -> 155, theta -0.0040, p=0.156. The lever trades recall for precision; the seed-0 acceptance fails (theta<0) -> DEAD.

Base arm (seed 0): **901 TP / 169 rawFP / 1070 dets** — frozen anchor reproduced.

### Seed 0 per-D (TP shipped -> lever)

| D | TP shipped -> lever | rawFP |
|---|---|---|
| 3 | 857 -> 853 | 213 -> 206 |
| 5 | 901 -> 896 | 169 -> 163 |
| 8 | 926 -> 921 | 144 -> 138 |
| 12 | 938 -> 933 | 132 -> 126 |

### Reading

- The lever reduces trueFP by 5 at every D but **loses 5 TP on seed 0** (4 on seed 1, 2 on seed 2); it is a precision trade, not a recall gain.
- Edgefit contributes unique true boxes even when the hole family is non-empty; removing it is not a safe clutter reduction.
- The all-cause `S5_edgefit_share` association is **not causal for removal**: conditioning on the share is not the same as deleting the family.
- Do not adopt. The candidate-clutter direction is closed in the image-level hole-first form; do not re-run global hue, fragment merge, cut-side sliding, joint refit, or per-image config.

Artifacts: `benchmark/loop50_holefirst.json`, `benchmark/loop50_holefirst_verdict.md`, `benchmark/loop50_panels/seed{0,1,2}_holefirst.csv`, `proto/loop50_holefirst.py`. No detector/GT/data/C# changes.
