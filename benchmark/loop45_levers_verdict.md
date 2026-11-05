# LOOP-45 — new recall-lever arms (MoonBit DoubleML PLR)

Acceptance: theta > 0 and p < 0.05 on seed 0, repeated on held-out seeds 1/2; otherwise NOT-REPLICATED.

Base arm (seed 0, shipped): **901 TP / 169 rawFP / 1070 detections** — reproduces the frozen pm_v10 anchor.

## seed 0 (500 images)

| lever | theta | p | D5 TP shipped -> lever | D5 rawFP shipped -> lever | verdict |
|---|---|---|---|---|---|
| card | +0.0000 | 0.724 | 901 -> 901 | 169 -> 169 | **NOT-REPLICATED** |
| side3 | -0.0060 | 0.0823 | 901 -> 898 | 169 -> 164 | **NOT-REPLICATED** |
| local | +0.0000 | 0.724 | 901 -> 901 | 169 -> 518 | **NOT-REPLICATED** |

## seed 1 (500 images)

| lever | theta | p | D5 TP shipped -> lever | D5 rawFP shipped -> lever | verdict |
|---|---|---|---|---|---|
| card | -0.0000 | 0.642 | 950 -> 950 | 184 -> 187 | **NOT-REPLICATED** |
| side3 | -0.0100 | 0.131 | 950 -> 945 | 184 -> 177 | **NOT-REPLICATED** |
| local | +0.0060 | 0.0823 | 950 -> 953 | 184 -> 602 | **NOT-REPLICATED** |

## seed 2 (500 images)

| lever | theta | p | D5 TP shipped -> lever | D5 rawFP shipped -> lever | verdict |
|---|---|---|---|---|---|
| card | +0.0000 | 0.678 | 911 -> 911 | 186 -> 190 | **NOT-REPLICATED** |
| side3 | +0.0020 | 0.317 | 911 -> 912 | 186 -> 171 | **NOT-REPLICATED** |
| local | +0.0040 | 0.157 | 911 -> 913 | 186 -> 579 | **NOT-REPLICATED** |

## Verdict

- **card (cardinality-triggered admission): NOT-REPLICATED / literal no-op.** The detection sets are unchanged on seed 0 (901/169) and the TP is unchanged on seeds 1/2; only 0-4 extra raw FPs appear. The cardinality prior did not trigger in a way that recovers a missed box.
- **side3 (3-of-4 side-support gate): NOT-REPLICATED / harmful.** Seed 0 loses 3 TP (901 -> 898); seed 1 loses 5 TP; seed 2 gains 1 TP. It buys 5-15 fewer raw FPs but fails the TP-primary gate and flips sign across samples.
- **local (local ROI adaptive redness): NOT-REPLICATED / precision catastrophe.** TP barely moves (901 -> 901, 950 -> 953, 911 -> 913) while rawFP explodes (169 -> 518, 184 -> 602, 186 -> 579); p > 0.05 on every seed. The local Otsu redness mask is not a usable proposal source.

**Conclusion:** none of the three proposed recall levers survives the pre-registered sign + p<0.05 test. Do not adopt them; the earlier non-GPU headroom statement remains a ceiling, not a demonstrated gain.

## Per-D metrics (TP / rawFP, shipped -> lever)

### seed 0

| lever | D=3 | D=5 | D=8 | D=12 |
|---|---|---|---|---|
| card | 857/213 -> 857/213 | 901/169 -> 901/169 | 926/144 -> 926/144 | 938/132 -> 938/132 |
| side3 | 857/213 -> 855/207 | 901/169 -> 898/164 | 926/144 -> 923/139 | 938/132 -> 935/127 |
| local | 857/213 -> 857/562 | 901/169 -> 901/518 | 926/144 -> 926/493 | 938/132 -> 940/479 |

### seed 1

| lever | D=3 | D=5 | D=8 | D=12 |
|---|---|---|---|---|
| card | 890/244 -> 890/247 | 950/184 -> 950/187 | 973/161 -> 973/164 | 1001/133 -> 1003/134 |
| side3 | 890/244 -> 889/233 | 950/184 -> 945/177 | 973/161 -> 965/157 | 1001/133 -> 992/130 |
| local | 890/244 -> 891/664 | 950/184 -> 953/602 | 973/161 -> 978/577 | 1001/133 -> 1009/546 |

### seed 2

| lever | D=3 | D=5 | D=8 | D=12 |
|---|---|---|---|---|
| card | 865/232 -> 865/236 | 911/186 -> 911/190 | 937/160 -> 939/162 | 959/138 -> 961/140 |
| side3 | 865/232 -> 866/217 | 911/186 -> 912/171 | 937/160 -> 938/145 | 959/138 -> 957/126 |
| local | 865/232 -> 866/626 | 911/186 -> 913/579 | 937/160 -> 940/552 | 959/138 -> 964/528 |
