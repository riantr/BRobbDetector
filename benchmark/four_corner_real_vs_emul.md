# Current four-corner algorithm — real vs simulated data

Algorithm: **pm_v10 shipped** (holes + edgefit + side refine), no overrides. Same point-match metric in both runs (TP = all 4 corners within D px).

| D | real: TP / trueFP | real R / P / F1 | emul: TP / FP | emul R / P / F1 | ΔF1 (emul − real) |
|---|---|---|---|---|---|
| **3** | 857 / 177 | 79.6 / 82.9 / **81.2** | 2544 / 622 | 77.2 / 80.3 / **78.7** | -2.5 |
| **5** | 901 / 134 | 83.7 / 87.1 / **85.4** | 2807 / 359 | 85.2 / 88.7 / **86.9** | +1.5 |
| **8** | 926 / 113 | 86.1 / 89.1 / **87.6** | 3010 / 156 | 91.3 / 95.1 / **93.2** | +5.6 |
| **12** | 938 / 103 | 87.2 / 90.1 / **88.6** | 3074 / 92 | 93.3 / 97.1 / **95.2** | +6.6 |

Denominators: **real** 1076 rule-consistent GT boxes on 500 images (1070 detections, 2.14/img); **emulation** 3295 RED-class GT boxes on 5000 images (3166 detections, 0.63/img). The emulation set has 9982 GT boxes in total, but only the red-class subset is the correct denominator for a red-only detector.

**Metric caveat:** the real column uses the GT-corrected metric (its `fp` is trueFP); the emulation GT is complete, so its column is plain P/R. Do not compare the `fp` counts across the two columns — compare the rates.

Reading: at **D=5** the two datasets agree within ~1.5 points of F1 (real 85.4 vs emul 86.9), which is why the emulation is a valid proxy for D=5 structural tuning. They diverge away from it: **real wins at D=3 by 2.5 F1** because the hand-drawn GT is precise (best-orthogonal fit p50 0.50 px), while the emulation is **optimistic at D>=8 by 5.6 (D8) and 6.6 (D12)** because its boxes are exact rectangles whereas real annotations are wobbled and arrow-corrupted.
