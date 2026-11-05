# LOOP-51 — D=3 validity-constrained corner snap

Target: D=3 corner precision (base 857 TP); D=5 must not regress (base 901).
One lever: snap each corner to the local `R-max(G,B)` ridge peak within
+-2 px, keep only if the shipped validity gate passes.

| seed | D3 TP shipped -> lever | D3 trueFP | D5 TP shipped -> lever | theta3 | p | snapped |
|---|---|---|---|---|---|---|
| 0 | 857 -> 758 | 177 -> 267 | 901 -> 901 | -0.1981 | 0 | 1064 |
| 1 | 890 -> 783 | 213 -> 318 | 950 -> 947 | -0.2140 | 0 | 1128 |
| 2 | 865 -> 765 | 200 -> 292 | 911 -> 913 | -0.2000 | 0 | 1096 |

**Verdict: DEAD.**

Base arm (seed 0): D3 857 / D5 901 TP / 169 rawFP / 1070 dets (frozen 857 / 901 / 169 / 1070).

## Reading

- The ridge peak at a **corner** is not the annotated corner: it is the stroke
  intersection, and the shipped quad's corner is already the better estimate.
  Snapping **moves ~1,050–1,130 quads per sample** and destroys ~100 D=3 TP.
- D=5 is flat (901→901, 950→947, 911→913), so the loss is purely corner
  precision — the opposite of the intended effect.
- This independently confirms LOOP-44: the `R-max(G,B)` ridge argmax is an
  unbiased stroke-centre locator on a **cross-section**, but it is not a valid
  corner refiner for the shipped TLS quad. Do not re-propose ridge-based corner
  or side refinement.
- The D=3 gap (857 vs 901) is therefore **not** recoverable by a classical
  post-pass on the same evidence. The remaining D=3 error is the same "which
  pixels the band selects" problem measured in P2/P3; the only credible paths
  left are DL/GPU corner regression (GPU excluded) or GT-scope changes, not
  another classical corner lever.

## Verdict

**DEAD.** Base arm (seed 0): D3 857 / D5 901 TP / 169 rawFP / 1070 dets —
frozen anchor reproduced. No detector/GT/data/C# changes; nothing committed.
