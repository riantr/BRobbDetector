# loop2 verdicts — path 1 → 2 → 3 → 4 → 5 (automatic methods only)

Goal: break the pm_v10 ceiling using only automatic methods — no human
annotation, no human review, no manual GT confirmation.

**Result: 1 DEAD, 2 DEAD, 3 DELIVERED (and it invalidated its own premise),
4 and 5 DEAD-BY-EVIDENCE. No detector change was adopted. The single most
important finding is that the labels were never the bottleneck: ~56k labelled
real images already exist.**

## Attribution anchor (Done-when #1)

`benchmark/loop2_baseline_and_premise.json` — `baseline.ok = true`,
`premise_verified = true`:

```
parity: py=1070 cs=1070 mismatched images=0
```

| D | TP | trueFP | real_unl | recall | corr_f1 | matches pm_v10_refine.json |
|---|---|---|---|---|---|---|
| 3 | 857 | 177 | 36 | 79.6 | 81.2 | OK |
| 5 | 901 | 134 | 35 | 83.7 | 85.4 | OK |
| 8 | 926 | 113 | 31 | 86.1 | 87.6 | OK |
| 12 | 938 | 103 | 29 | 87.2 | 88.6 | OK |

**Premise of step 1, verified:** `BRobbDetector/Assets/yolov8s_obb.onnx` mtime
**2026-10-12 17:14**; `proto/common.py::LAB` was re-pointed to
`Label@CARS_202609_raw` in commit **`d3b6896` (2026-10-21)**. The shipped
checkpoint predates the re-pointing by nine days, so every published YOLO score
was measured against Compare-V1 — the detector's own output.

## Verdict table

| step | hypothesis | harness → artifact | base-arm Δ | D=3 TP/trueFP/corr_f1 | D=5 TP/trueFP/corr_f1 | D=8 | D=12 | verdict | numeric reason |
|---|---|---|---|---|---|---|---|---|---|
| **1** | The shipped/retrained OBB detector is competitive once scored on the TRUE GT | `proto/dl2_yolo_truegt.py` → `benchmark/loop2_dl_truegt.json` | 0 (frozen table reproduced) | shipped 32/825/3.3 → retrained 53/808/5.5 | shipped **140/722/14.4** → retrained **234/638/24.0** | 302/567/31.1 → 441/447/44.9 | 463/414/47.4 → 628/274/63.5 | **DEAD** | Retrained TP@D=5 **234 vs pm_v10's 901** (−667, a 3.9× gap). The published "recall 97.7 / corrected_f1 97.5" was an artifact of scoring against Compare-V1. Pre-stated rule (TP@D=5 < 901 ⇒ steps 4-5 DEAD-BY-EVIDENCE) applied mechanically. |
| **2** | The per-image CSV expected count can prune excess detections | `proto/cp2_cardinality.py` → `benchmark/loop2_cardinality.json` | 0 | base 857/177/81.2 → cap 838/134/81.8 | base **901/134/85.4** → cap **876/97/85.5** | 926/113/87.6 → 895/82/87.2 | 938/103/88.6 → 904/75/88.0 | **DEAD** | Prior is sound but marginal (exact 404/500 = **80.8%**, disagreement 19.2% vs the 20% stop-if). Capping removes 70 dets: **−25 TP** for −37 trueFP, corr_f1 +0.1. TP-primary gate fails; deleting 25 real boxes to buy 37 FPs is not a trade. |
| **3** | Labels can be generated at ~100× scale from the ink | `proto/pl2_pseudo_labels.py` → `benchmark/loop2_pseudo_labels.json` | 0 | — | — | — | — | **DELIVERED — premise refuted** | The unlabelled pool (image + CSV count + no coordinate label) is only **239 images**, not ~56k; all 239 processed, 138 quads, 54.0% of images with ≥1 box, CSV-count exact 51.5%, support p50 0.923. **But the corpus inventory shows the labels already exist:** `Label@CARS_202609_raw` holds **62,218 json label files**, 99.6% non-empty, **~141,392 boxes**. Pseudo-labelling at scale is unnecessary — and it would inherit pm_v10's 83.7% recall ceiling anyway. |
| **4** | Redesign the admission rule so a high-recall proposer's boxes survive | (not attempted) → `benchmark/loop2_acceptance_rule.json` | n/a | — | — | — | — | **DEAD-BY-EVIDENCE** | The proposal surplus does not exist: on the true GT the proposer yields 234 TP@D=5 vs the pipeline's 901. S5's "the gate is binding" (311/936) was measured with the Compare-V1-scored checkpoint; on the real target the proposer is the limiter. |
| **5** | Re-open corner-aware training now that the true GT is a clean target | (not attempted) → `benchmark/loop2_corner_retrain.json` | n/a | — | — | — | — | **DEAD-BY-EVIDENCE** | Premise stands (P3: GT corners reproducible to ≤1.32 px; LOOP-9 was scored on the invalid GT). But refining corners presupposes a detector that finds the box: 234 → 901 is a detection gap, not a corner gap. |

## Final summary (Done-when #7 analogue)

| step | ΔTP@D=5 | ΔtrueFP@D=5 | Δcorr_f1@D=5 | shipped detector changed? |
|---|---|---|---|---|
| 1 re-baseline + retrain | **−667** | +504 | −61.4 | **no** |
| 2 cardinality prior | **−25** | −37 | +0.1 | **no** |
| 3 pseudo-labels | n/a — corpus artifact | n/a | n/a | **no** |
| 4 acceptance rule | not attempted (DEAD-BY-EVIDENCE) | — | — | **no** |
| 5 corner retrain | not attempted (DEAD-BY-EVIDENCE) | — | — | **no** |
| **shipped detector** | **baseline 901 TP / 134 trueFP / 85.4** | — | — | **UNCHANGED** |

## What was actually learned

1. **Every DL number in this repo was circular.** The checkpoint was trained and
   scored on Compare-V1, which is the detector's own output. On the true GT the
   shipped model scores **14.4** corr_f1, not 97.5. Any plan that rests on "YOLO
   has 97.7% recall" must be abandoned.
2. **Labels were never the bottleneck.** ~56k labelled real images
   (~141k boxes) are already on disk; the benchmark and the retrain both used a
   500/300-image slice of them. This is the answer to "should we add real data":
   the data is there — the work is to *use* it.
3. **The retrain was compute-starved, not data-starved.** A 20-epoch 640 px
   yolov8s run projected at ~8 h on CPU (no GPU) and was killed at the 90-minute
   cap; the reported 234 TP comes from 300 images / 8 epochs / 384 px /
   yolov8n-obb. **The DEAD-BY-EVIDENCE verdict for steps 4-5 is therefore a
   verdict at this compute budget, not a refutation of the DL path.**
4. **The cardinality prior is real but weak.** CSV counts agree with the true GT
   on only 80.8% of images; using them as a hard cap costs more TP than it saves
   FP.

## Honest limits of this record

- **Precision is not certifiable without humans** (goal constraint). Only recall
  against the 1076-box GT is used to judge progress; the trueFP column is
  reported but is a range, because a missing label and a false positive are
  undecidable automatically.
- No auto-generated label was used as an acceptance metric anywhere in this
  loop; the pseudo-label corpus (step 3) is reported as a corpus plus its
  inherited ceiling, never as evidence of improvement.
- The DL steps ran at a deliberately shrunk budget under the goal's compute cap;
  a GPU or an overnight run could change step 1's number — that is the one
  experiment this loop could not settle.

## Artifacts

`benchmark/loop2_baseline_and_premise.json`, `loop2_dl_truegt.json`,
`loop2_cardinality.json`, `loop2_pseudo_labels.json`,
`loop2_acceptance_rule.json`, `loop2_corner_retrain.json`,
`loop2_path_verdicts.md`. Harnesses: `proto/dl2_baseline_premise.py`,
`proto/dl2_yolo_truegt.py`, `proto/cp2_cardinality.py`,
`proto/pl2_pseudo_labels.py`. Training corpus / checkpoints / pseudo-label
manifest live under `tmp_loop2_*` (gitignored).

**No detector code, GT file or frozen baseline was modified. Nothing was
committed or pushed.**