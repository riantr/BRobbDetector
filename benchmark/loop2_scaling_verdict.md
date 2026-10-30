# loop2 scaling verdict — GPU-WARRANTED
- Both arms measured by the same script (`proto/cs2_scaling_probe.py`), same model/epochs/imgsz/batch/workers/conf and the same RandomState(7) split; arm A's 300 images are a strict subset of arm B's 2,000 and the 100 validation images are identical.
- **Arm A (300 images):** D=5 **234 TP / 638 trueFP / corr_f1 24.0** (D=3 53/808/5.5, D=8 44.9, D=12 63.5); val-split 49 TP / f25.5; training **6.0 min** at `tmp_loop2_scale_a`.
- **Arm B (2,000 images = the 300 + 1,700 drawn from `Label@CARS_202609_raw`):** D=5 **502 TP / 436 trueFP / corr_f1 49.9** (D=3 212/711/21.2, D=8 68.0, D=12 80.2); val-split 108 TP / f53.7; training **61.8 min** at `tmp_loop2_scale`.
- Arm A reproduces the previously cited anchor exactly (234/638/24.0, val 49/25.5), which is the proof that both arms share one pipeline and one split; the split also re-derives the anchor's exact 889 labels.
- The gain appears on the **held-out** val split too (49 → 108 TP), so it is data scaling, not train-set memorisation.
- **Slope: +157.6 TP@D=5 per 1,000 training images** (2.145x TP for 6.7x data) — roughly linear over 300–2,000 images, not saturating in this range.
- **Rule, fixed before measurement:** TP@D=5 must grow >= 1.4x (>= 328 TP) ⇒ observed **2.145x** ⇒ **GPU-WARRANTED**.
- **Next action:** commission a full-corpus GPU run (640 px, more epochs). On this slope pm_v10's 901 TP needs ~4,500 training images, and ~56k labelled images exist — the corpus is not the constraint.
- **Caveat:** the slope covers 300–2,000 images only; re-run this probe as a 2,000 → 8,000 check before committing to a large run.