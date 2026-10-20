"""LOOP-31c: Re-evaluate dedup_iou sweep with corrected_f1 metric.

LOOP-31b showed TP increasing monotonically with dedup_iou, but raw_fp
also exploding (293→1172 at dedup_iou=1.0). This is likely metric gaming
from keeping duplicate detections. The corrected_f1 metric (which classifies
FPs as true_fp vs real_unlabeled) will reveal the true optimum.

Also re-evaluates boundary_band with corrected_f1.

Usage: python proto/cv_scope31c.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_hybrid import outer_contours, _pass, dedup_keep_order, ROOT
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope18_real import evaluate_det_map
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

D_EVAL = 5.0
BASELINE_TP = 502
BASELINE_F1 = 71.5

DEDUP_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0]
BND_GRID = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0]


def base_params():
    p = pm5_params()
    p["min_area"] = 1000.0
    p["smin"] = 80
    p["vmin"] = 110
    return p


def detect(img, p):
    H, W = img.shape[:2]
    mask = red_mask(img, p["smin"], p["vmin"])
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    cnts = outer_contours(mask, p)
    all_quads = []
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                all_quads.append(q)
    return dedup_keep_order(all_quads, p["dedup_iou"])


def main():
    ids = get_ids()
    exp = load_expected()
    print(f"LOOP-31c: corrected_f1 evaluation on {len(ids)} imgs, D={D_EVAL:.0f}")
    print(f"Baseline: TP={BASELINE_TP}, corr_f1={BASELINE_F1}")

    print("Loading images...", end=" ", flush=True)
    images = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is not None:
            images[iid] = img
    print(f"{len(images)} loaded")

    p = base_params()

    # Phase 1: dedup_iou sweep with corrected_f1 (baseline boundary_band=12)
    print(f"\n{'='*80}")
    print(f"Phase 1: dedup_iou sweep with corrected_f1 (boundary_band=12)")
    print(f"{'='*80}")
    print(f"  {'dedup':>6s} {'TP':>5s} {'n_det':>6s} {'raw_fp':>7s} {'true_fp':>8s} "
          f"{'real_unl':>9s} {'recall':>7s} {'corr_f1':>8s} {'dTP':>5s} {'df1':>6s}")
    dedup_rows = []
    best_dedup = 0.30
    best_f1 = BASELINE_F1
    for val in DEDUP_GRID:
        p2 = dict(p)
        p2["dedup_iou"] = val
        det_map = {}
        for iid in ids:
            det_map[iid] = detect(images[iid], p2) if iid in images else []
        m = evaluate_det_map(det_map, ids, D_EVAL)
        ndet = sum(len(v) for v in det_map.values())
        df1 = m["corrected_f1"] - BASELINE_F1
        dtp = m["tp"] - BASELINE_TP
        marker = " *" if val == 0.30 else ""
        print(f"  {val:6.2f} {m['tp']:5d} {ndet:6d} {m['raw_fp']:7d} {m['true_fp']:8d} "
              f"{m['real_unlabeled']:9d} {m['recall']:6.1f}% {m['corrected_f1']:7.1f}% "
              f"{dtp:+5d} {df1:+6.1f}{marker}")
        dedup_rows.append({"dedup_iou": val, "tp": m["tp"], "n_det": ndet,
                           "raw_fp": m["raw_fp"], "true_fp": m["true_fp"],
                           "real_unlabeled": m["real_unlabeled"],
                           "recall": m["recall"],
                           "corrected_f1": m["corrected_f1"]})
        if m["corrected_f1"] > best_f1:
            best_f1 = m["corrected_f1"]
            best_dedup = val

    print(f"\n  Best dedup_iou = {best_dedup} (corr_f1={best_f1})")
    p["dedup_iou"] = best_dedup

    # Phase 2: boundary_band sweep with corrected_f1 (best dedup_iou)
    print(f"\n{'='*80}")
    print(f"Phase 2: boundary_band sweep with corrected_f1 (dedup_iou={best_dedup})")
    print(f"{'='*80}")
    print(f"  {'bnd':>6s} {'TP':>5s} {'n_det':>6s} {'raw_fp':>7s} {'true_fp':>8s} "
          f"{'real_unl':>9s} {'recall':>7s} {'corr_f1':>8s} {'dTP':>5s} {'df1':>6s}")
    bnd_rows = []
    best_bnd = 12.0
    best_f1_b = best_f1
    for val in BND_GRID:
        p2 = dict(p)
        p2["boundary_band"] = val
        det_map = {}
        for iid in ids:
            det_map[iid] = detect(images[iid], p2) if iid in images else []
        m = evaluate_det_map(det_map, ids, D_EVAL)
        ndet = sum(len(v) for v in det_map.values())
        df1 = m["corrected_f1"] - BASELINE_F1
        dtp = m["tp"] - BASELINE_TP
        marker = " *" if val == 12.0 else ""
        print(f"  {val:6.1f} {m['tp']:5d} {ndet:6d} {m['raw_fp']:7d} {m['true_fp']:8d} "
              f"{m['real_unlabeled']:9d} {m['recall']:6.1f}% {m['corrected_f1']:7.1f}% "
              f"{dtp:+5d} {df1:+6.1f}{marker}")
        bnd_rows.append({"boundary_band": val, "tp": m["tp"], "n_det": ndet,
                         "raw_fp": m["raw_fp"], "true_fp": m["true_fp"],
                         "real_unlabeled": m["real_unlabeled"],
                         "recall": m["recall"],
                         "corrected_f1": m["corrected_f1"]})
        if m["corrected_f1"] > best_f1_b:
            best_f1_b = m["corrected_f1"]
            best_bnd = val

    print(f"\n  Best boundary_band = {best_bnd} (corr_f1={best_f1_b})")

    # Final
    p["boundary_band"] = best_bnd
    det_map = {iid: detect(images[iid], p) if iid in images else [] for iid in ids}
    m = evaluate_det_map(det_map, ids, D_EVAL)
    ndet = sum(len(v) for v in det_map.values())
    print(f"\n{'='*80}")
    print(f"FINAL: dedup_iou={p['dedup_iou']}, boundary_band={p['boundary_band']}")
    print(f"  TP={m['tp']} (delta {m['tp']-BASELINE_TP:+d})  "
          f"corr_f1={m['corrected_f1']} (delta {m['corrected_f1']-BASELINE_F1:+.1f})  "
          f"n_det={ndet}  true_fp={m['true_fp']}")

    out = os.path.join(ROOT, "benchmark", "cv_scope31c.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"final": {"dedup_iou": p["dedup_iou"],
                             "boundary_band": p["boundary_band"],
                             "tp": m["tp"], "corrected_f1": m["corrected_f1"],
                             "n_det": ndet, "true_fp": m["true_fp"],
                             "dtp": m["tp"] - BASELINE_TP,
                             "df1": m["corrected_f1"] - BASELINE_F1},
                    "dedup_sweep": dedup_rows,
                    "bnd_sweep": bnd_rows},
                   f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()