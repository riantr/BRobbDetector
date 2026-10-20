"""LOOP-31b: Extend dedup_iou sweep + second-pass coordinate descent.

LOOP-31 found +6 TP (boundary_band=20, dedup_iou=0.50). The dedup_iou
trend was monotonic increasing up to 0.50. This script:
  1. Extends dedup_iou to [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0]
  2. Re-sweeps all params with the new baseline (boundary_band=20, best dedup)
  3. Reports final config

Usage: python proto/cv_scope31b.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask
from canny_quad import get_ids, denoise
from cv_hybrid import outer_contours, _pass, dedup_keep_order, ROOT
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import load_boxes, match_boxes_point

D_EVAL = 5.0
BASELINE_TP = 502

DEDUP_EXT = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0]

SECOND_PASS = [
    ("smin",             [60, 80, 100, 120]),
    ("vmin",             [100, 110, 120, 130]),
    ("min_area",         [800, 900, 1000, 1100, 1200]),
    ("fit_band",         [5.0, 6.0, 7.0, 8.0]),
    ("ortho_tol",        [13.0, 15.0, 17.0]),
    ("boundary_band",    [15.0, 18.0, 20.0, 25.0, 30.0]),
    ("min_side",         [12.0, 15.0, 18.0]),
    ("max_aspect",       [10.0, 12.0, 15.0]),
]


def base_params():
    p = pm5_params()
    p["min_area"] = 1000.0
    p["smin"] = 80
    p["vmin"] = 110
    p["boundary_band"] = 20.0  # LOOP-31 WIN
    p["dedup_iou"] = 0.50      # LOOP-31 WIN (to be extended)
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


def evaluate(ids, images, gt_map, p):
    TP = n_det = raw_fp = 0
    for iid in ids:
        det = [d.astype(np.int32) for d in detect(images[iid], p)]
        n_det += len(det)
        tp, unmatched = match_boxes_point(det, gt_map[iid], D_EVAL)
        TP += tp
        raw_fp += len(unmatched)
    return TP, n_det, raw_fp


def classify(tp):
    if tp > BASELINE_TP: return "WIN"
    elif tp == BASELINE_TP: return "TIE"
    elif tp >= BASELINE_TP - 5: return "MARGINAL"
    else: return "DEAD"


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    print(f"LOOP-31b: dedup extension + second pass on {len(ids)} imgs, "
          f"D={D_EVAL:.0f}, baseline TP={BASELINE_TP}")

    print("Loading images...", end=" ", flush=True)
    images = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is not None:
            images[iid] = img
    print(f"{len(images)} loaded")

    p = base_params()
    tp0, ndet0, fp0 = evaluate(ids, images, gt_map, p)
    print(f"Current config (bnd=20, dedup=0.50): TP={tp0} n_det={ndet0} fp={fp0}")

    # Phase 1: extend dedup_iou
    print(f"\n{'='*72}")
    print(f"Phase 1: Extend dedup_iou sweep")
    print(f"{'='*72}")
    print(f"  {'value':>8s} {'TP':>6s} {'n_det':>6s} {'fp':>6s} {'dTP':>5s} {'verdict':>10s}")
    best_dedup = 0.50
    best_tp = tp0
    dedup_rows = []
    for val in DEDUP_EXT:
        p2 = dict(p)
        p2["dedup_iou"] = val
        t0 = time.time()
        tp, ndet, fp = evaluate(ids, images, gt_map, p2)
        dt = time.time() - t0
        dtp = tp - BASELINE_TP
        verdict = classify(tp)
        marker = " *" if val == 0.50 else ""
        print(f"  {val:8.2f} {tp:6d} {ndet:6d} {fp:6d} {dtp:+5d} {verdict:>10s}{marker}  ({dt:.0f}s)")
        dedup_rows.append({"value": val, "tp": tp, "n_det": ndet, "fp": fp})
        if tp > best_tp:
            best_tp = tp
            best_dedup = val

    p["dedup_iou"] = best_dedup
    tp0 = best_tp
    print(f"  -> dedup_iou = {best_dedup} (TP={best_tp})")

    # Phase 2: second pass
    print(f"\n{'='*72}")
    print(f"Phase 2: Second-pass coordinate descent (with bnd=20, dedup={best_dedup})")
    print(f"{'='*72}")

    for pname, grid in SECOND_PASS:
        cur_val = p[pname]
        print(f"\n  Sweep {pname} (current={cur_val}):")
        print(f"  {'value':>10s} {'TP':>6s} {'n_det':>6s} {'fp':>6s} {'dTP':>5s} {'verdict':>10s}")
        best_val = cur_val
        best_tp2 = tp0
        for val in grid:
            p2 = dict(p)
            p2[pname] = val
            t0 = time.time()
            tp, ndet, fp = evaluate(ids, images, gt_map, p2)
            dt = time.time() - t0
            dtp = tp - BASELINE_TP
            verdict = classify(tp)
            marker = " *" if val == cur_val else ""
            print(f"  {str(val):>10s} {tp:6d} {ndet:6d} {fp:6d} {dtp:+5d} {verdict:>10s}{marker}  ({dt:.0f}s)")
            if tp > best_tp2:
                best_tp2 = tp
                best_val = val
        if best_tp2 > tp0:
            p[pname] = best_val
            print(f"  -> {pname} = {best_val} (TP {tp0}->{best_tp2}, +{best_tp2-tp0})")
            tp0 = best_tp2
        else:
            print(f"  -> {pname} unchanged (best TP={best_tp2})")

    # Final
    print(f"\n{'='*72}")
    print(f"FINAL CONFIG")
    print(f"{'='*72}")
    final_tp, final_ndet, final_fp = evaluate(ids, images, gt_map, p)
    print(f"  TP={final_tp} (baseline {BASELINE_TP}, delta {final_tp-BASELINE_TP:+d})")
    print(f"  n_det={final_ndet}  raw_fp={final_fp}")
    for k in ("smin", "vmin", "denoise_min_area", "min_area", "fit_band",
              "fit_min_pts", "fit_jump", "min_side", "max_side", "max_aspect",
              "ortho_tol", "boundary_band", "dedup_iou"):
        print(f"    {k} = {p[k]}")

    out = os.path.join(ROOT, "benchmark", "cv_scope31b.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"final": {"tp": final_tp, "n_det": final_ndet,
                             "raw_fp": final_fp,
                             "dtp": final_tp - BASELINE_TP,
                             "params": {k: p[k] for k in (
                                 "smin", "vmin", "denoise_min_area", "min_area",
                                 "fit_band", "fit_min_pts", "fit_jump", "min_side",
                                 "max_side", "max_aspect", "ortho_tol",
                                 "boundary_band", "dedup_iou")}},
                    "dedup_extension": dedup_rows}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()