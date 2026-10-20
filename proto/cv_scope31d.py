"""LOOP-31d: Second-pass full sweep with corrected_f1 metric.

LOOP-31c found dedup_iou=0.70 (corr_f1=75.0, +3.5) and boundary_band=20
(corr_f1=75.2, +3.7) as true optima. Now re-sweep ALL params with
corrected_f1 to catch interactions.

Usage: python proto/cv_scope31d.py
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
from eval_pointmatch import load_boxes

D_EVAL = 5.0
BASELINE_TP = 502
BASELINE_F1 = 71.5

GRIDS = {
    "smin":             [60, 80, 100, 120],
    "vmin":             [100, 110, 120],
    "denoise_min_area": [8, 40, 80, 200],
    "min_area":         [800, 900, 1000, 1100, 1500],
    "fit_band":         [4.0, 5.0, 6.0, 8.0],
    "fit_min_pts":      [8, 12, 16],
    "fit_jump":         [15.0, 20.0, 25.0],
    "min_side":         [12.0, 15.0, 20.0],
    "max_side":         [600.0, 800.0, 1200.0],
    "max_aspect":       [10.0, 12.0, 15.0],
    "ortho_tol":        [12.0, 15.0, 17.0, 20.0],
    "boundary_band":    [15.0, 20.0, 25.0],
    "dedup_iou":        [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
}

ORDER = ["smin", "vmin", "denoise_min_area", "min_area",
         "fit_band", "fit_min_pts", "fit_jump",
         "min_side", "max_side", "max_aspect",
         "ortho_tol", "boundary_band", "dedup_iou"]


def base_params():
    p = pm5_params()
    p["min_area"] = 1000.0
    p["smin"] = 80
    p["vmin"] = 110
    p["boundary_band"] = 20.0
    p["dedup_iou"] = 0.70
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


def evaluate(ids, images, p):
    det_map = {iid: detect(images[iid], p) if iid in images else [] for iid in ids}
    m = evaluate_det_map(det_map, ids, D_EVAL)
    ndet = sum(len(v) for v in det_map.values())
    return m, ndet


def main():
    ids = get_ids()
    print(f"LOOP-31d: second-pass full sweep with corrected_f1 on {len(ids)} imgs")
    print(f"Baseline: TP={BASELINE_TP}, corr_f1={BASELINE_F1}")
    print(f"Starting config: dedup_iou=0.70, boundary_band=20.0")

    print("Loading images...", end=" ", flush=True)
    images = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is not None:
            images[iid] = img
    print(f"{len(images)} loaded")

    p = base_params()
    m0, ndet0 = evaluate(ids, images, p)
    tp0 = m0["tp"]
    f1_0 = m0["corrected_f1"]
    print(f"Current: TP={tp0} corr_f1={f1_0} n_det={ndet0} true_fp={m0['true_fp']}\n")

    results = {"start": {"tp": tp0, "corr_f1": f1_0,
                         "params": {k: p[k] for k in ORDER}},
               "sweeps": []}

    for pname in ORDER:
        grid = GRIDS[pname]
        cur_val = p[pname]
        print(f"{'='*72}")
        print(f"Sweep {pname} (current={cur_val}):")
        print(f"  {'value':>10s} {'TP':>5s} {'n_det':>6s} {'true_fp':>8s} "
              f"{'recall':>7s} {'corr_f1':>8s} {'dTP':>5s} {'df1':>6s}")
        best_val = cur_val
        best_f1 = f1_0
        rows = []
        for val in grid:
            p2 = dict(p)
            p2[pname] = val
            t0 = time.time()
            m, ndet = evaluate(ids, images, p2)
            dt = time.time() - t0
            dtp = m["tp"] - BASELINE_TP
            df1 = m["corrected_f1"] - BASELINE_F1
            marker = " *" if val == cur_val else ""
            print(f"  {str(val):>10s} {m['tp']:5d} {ndet:6d} {m['true_fp']:8d} "
                  f"{m['recall']:6.1f}% {m['corrected_f1']:7.1f}% "
                  f"{dtp:+5d} {df1:+6.1f}{marker}  ({dt:.0f}s)")
            rows.append({"value": val, "tp": m["tp"],
                         "corr_f1": m["corrected_f1"],
                         "n_det": ndet, "true_fp": m["true_fp"]})
            if m["corrected_f1"] > best_f1:
                best_f1 = m["corrected_f1"]
                best_val = val
        if best_f1 > f1_0:
            p[pname] = best_val
            print(f"  -> {pname} = {best_val} "
                  f"(corr_f1 {f1_0}->{best_f1}, +{best_f1-f1_0:.1f})")
            f1_0 = best_f1
        else:
            print(f"  -> {pname} unchanged (best corr_f1={best_f1})")
        results["sweeps"].append({"param": pname, "rows": rows,
                                  "best_val": best_val, "best_f1": best_f1})

    m_f, ndet_f = evaluate(ids, images, p)
    print(f"\n{'='*72}")
    print(f"FINAL CONFIG")
    print(f"  TP={m_f['tp']} (delta {m_f['tp']-BASELINE_TP:+d})")
    print(f"  corr_f1={m_f['corrected_f1']} "
          f"(delta {m_f['corrected_f1']-BASELINE_F1:+.1f})")
    print(f"  n_det={ndet_f}  true_fp={m_f['true_fp']}  "
          f"real_unl={m_f['real_unlabeled']}")
    for k in ORDER:
        print(f"    {k} = {p[k]}")

    results["final"] = {"tp": m_f["tp"], "corr_f1": m_f["corrected_f1"],
                        "n_det": ndet_f, "true_fp": m_f["true_fp"],
                        "dtp": m_f["tp"] - BASELINE_TP,
                        "df1": m_f["corrected_f1"] - BASELINE_F1,
                        "params": {k: p[k] for k in ORDER}}
    out = os.path.join(ROOT, "benchmark", "cv_scope31d.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()