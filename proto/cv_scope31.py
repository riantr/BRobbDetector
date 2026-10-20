"""LOOP-31: Systematic parameter sweep of all 6 pipeline stages on real data.

Algorithm is FIXED (pm_v5_1 structure). Only parameters are swept.
Coordinate descent: optimize each stage sequentially, updating best params
after each stage. Within each stage, sweep one parameter at a time.

Stages and their parameters:
  1. red_mask:      smin, vmin
  2. denoise:        denoise_min_area
  3. outer_contours: min_area
  4. base_quads:     fit_band, fit_min_pts, fit_jump
  5. _pass:          min_side, max_side, max_aspect, ortho_tol, boundary_band
  6. dedup:          dedup_iou

Baseline: pm_v5_1 (TP=502, corr_f1=71.5 at D=5, 500 images).

Usage:
  python proto/cv_scope31.py
  python proto/cv_scope31.py --stage 4      # run only stage 4
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import (get_ids, denoise, _quad_adaptive, _side_ok,
                        DEFAULT as CQ_DEFAULT)
from cv_hybrid import (outer_contours, _pass, dedup_keep_order,
                       _edgefit_corners, FIT_DEFAULT, ROOT)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import load_boxes, match_boxes_point

D_EVAL = 5.0
BASELINE_TP = 502

# Parameter grids — comprehensive coverage of each state space
GRIDS = {
    "smin":             [40, 60, 80, 100, 120, 140, 160],
    "vmin":             [60, 80, 100, 110, 120, 140, 160],
    "denoise_min_area": [0, 20, 40, 60, 80, 100, 150, 200, 300, 500],
    "min_area":         [600, 800, 900, 1000, 1100, 1200, 1300, 1500, 2000],
    "fit_band":         [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "fit_min_pts":      [6, 8, 10, 12, 16, 20],
    "fit_jump":         [10.0, 15.0, 20.0, 25.0, 30.0],
    "min_side":         [10.0, 15.0, 20.0, 25.0, 30.0],
    "max_side":         [600.0, 800.0, 1000.0, 1200.0],
    "max_aspect":       [8.0, 10.0, 12.0, 15.0, 20.0],
    "ortho_tol":        [10.0, 12.0, 15.0, 18.0, 20.0, 25.0],
    "boundary_band":    [6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
    "dedup_iou":        [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
}

STAGES = [
    ("Stage 1: red_mask",      ["smin", "vmin"]),
    ("Stage 2: denoise",       ["denoise_min_area"]),
    ("Stage 3: outer_contours", ["min_area"]),
    ("Stage 4: base_quads",    ["fit_band", "fit_min_pts", "fit_jump"]),
    ("Stage 5: _pass",         ["min_side", "max_side", "max_aspect",
                                "ortho_tol", "boundary_band"]),
    ("Stage 6: dedup",         ["dedup_iou"]),
]


def base_params():
    """pm_v5_1 frozen params + smin/vmin (not in pm5_params, from red_mask default)."""
    p = pm5_params()
    p["min_area"] = 1000.0
    p["smin"] = 80
    p["vmin"] = 110
    return p


def detect(img, p):
    """Full pipeline with all params from p. Returns list of quads."""
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
    """Run detect on all images, return (TP, n_det, raw_fp) at D_EVAL."""
    TP = 0
    n_det = 0
    raw_fp = 0
    for iid in ids:
        det = detect(images[iid], p)
        det = [d.astype(np.int32) for d in det]
        n_det += len(det)
        tp, unmatched = match_boxes_point(det, gt_map[iid], D_EVAL)
        TP += tp
        raw_fp += len(unmatched)
    return TP, n_det, raw_fp


def classify(tp):
    if tp > BASELINE_TP:
        return "WIN"
    elif tp == BASELINE_TP:
        return "TIE"
    elif tp >= BASELINE_TP - 5:
        return "MARGINAL"
    else:
        return "DEAD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0,
                    help="run only this stage (1-6), 0=all")
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"LOOP-31: 6-stage parameter sweep on {len(ids)} imgs, "
          f"{total_gt} GT, D={D_EVAL:.0f}, baseline TP={BASELINE_TP}")

    # Cache images in memory
    print("Loading images...", end=" ", flush=True)
    images = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is not None:
            images[iid] = img
    print(f"{len(images)} loaded")

    p = base_params()
    print(f"\nInitial params: smin={p['smin']} vmin={p['vmin']} "
          f"denoise={p['denoise_min_area']} min_area={p['min_area']} "
          f"fit_band={p['fit_band']} fit_min_pts={p['fit_min_pts']} "
          f"fit_jump={p['fit_jump']} min_side={p['min_side']} "
          f"max_side={p['max_side']} max_aspect={p['max_aspect']} "
          f"ortho_tol={p['ortho_tol']} bnd={p['boundary_band']} "
          f"dedup={p['dedup_iou']}")

    # Verify baseline
    t0 = time.time()
    tp0, ndet0, fp0 = evaluate(ids, images, gt_map, p)
    print(f"Baseline check: TP={tp0} n_det={ndet0} raw_fp={fp0} "
          f"({time.time()-t0:.0f}s)")
    if tp0 != BASELINE_TP:
        print(f"  !! WARNING: baseline TP={tp0} != expected {BASELINE_TP}")

    all_results = {"baseline": {"tp": tp0, "n_det": ndet0, "raw_fp": fp0,
                                 "params": {k: p[k] for k in GRIDS}},
                   "stages": []}

    for si, (stage_name, param_names) in enumerate(STAGES):
        stage_num = si + 1
        if args.stage > 0 and args.stage != stage_num:
            continue

        print(f"\n{'='*72}")
        print(f"{stage_name} — sweeping {', '.join(param_names)}")
        print(f"{'='*72}")

        stage_result = {"name": stage_name, "params": param_names,
                        "sweeps": [], "best_tp": tp0, "improved": False}

        for pname in param_names:
            grid = GRIDS[pname]
            cur_val = p[pname]
            print(f"\n  Sweep {pname} (current={cur_val}):")
            print(f"  {'value':>10s} {'TP':>6s} {'n_det':>6s} {'fp':>6s} "
                  f"{'dTP':>5s} {'verdict':>10s}")

            best_val = cur_val
            best_tp = tp0
            sweep_rows = []

            for val in grid:
                p2 = dict(p)
                p2[pname] = val
                t0 = time.time()
                tp, ndet, fp = evaluate(ids, images, gt_map, p2)
                dt = time.time() - t0
                dtp = tp - BASELINE_TP
                verdict = classify(tp)
                marker = " *" if val == cur_val else ""
                print(f"  {str(val):>10s} {tp:6d} {ndet:6d} {fp:6d} "
                      f"{dtp:+5d} {verdict:>10s}{marker}  ({dt:.0f}s)")
                sweep_rows.append({"value": val, "tp": tp, "n_det": ndet,
                                   "raw_fp": fp, "dtp": dtp, "verdict": verdict})
                if tp > best_tp:
                    best_tp = tp
                    best_val = val

            # Update param if improved
            if best_tp > tp0:
                p[pname] = best_val
                print(f"  -> {pname} = {best_val} (TP {tp0}->{best_tp}, "
                      f"+{best_tp-tp0})")
                tp0 = best_tp
                stage_result["improved"] = True
                stage_result["best_tp"] = best_tp
            else:
                print(f"  -> {pname} unchanged (no improvement, best TP={best_tp})")

            stage_result["sweeps"].append({"param": pname, "rows": sweep_rows,
                                            "best_val": best_val,
                                            "best_tp": best_tp})

        all_results["stages"].append(stage_result)

    # Final summary
    print(f"\n{'='*72}")
    print(f"FINAL CONFIG (coordinate descent result)")
    print(f"{'='*72}")
    final_tp, final_ndet, final_fp = evaluate(ids, images, gt_map, p)
    print(f"  TP={final_tp} (baseline {BASELINE_TP}, "
          f"delta {final_tp-BASELINE_TP:+d})")
    print(f"  n_det={final_ndet}  raw_fp={final_fp}")
    print(f"  params:")
    for k in GRIDS:
        print(f"    {k} = {p[k]}")

    all_results["final"] = {"tp": final_tp, "n_det": final_ndet,
                            "raw_fp": final_fp,
                            "dtp": final_tp - BASELINE_TP,
                            "params": {k: p[k] for k in GRIDS}}

    out = os.path.join(ROOT, "benchmark", "cv_scope31.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()