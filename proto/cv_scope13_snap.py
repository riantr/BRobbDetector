"""LOOP-13 targeted method: corner centroid snap.

Hypothesis from FN analysis: 65/245 (26.5%) FN are "close" (<5px to nearest det),
117/245 (47.8%) are "medium" (5-20px). A better corner-refinement method that
snaps each corner to the LOCAL CENTROID of red pixels (not the nearest single
red pixel like LOOP-4) should improve precision.

This is fundamentally different from LOOP-4's snap which used `snap_to_pts`
(nearest contour point) — that under-corrected because it landed on wobbly
stroke-noise points. The centroid snap uses the CENTROID of all red pixels
in a local window, which is more robust to noise.

Variants:
  base (no snap, frozen pm_v5)
  centroid_5:  window r=5, max move 5px
  centroid_10: window r=10, max move 10px
  centroid_15: window r=15, max move 15px

Usage: python proto/cv_scope13_snap.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import (
    outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES,
)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

OUT_ROOT = os.path.join(ROOT, "tmp_sc13_snap")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope13_snap.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]


def centroid_snap(quad, mask, window_r=10, max_move=10):
    """Snap each corner to the centroid of red pixels in a local window.

    Returns snapped quad (int32) or original quad if no inlier found.
    """
    quad_f = np.asarray(quad, np.float32)
    H, W = mask.shape
    out = quad_f.copy()
    for i in range(4):
        cx, cy = quad_f[i]
        x_lo = max(0, int(cx - window_r))
        x_hi = min(W, int(cx + window_r + 1))
        y_lo = max(0, int(cy - window_r))
        y_hi = min(H, int(cy + window_r + 1))
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        win = mask[y_lo:y_hi, x_lo:x_hi]
        ys, xs = np.where(win > 0)
        if len(xs) < 3:
            continue
        cxs = xs + x_lo
        cys = ys + y_lo
        # Centroid
        nx = float(cxs.mean())
        ny = float(cys.mean())
        # Constrain movement
        dx = nx - cx
        dy = ny - cy
        d = np.sqrt(dx * dx + dy * dy)
        if d > max_move:
            scale = max_move / d
            nx = cx + dx * scale
            ny = cy + dy * scale
        out[i] = [nx, ny]
    return out.astype(np.int32)


def run_variant(name, ids, p, window_r, max_move):
    """Run pm_v5 + centroid snap on all images."""
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    det = {}
    total = 0
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det[iid] = []
            continue
        H, W = img.shape[:2]
        mask = red_mask(img)
        if p['denoise_min_area'] > 0:
            mask = denoise(mask, p['denoise_min_area'])
        cnts = outer_contours(mask, p)
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base.get(m), W, H, p)
                if q is not None:
                    hy_quads.append(q)
        try:
            hy = dedup_keep_order(hy_quads, p['dedup_iou'])
        except cv2.error:
            hy = dedup_keep_order(
                [q for q in hy_quads if q is not None and len(q) == 4],
                p['dedup_iou'],
            )
        # Apply centroid snap
        if window_r > 0:
            hy = [centroid_snap(q, mask, window_r, max_move) for q in hy]
            # Re-filter (snap might violate orthogonality)
            hy = [q for q in hy if _pass(q.astype(np.float32), W, H, p) is not None]
            try:
                hy = dedup_keep_order(hy, p['dedup_iou'])
            except cv2.error:
                hy = []
        det[iid] = hy
        total += len(hy)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in hy]), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  [{name} {n+1}/{len(ids)}] boxes={total} elapsed={dt:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"  {name}: {total} boxes in {dt:.0f}s", flush=True)
    return det


def evaluate_det_map(det_map, ids, dist_thr=5.0):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)
        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp
        FN += len(gt) - tp
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    return {
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
        "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "corrected_f1": round(f1, 1),
    }


def evaluate_pm5_ref(ids, dist_thr=5.0):
    det_map = {iid: load_boxes(PM5_REF, iid) for iid in ids}
    return evaluate_det_map(det_map, ids, dist_thr)


def make_verdict(r, pm5):
    tp_b, f1_b = pm5["tp"], pm5["corrected_f1"]
    tp_n, f1_n = r["tp"], r["corrected_f1"]
    if tp_n >= tp_b and f1_n >= f1_b:
        return "WIN"
    if tp_n >= tp_b - 5 and f1_n >= f1_b - 1.0:
        return "MARGINAL"
    return "DEAD"


def main():
    ids = get_ids()
    p = pm5_params()
    print(f"LOOP-13 targeted method: corner centroid snap (D={D})", flush=True)
    print(f"  pm_v5 baseline: min_area={p['min_area']:.0f}  ortho={p['ortho_tol']}", flush=True)

    print(f"\nEvaluating pm_v5 baseline at D={D}...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    variants = [
        # name, window_r, max_move
        ("snap_r5",  5,  5),
        ("snap_r10", 10, 10),
        ("snap_r15", 15, 15),
    ]

    results = {"pm5": pm5, "methods": {}, "dist_thr": D, "n_images": len(ids)}
    for name, wr, mm in variants:
        print(f"\nRunning {name} (window_r={wr}, max_move={mm})...", flush=True)
        t0 = time.time()
        det_map = run_variant(name, ids, p, wr, mm)
        dt = time.time() - t0
        r = evaluate_det_map(det_map, ids, D)
        r["runtime_s"] = round(dt, 1)
        r["window_r"] = wr
        r["max_move"] = mm
        r["verdict"] = make_verdict(r, pm5)
        results["methods"][name] = r
        print(f"  {name}: TP={r['tp']} corr_f1={r['corrected_f1']} "
              f"recall={r['recall']} ({dt:.0f}s) [{r['verdict']}]", flush=True)

    winners = [n for n, _, _ in variants if results["methods"][n]["verdict"] == "WIN"]
    if winners:
        print(f"\nRe-evaluating WINners at all D values: {winners}", flush=True)
        for name in winners:
            det_map = {iid: load_boxes(os.path.join(OUT_ROOT, name), iid) for iid in ids}
            results["methods"][name]["full_D"] = {}
            for dd in DIST_VALUES_FULL:
                results["methods"][name]["full_D"][f"D{dd}"] = evaluate_det_map(det_map, ids, float(dd))

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, 500 images, pm_v5 baseline TP={pm5['tp']} corr_f1={pm5['corrected_f1']})", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':14s}  {'win_r':>5s}  {'max_mv':>6s}  {'TP':>5s}  {'rawFP':>6s}  "
          f"{'trueFP':>7s}  {'recall':>6s}  {'corr_f1':>8s}  Verdict", flush=True)
    print(f"{'pm_v5':14s}  {'-':>5s}  {'-':>6s}  {pm5['tp']:>5d}  {pm5['raw_fp']:>6d}  "
          f"{pm5['true_fp']:>7d}  {pm5['recall']:>6.1f}  {pm5['corrected_f1']:>8.1f}  baseline", flush=True)
    for name, wr, mm in variants:
        r = results["methods"][name]
        print(f"{name:14s}  {wr:>5d}  {mm:>6d}  {r['tp']:>5d}  {r['raw_fp']:>6d}  "
              f"{r['true_fp']:>7d}  {r['recall']:>6.1f}  {r['corrected_f1']:>8.1f}  {r['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()