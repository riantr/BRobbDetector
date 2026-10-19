"""LOOP-15b: Evaluate the LOOP-15 best-axis combination on the 500-image real benchmark.

Best per-axis from LOOP-15:
  smin=60, vmin=130, denoise_min=20, min_area=1200, min_side=10, max_aspect=6,
  ortho_tol=10, boundary_band=5

Run pm_v5 with these params on the 500-image real benchmark and compare to
pm_v5 baseline (D=5 TP=502, corr_f1=71.3).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope15 import red_mask_with_hsv
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

OUT_ROOT = os.path.join(ROOT, "tmp_sc15_real")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope15_real.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0


def run_variant(name, ids, p, smin, vmin, denoise_min):
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
        mask = red_mask_with_hsv(img, smin, vmin)
        if denoise_min > 0:
            mask = denoise(mask, denoise_min)
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
    print(f"LOOP-15b: validate LOOP-15 best-axis on 500-real (D={D})", flush=True)

    # Best per-axis from LOOP-15
    p_opt = dict(p)
    p_opt['min_area'] = 1200
    p_opt['min_side'] = 10
    p_opt['max_aspect'] = 6
    p_opt['ortho_tol'] = 10
    p_opt['boundary_band'] = 5
    smin_opt, vmin_opt, denoise_opt = 60, 130, 20

    print(f"  Optimized params: min_area={p_opt['min_area']} min_side={p_opt['min_side']} "
          f"max_aspect={p_opt['max_aspect']} ortho_tol={p_opt['ortho_tol']} "
          f"boundary_band={p_opt['boundary_band']}", flush=True)
    print(f"  Optimized HSV: smin={smin_opt} vmin={vmin_opt} denoise={denoise_opt}", flush=True)

    print(f"\nEvaluating pm_v5 baseline...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    print(f"\nRunning optimized (best-axis) variant on 500 imgs...", flush=True)
    t0 = time.time()
    det_map = run_variant("optimized", ids, p_opt, smin_opt, vmin_opt, denoise_opt)
    dt = time.time() - t0
    r = evaluate_det_map(det_map, ids, D)
    r["runtime_s"] = round(dt, 1)
    r["verdict"] = make_verdict(r, pm5)
    print(f"  optimized: TP={r['tp']} corr_f1={r['corrected_f1']} "
          f"recall={r['recall']} ({dt:.0f}s) [{r['verdict']}]", flush=True)

    # Also test only-HSV (vmin=130 alone) on real
    print(f"\nRunning vmin-only (vmin=130) on 500 imgs...", flush=True)
    t0 = time.time()
    det_map_v = run_variant("vmin130", ids, p, 80, 130, 40)
    dt = time.time() - t0
    r_v = evaluate_det_map(det_map_v, ids, D)
    r_v["runtime_s"] = round(dt, 1)
    r_v["verdict"] = make_verdict(r_v, pm5)
    print(f"  vmin130: TP={r_v['tp']} corr_f1={r_v['corrected_f1']} "
          f"recall={r_v['recall']} ({dt:.0f}s) [{r_v['verdict']}]", flush=True)

    # And min_area=1200 only
    print(f"\nRunning min_area-only (min_area=1200) on 500 imgs...", flush=True)
    t0 = time.time()
    p_ma = dict(p)
    p_ma['min_area'] = 1200
    det_map_ma = run_variant("minarea1200", ids, p_ma, 80, 110, 40)
    dt = time.time() - t0
    r_ma = evaluate_det_map(det_map_ma, ids, D)
    r_ma["runtime_s"] = round(dt, 1)
    r_ma["verdict"] = make_verdict(r_ma, pm5)
    print(f"  minarea1200: TP={r_ma['tp']} corr_f1={r_ma['corrected_f1']} "
          f"recall={r_ma['recall']} ({dt:.0f}s) [{r_ma['verdict']}]", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, 500 imgs)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':16s}  {'TP':>5s}  {'rawFP':>6s}  {'trueFP':>7s}  "
          f"{'recall':>6s}  {'corr_f1':>8s}  Verdict", flush=True)
    print(f"{'pm_v5':16s}  {pm5['tp']:>5d}  {pm5['raw_fp']:>6d}  {pm5['true_fp']:>7d}  "
          f"{pm5['recall']:>6.1f}  {pm5['corrected_f1']:>8.1f}  baseline", flush=True)
    print(f"{'optimized':16s}  {r['tp']:>5d}  {r['raw_fp']:>6d}  {r['true_fp']:>7d}  "
          f"{r['recall']:>6.1f}  {r['corrected_f1']:>8.1f}  {r['verdict']}", flush=True)
    print(f"{'vmin130':16s}  {r_v['tp']:>5d}  {r_v['raw_fp']:>6d}  {r_v['true_fp']:>7d}  "
          f"{r_v['recall']:>6.1f}  {r_v['corrected_f1']:>8.1f}  {r_v['verdict']}", flush=True)
    print(f"{'minarea1200':16s}  {r_ma['tp']:>5d}  {r_ma['raw_fp']:>6d}  {r_ma['true_fp']:>7d}  "
          f"{r_ma['recall']:>6.1f}  {r_ma['corrected_f1']:>8.1f}  {r_ma['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "pm5": pm5,
            "optimized": r,
            "vmin130": r_v,
            "minarea1200": r_ma,
            "dist_thr": D,
            "n_images": len(ids),
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()