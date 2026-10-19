"""LOOP-13 real-data validation: test targeted min_area lowering on the 500-image
real benchmark vs pm_v5 baseline (D=5 TP=502, corr_f1=71.3).

Hypothesis from synthetic data: lowering min_area from 900 to 200 will recover
broken-outline boxes (+12 TP on synthetic), but at a cost of more FPs.
On real data the trade-off may be worse (more red noise).

Variants:
  - min_area=900 (pm_v5 baseline, already frozen)
  - min_area=500
  - min_area=200
  - min_area=100

Usage: python proto/cv_scope13_real.py
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

OUT_ROOT = os.path.join(ROOT, "tmp_sc13_real")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope13_real.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0


def run_variant(name, ids, p, min_area):
    """Run pm_v5 with custom min_area."""
    p2 = dict(p)
    p2['min_area'] = min_area
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
        if p2['denoise_min_area'] > 0:
            mask = denoise(mask, p2['denoise_min_area'])
        cnts = outer_contours(mask, p2)
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p2)
            for m in HY_ORDER:
                q = _pass(base.get(m), W, H, p2)
                if q is not None:
                    hy_quads.append(q)
        try:
            hy = dedup_keep_order(hy_quads, p2['dedup_iou'])
        except cv2.error:
            hy = dedup_keep_order(
                [q for q in hy_quads if q is not None and len(q) == 4],
                p2['dedup_iou'],
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
    print(f"LOOP-13 real-data validation: min_area sweep vs pm_v5 (D={D})", flush=True)
    print(f"  pm_v5 baseline: min_area={p['min_area']:.0f}", flush=True)

    print(f"\nEvaluating pm_v5 baseline at D={D}...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    variants = [
        ("minarea_500", 500),
        ("minarea_200", 200),
        ("minarea_100", 100),
    ]

    results = {"pm5": pm5, "methods": {}, "dist_thr": D, "n_images": len(ids)}
    for name, ma in variants:
        print(f"\nRunning {name} (min_area={ma})...", flush=True)
        t0 = time.time()
        det_map = run_variant(name, ids, p, ma)
        dt = time.time() - t0
        r = evaluate_det_map(det_map, ids, D)
        r["runtime_s"] = round(dt, 1)
        r["min_area"] = ma
        r["verdict"] = make_verdict(r, pm5)
        results["methods"][name] = r
        print(f"  {name}: TP={r['tp']} corr_f1={r['corrected_f1']} "
              f"recall={r['recall']} ({dt:.0f}s) [{r['verdict']}]", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, 500 images, pm_v5 baseline TP={pm5['tp']} corr_f1={pm5['corrected_f1']})", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':16s}  {'min_area':>9s}  {'TP':>5s}  {'rawFP':>6s}  "
          f"{'trueFP':>7s}  {'recall':>6s}  {'corr_f1':>8s}  Verdict", flush=True)
    print(f"{'pm_v5':16s}  {p['min_area']:>9.0f}  {pm5['tp']:>5d}  {pm5['raw_fp']:>6d}  "
          f"{pm5['true_fp']:>7d}  {pm5['recall']:>6.1f}  {pm5['corrected_f1']:>8.1f}  baseline", flush=True)
    for name, ma in variants:
        r = results["methods"][name]
        print(f"{name:16s}  {ma:>9d}  {r['tp']:>5d}  {r['raw_fp']:>6d}  "
              f"{r['true_fp']:>7d}  {r['recall']:>6.1f}  {r['corrected_f1']:>8.1f}  {r['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()