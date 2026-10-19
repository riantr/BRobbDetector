"""Validate corner-gap synthetic model against the 500-image real benchmark.

The corner-gap fix in `cv_scope13.draw_obb` only affects SYNTHETIC data generation
(it's how the brokenness is drawn). Real data doesn't go through draw_obb.

So this validation is really checking two things:
  1. Confirm that the pm_v5 real-data path is byte-identical to the frozen
     baseline (proves the corner-gap fix didn't accidentally change real-data code).
  2. Re-confirm the LOOP-15c winner (min_area=1000) on real data — does it
     still help with the more realistic synthetic model available for tuning?

The corner-gap synthetic data (f1 46.8% on synth) is a better predictor of real
performance than the per_edge synthetic (f1 20.8%). Future tuning can use the
new synthetic baseline; real-data validation remains the ground truth.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

OUT_ROOT = os.path.join(ROOT, "tmp_corner_real")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope18_real.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]


def run_variant(name, ids, p, min_area=900):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    p2 = dict(p)
    p2['min_area'] = min_area
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
    print(f"LOOP-18 real validation: corner-gap fix + min_area=1000 on 500 imgs (D={D})", flush=True)
    print(f"  pm_v5 baseline: min_area={p['min_area']:.0f}", flush=True)

    print(f"\nEvaluating pm_v5 baseline (frozen tmp_sc6_hy_o15_ba) at D={D}...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    # Re-run pm_v5 with current code path to verify byte-equality
    print(f"\nRe-running pm_v5 (min_area=900) with current code...", flush=True)
    t0 = time.time()
    det_map = run_variant("pm_v5_900", ids, p, min_area=900)
    dt = time.time() - t0
    r900 = evaluate_det_map(det_map, ids, D)
    r900["runtime_s"] = round(dt, 1)
    r900["min_area"] = 900
    r900["verdict"] = make_verdict(r900, pm5)
    print(f"  pm_v5_900 (re-run): TP={r900['tp']} corr_f1={r900['corrected_f1']} "
          f"recall={r900['recall']} ({dt:.0f}s) [{r900['verdict']}]", flush=True)

    # Test pm_v5.1 (min_area=1000, LOOP-15c WIN)
    print(f"\nRunning pm_v5_1 (min_area=1000, LOOP-15c WIN)...", flush=True)
    t0 = time.time()
    det_map_1 = run_variant("pm_v5_1000", ids, p, min_area=1000)
    dt = time.time() - t0
    r1000 = evaluate_det_map(det_map_1, ids, D)
    r1000["runtime_s"] = round(dt, 1)
    r1000["min_area"] = 1000
    r1000["verdict"] = make_verdict(r1000, pm5)
    print(f"  pm_v5_1000: TP={r1000['tp']} corr_f1={r1000['corrected_f1']} "
          f"recall={r1000['recall']} ({dt:.0f}s) [{r1000['verdict']}]", flush=True)

    # Multi-D for the WIN
    if r1000["verdict"] == "WIN":
        print(f"\nRe-evaluating pm_v5_1000 at all D values...", flush=True)
        det_map_w = {iid: load_boxes(os.path.join(OUT_ROOT, "pm_v5_1000"), iid) for iid in ids}
        r1000["full_D"] = {}
        for dd in DIST_VALUES_FULL:
            r1000["full_D"][f"D{dd}"] = evaluate_det_map(det_map_w, ids, float(dd))

    # Byte-equality check
    print(f"\nByte-equality check (pm_v5 re-run vs frozen baseline):", flush=True)
    pm5_rerun = {iid: load_boxes(os.path.join(OUT_ROOT, "pm_v5_900"), iid) for iid in ids}
    pm5_frozen = {iid: load_boxes(PM5_REF, iid) for iid in ids}
    diff_count = 0
    for iid in ids:
        d_rerun = sorted([np.asarray(q).tobytes() for q in pm5_rerun.get(iid, [])])
        d_frozen = sorted([np.asarray(q).tobytes() for q in pm5_frozen.get(iid, [])])
        if d_rerun != d_frozen:
            diff_count += 1
    print(f"  {diff_count} / {len(ids)} images differ between re-run and frozen baseline", flush=True)
    if diff_count == 0:
        print(f"  ✓ BYTE-IDENTICAL — corner-gap fix did NOT change real-data path", flush=True)
    else:
        print(f"  ✗ {diff_count} differ — investigate why", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, 500 imgs, pm_v5 baseline TP=502 corr_f1=71.3)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':16s}  {'min_area':>9s}  {'TP':>5s}  {'rawFP':>6s}  "
          f"{'trueFP':>7s}  {'recall':>6s}  {'corr_f1':>8s}  Verdict", flush=True)
    print(f"{'pm_v5 (frozen)':16s}  {p['min_area']:>9.0f}  {pm5['tp']:>5d}  {pm5['raw_fp']:>6d}  "
          f"{pm5['true_fp']:>7d}  {pm5['recall']:>6.1f}  {pm5['corrected_f1']:>8.1f}  baseline", flush=True)
    print(f"{'pm_v5_900 rerun':16s}  {r900['min_area']:>9d}  {r900['tp']:>5d}  {r900['raw_fp']:>6d}  "
          f"{r900['true_fp']:>7d}  {r900['recall']:>6.1f}  {r900['corrected_f1']:>8.1f}  {r900['verdict']}", flush=True)
    print(f"{'pm_v5_1000':16s}  {r1000['min_area']:>9d}  {r1000['tp']:>5d}  {r1000['raw_fp']:>6d}  "
          f"{r1000['true_fp']:>7d}  {r1000['recall']:>6.1f}  {r1000['corrected_f1']:>8.1f}  {r1000['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "frozen_baseline": pm5,
            "pm_v5_900_rerun": r900,
            "pm_v5_1000": r1000,
            "byte_identical_to_frozen": diff_count == 0,
            "diff_count": diff_count,
            "dist_thr": D,
            "n_images": len(ids),
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()