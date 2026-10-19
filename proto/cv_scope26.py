"""LOOP-26 (Path-1): GT label completion.

LOOP-20 found 285 real-looking detections not in GT (267 isolated + 18 extension).
Adding ALL to GT: recall 67.3%->76.3%, f1 65.2->86.2.

This LOOP:
  1. Extract the 285 unlabeled-real detections from pm_v5_1
  2. Apply tiered confirmation heuristics (strict/medium/loose)
  3. Create completed GT files for each tier
  4. Re-evaluate pm_v5_1 with completed GT at D=3,5,8,12
  5. Report per-tier metrics

Confirmation tiers (stricter than is_real_unlabeled which uses sup>=0.3, side>=12, aspect<=8):
  strict:  red_support >= 0.60, min_side >= 25, max_aspect <= 5.0, ortho_dev <= 10
  medium:  red_support >= 0.50, min_side >= 20, max_aspect <= 6.0, ortho_dev <= 12
  loose:   red_support >= 0.40, min_side >= 15, max_aspect <= 8.0, ortho_dev <= 15
  all:     everything is_real_unlabeled (LOOP-20 baseline)

Output:
  benchmark/cv_scope26.json
  tmp_gt_completed_<tier>/  (completed GT JSON files)
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json
from cv_hybrid import ROOT, DIST_VALUES
from eval_pointmatch import (load_boxes, match_boxes_point, is_real_unlabeled,
                             red_support, corner_match_distance, PERMS_4)
from quad_ortho import interior_angles

PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o1000_ba")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope26.json")
D = 5.0

TIERS = {
    "strict": {"sup_min": 0.60, "min_side": 25.0, "max_aspect": 5.0, "ortho_dev": 10.0},
    "medium": {"sup_min": 0.50, "min_side": 20.0, "max_aspect": 6.0, "ortho_dev": 12.0},
    "loose":  {"sup_min": 0.40, "min_side": 15.0, "max_aspect": 8.0, "ortho_dev": 15.0},
    "all":    {"sup_min": 0.30, "min_side": 12.0, "max_aspect": 8.0, "ortho_dev": 15.0},
}


def max_ortho_deviation(quad):
    """Max deviation of interior angles from 90 degrees."""
    angles = interior_angles(quad)
    return max(abs(a - 90.0) for a in angles if math.isfinite(a))


def confirm_detection(box, mask_d, tier_params):
    """Check if an unmatched detection passes the confirmation heuristic."""
    sides = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
    mn, mx = min(sides), max(sides)
    if mn < tier_params["min_side"]:
        return False
    if mx / max(mn, 1e-6) > tier_params["max_aspect"]:
        return False
    if red_support(box, mask_d) < tier_params["sup_min"]:
        return False
    if max_ortho_deviation(box) > tier_params["ortho_dev"]:
        return False
    return True


def get_unmatched_detections(ids, det_map, gt_map, mask_cache):
    """For each image, find unmatched detections that are real_unlabeled."""
    unmatched_real = {}
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        if not det:
            unmatched_real[iid] = []
            continue
        det_i = [d.astype(np.int32) for d in det]
        gt_i = [g.astype(np.int32) for g in gt]
        tp, unmatched = match_boxes_point(det_i, gt_i, D)
        exp = load_expected()
        csv_n = exp.get(iid, 0)
        real_dets = []
        for di in unmatched:
            if is_real_unlabeled(det[di].astype(np.int32), csv_n, len(gt), mask_cache[iid]):
                real_dets.append(det[di].astype(np.int32))
        unmatched_real[iid] = real_dets
    return unmatched_real


def evaluate_with_gt(ids, det_map, gt_map, dist_thr):
    """Evaluate pm_v5_1 against a GT map at a given D threshold."""
    TP = FN = FP = 0
    n_gt = n_det = 0
    for iid in ids:
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        gt = [g.astype(np.int32) for g in gt_map.get(iid, [])]
        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp
        FN += len(gt) - tp
        FP += len(unmatched)
        n_gt += len(gt)
        n_det += len(det)
    recall = TP / n_gt * 100 if n_gt else 0
    prec = TP / n_det * 100 if n_det else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    return {
        "tp": TP, "fn": FN, "fp": FP,
        "n_gt": n_gt, "n_det": n_det,
        "recall": round(recall, 2),
        "precision": round(prec, 2),
        "f1": round(f1, 2),
    }


def main():
    ids = get_ids()
    print(f"LOOP-26: GT label completion on {len(ids)} real images (D={D})", flush=True)
    print(f"  pm_v5_1 reference: {PM5_REF}", flush=True)

    # Load pm_v5_1 detections and GT
    det_map = {iid: load_boxes(PM5_REF, iid) for iid in ids}
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}

    # Build mask cache
    print(f"\nBuilding mask cache...", flush=True)
    mask_cache = {}
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask_cache[iid] = mask_d
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] masks cached", flush=True)
    print(f"  {len(mask_cache)} masks cached", flush=True)

    # Extract unmatched real-looking detections
    print(f"\nExtracting unmatched real-looking detections...", flush=True)
    unmatched_real = get_unmatched_detections(ids, det_map, gt_map, mask_cache)
    total_real = sum(len(v) for v in unmatched_real.values())
    print(f"  Total unmatched real-looking: {total_real}")

    # Baseline metrics (original GT)
    print(f"\nBaseline metrics (original GT, {sum(len(v) for v in gt_map.values())} GT boxes):", flush=True)
    base_metrics = {}
    for dd in DIST_VALUES:
        m = evaluate_with_gt(ids, det_map, gt_map, float(dd))
        base_metrics[str(dd)] = m
        print(f"  D={dd:2d}: TP={m['tp']:4d} recall={m['recall']:5.1f}% f1={m['f1']:5.1f}")

    # Per-tier completion
    tier_results = {}
    for tier_name, tier_params in TIERS.items():
        print(f"\n--- Tier: {tier_name} (sup>={tier_params['sup_min']}, "
              f"side>={tier_params['min_side']}, aspect<={tier_params['max_aspect']}, "
              f"ortho<={tier_params['ortho_dev']}) ---", flush=True)

        # Confirm detections
        confirmed = {}
        n_confirmed = 0
        for iid in ids:
            confirmed_iid = []
            for box in unmatched_real.get(iid, []):
                if confirm_detection(box, mask_cache[iid], tier_params):
                    confirmed_iid.append(box)
            confirmed[iid] = confirmed_iid
            n_confirmed += len(confirmed_iid)
        print(f"  Confirmed: {n_confirmed} / {total_real}")

        # Build completed GT
        completed_gt = {}
        for iid in ids:
            completed_gt[iid] = list(gt_map.get(iid, [])) + confirmed.get(iid, [])

        new_gt_size = sum(len(v) for v in completed_gt.values())
        print(f"  New GT size: {sum(len(v) for v in gt_map.values())} + {n_confirmed} = {new_gt_size}")

        # Write completed GT files
        gt_out_dir = os.path.join(ROOT, f"tmp_gt_completed_{tier_name}")
        os.makedirs(gt_out_dir, exist_ok=True)
        for iid in ids:
            quads = completed_gt[iid]
            with open(os.path.join(gt_out_dir, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, [np.asarray(q, np.int32) for q in quads]), f,
                          ensure_ascii=False)

        # Evaluate with completed GT
        tier_metrics = {}
        for dd in DIST_VALUES:
            m = evaluate_with_gt(ids, det_map, completed_gt, float(dd))
            tier_metrics[str(dd)] = m
            b = base_metrics[str(dd)]
            d_tp = m["tp"] - b["tp"]
            d_recall = m["recall"] - b["recall"]
            d_f1 = m["f1"] - b["f1"]
            print(f"  D={dd:2d}: TP={m['tp']:4d} ({d_tp:+d})  "
                  f"recall={m['recall']:5.1f}% ({d_recall:+.1f})  "
                  f"f1={m['f1']:5.1f} ({d_f1:+.1f})")

        tier_results[tier_name] = {
            "n_confirmed": n_confirmed,
            "new_gt_size": new_gt_size,
            "metrics": tier_metrics,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"GT COMPLETION SUMMARY (D={D}, {len(ids)} imgs)")
    print(f"{'='*80}")
    print(f"\nBaseline (original GT, {sum(len(v) for v in gt_map.values())} boxes):")
    b5 = base_metrics["5"]
    print(f"  D=5: TP={b5['tp']} recall={b5['recall']}% f1={b5['f1']}%")
    print(f"\n{'Tier':8s} {'Confirmed':>9s} {'NewGT':>6s}  "
          f"{'TP@5':>5s} {'dTP':>4s}  {'recall':>7s} {'drecall':>7s}  "
          f"{'f1':>6s} {'df1':>6s}")
    for tier_name in TIERS:
        r = tier_results[tier_name]
        m = r["metrics"]["5"]
        b = base_metrics["5"]
        print(f"{tier_name:8s} {r['n_confirmed']:>9d} {r['new_gt_size']:>6d}  "
              f"{m['tp']:>5d} {m['tp']-b['tp']:>+4d}  "
              f"{m['recall']:>6.1f}% {m['recall']-b['recall']:>+6.1f}  "
              f"{m['f1']:>5.1f}% {m['f1']-b['f1']:>+5.1f}")
    print(f"{'='*80}")

    # Save results
    results = {
        "n_images": len(ids),
        "baseline_gt_size": sum(len(v) for v in gt_map.values()),
        "total_unmatched_real": total_real,
        "base_metrics": base_metrics,
        "tier_results": tier_results,
        "tier_params": TIERS,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()