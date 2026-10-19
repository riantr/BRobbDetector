"""LOOP-20: GT label completion analysis.

Question: are the 245 FN (GT boxes pm_v5_1 doesn't match) actually unannotated-real
boxes that pm_v5 detects with imprecise corners, or are they truly missed/noise?

Method:
  1. Load pm_v5_1 detection results (502 TP, 157 trueFP, 136 real_unlabeled)
  2. For each of the 245 FN:
     a. Find the nearest unmatched detection (any detection, not just matched ones)
     b. If nearest detection is <20px away AND is "real-looking" (has red pixels in its
        region), the FN is "imprecise match" — pm_v5 sees the box but at slightly
        different corners
     c. If nearest detection is >20px away, the FN is "truly missed" — no detection
  3. For each of the 136 real_unlabeled:
     a. Check if it's near an existing GT box (would be an extension of GT)
     b. Check if it's isolated (would be a new unannotated box)
  4. Re-compute metrics assuming GT completion (add 136 to GT)

Output:
  benchmark/cv_scope20.json
  benchmark/cv_scope20.md
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids
from cv_hybrid import ROOT, DIST_VALUES
from cv_scope13_fn_analysis import (min_corner_dist, quad_corners_to_box)
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled, red_support

PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o1000_ba")  # pm_v5_1 frozen
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope20.json")
D = 5.0


def is_detection_real(det, mask_d):
    """Check if a detection is a 'real' OBB shape (shape + red support, no csv check)."""
    sides = [np.linalg.norm(det[(i + 1) % 4] - det[i]) for i in range(4)]
    mn, mx = min(sides), max(sides)
    if mn < 15.0:  # MIN_SIDE
        return False
    if (mx / max(mn, 1e-6)) > 12.0:  # MAX_ASPECT
        return False
    return red_support(det, mask_d) >= 0.3  # SUP_THR ≈ 0.3


def match_both_unmatched(det, gt, dist_thr):
    """Run match_boxes_point then also compute unmatched GT indices."""
    tp, unmatched_det = match_boxes_point(det, gt, dist_thr)
    # Reconstruct matched pairs to know which GTs were matched
    candidates = []
    for di in range(len(det)):
        for gj in range(len(gt)):
            from eval_pointmatch import corner_match_distance
            d = corner_match_distance(det[di], gt[gj])
            if d <= dist_thr:
                candidates.append((d, di, gj))
    candidates.sort()
    used_det = set()
    used_gt = set()
    for d, di, gj in candidates:
        if di in used_det or gj in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gj)
    unmatched_gt = [gj for gj in range(len(gt)) if gj not in used_gt]
    return tp, unmatched_det, unmatched_gt


def analyze_fn_neighborhood(ids, det_map, gt_map, mask_cache):
    """For each FN, find the nearest detection (matched or unmatched) and characterize."""
    fn_analysis = []
    for iid in ids:
        gt = gt_map.get(iid, [])
        det = det_map.get(iid, [])
        if not gt or not det:
            continue
        tp, unmatched_det, unmatched_gt = match_both_unmatched(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            D,
        )
        # For each unmatched GT (FN), find nearest detection (any detection)
        for ui in unmatched_gt:
            g = gt[ui].astype(np.float32)
            nearest_dist, nearest_det_idx = float('inf'), -1
            for di, d in enumerate(det):
                dist = min_corner_dist(g, [d.astype(np.float32)])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_det_idx = di
            # Check if nearest detection is "real"
            is_real = False
            if nearest_det_idx >= 0:
                is_real = is_detection_real(
                    det[nearest_det_idx].astype(np.int32),

                    mask_cache[iid],
                )
            fn_analysis.append({
                'iid': iid,
                'gt_idx': ui,
                'corner_d': nearest_dist,
                'nearest_det_idx': nearest_det_idx,
                'nearest_det_is_real': is_real,
                'category': (
                    'imprecise_match' if nearest_dist < D and is_real
                    else 'truly_missed' if nearest_dist >= D or not is_real
                    else 'imprecise_match_no_real'
                ),
            })
    return fn_analysis


def analyze_real_unlabeled(ids, det_map, gt_map, mask_cache):
    """For each real_unlabeled detection, check if it's near existing GT (extension)
    or isolated (new unannotated-real box)."""
    ru_analysis = []
    for iid in ids:
        gt = gt_map.get(iid, [])
        det = det_map.get(iid, [])
        if not det:
            continue
        # Find unmatched detections
        tp, unmatched_det, _ = match_both_unmatched(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            D,
        )
        for di in unmatched_det:
            d = det[di]
            # Check if real
            is_real = is_detection_real(
                d.astype(np.int32), mask_cache[iid]
            )
            if not is_real:
                continue
            # Find nearest GT
            nearest_gt_dist = float('inf')
            for g in gt:
                dist = min_corner_dist(g, [d.astype(np.float32)])
                if dist < nearest_gt_dist:
                    nearest_gt_dist = dist
            ru_analysis.append({
                'iid': iid,
                'det_idx': di,
                'nearest_gt_dist': nearest_gt_dist,
                'category': (
                    'gt_extension' if nearest_gt_dist < D
                    else 'isolated_unannotated'
                ),
            })
    return ru_analysis


def recompute_with_completion(fn_analysis, ru_analysis, ids, det_map, gt_map, mask_cache):
    """Re-compute metrics assuming GT completion.

    1. Add 'isolated_unannotated' real_unlabeled to GT (new GT boxes)
    2. Add 'gt_extension' real_unlabeled to GT (also new GT boxes, but very close to existing)
    3. Re-compute TP/FP at multiple D values
    """
    base_metrics_per_d = {}
    for dd in DIST_VALUES:
        det_map_d = det_map
        TP = FN = FP = 0
        n_gt = 0
        n_det = 0
        for iid in ids:
            gt = gt_map.get(iid, [])
            det = det_map_d.get(iid, [])
            tp, unmatched = match_boxes_point(
                [d.astype(np.int32) for d in det],
                [g.astype(np.int32) for g in gt],
                float(dd),
            )
            TP += tp
            FN += len(gt) - tp
            FP += len(unmatched)
            n_gt += len(gt)
            n_det += len(det)
        recall = TP / n_gt * 100 if n_gt else 0
        prec = TP / n_det * 100 if n_det else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        base_metrics_per_d[str(int(dd))] = {
            'tp': TP, 'fn': FN, 'fp': FP,
            'n_gt': n_gt, 'n_det': n_det,
            'recall': round(recall, 2),
            'precision': round(prec, 2),
            'f1': round(f1, 2),
        }

    # Add real_unlabeled to GT (augmented GT)
    # Build new GT map: original GT + isolated_unannotated + gt_extension real_unlabeled
    new_gt_map = {iid: list(gt) for iid, gt in gt_map.items()}
    n_added_isolated = 0
    n_added_extension = 0
    for ru in ru_analysis:
        if ru['category'] == 'isolated_unannotated':
            new_gt_map[ru['iid']].append(det_map[ru['iid']][ru['det_idx']])
            n_added_isolated += 1
        elif ru['category'] == 'gt_extension':
            new_gt_map[ru['iid']].append(det_map[ru['iid']][ru['det_idx']])
            n_added_extension += 1

    # Re-compute with augmented GT
    aug_metrics_per_d = {}
    for dd in DIST_VALUES:
        TP = FN = FP = 0
        n_gt = 0
        n_det = 0
        for iid in ids:
            gt = new_gt_map.get(iid, [])
            det = det_map.get(iid, [])
            tp, unmatched = match_boxes_point(
                [d.astype(np.int32) for d in det],
                [g.astype(np.int32) for g in gt],
                float(dd),
            )
            TP += tp
            FN += len(gt) - tp
            FP += len(unmatched)
            n_gt += len(gt)
            n_det += len(det)
        recall = TP / n_gt * 100 if n_gt else 0
        prec = TP / n_det * 100 if n_det else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        aug_metrics_per_d[str(int(dd))] = {
            'tp': TP, 'fn': FN, 'fp': FP,
            'n_gt': n_gt, 'n_det': n_det,
            'recall': round(recall, 2),
            'precision': round(prec, 2),
            'f1': round(f1, 2),
        }

    return base_metrics_per_d, aug_metrics_per_d, n_added_isolated, n_added_extension


def main():
    ids = get_ids()
    print(f"LOOP-20: GT label completion analysis on {len(ids)} real images (D={D})", flush=True)
    print(f"  pm_v5_1 reference: {PM5_REF}", flush=True)

    # Load pm_v5_1 detections
    det_map = {iid: load_boxes(PM5_REF, iid) for iid in ids}
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}

    # Build mask cache (for is_real_unlabeled)
    print(f"\nBuilding mask cache (dilated red mask per image)...", flush=True)
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

    # Get pm_v5_1 baseline metrics
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    n_gt = 0
    n_det = 0
    for iid in ids:
        gt = gt_map.get(iid, [])
        det = det_map.get(iid, [])
        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            D,
        )
        TP += tp
        FN += len(gt) - tp
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di].astype(np.int32), exp.get(iid, 0), len(gt), mask_cache[iid]):
                real_unl += 1
            else:
                true_fp += 1
        n_gt += len(gt)
        n_det += len(det)
    pm5_d5 = {
        'tp': TP, 'fn': FN, 'raw_fp': raw_fp, 'true_fp': true_fp,
        'real_unlabeled': real_unl,
        'n_gt': n_gt, 'n_det': n_det,
    }
    print(f"\npm_v5_1 baseline (D={D}):")
    print(f"  TP={TP} FN={FN} rawFP={raw_fp} trueFP={true_fp} real_unl={real_unl}")
    print(f"  n_gt={n_gt} n_det={n_det}")

    # Step 1: Analyze 245 FN
    print(f"\n[Step 1] Analyzing 245 FN (finding nearest detection per FN)...", flush=True)
    fn_analysis = analyze_fn_neighborhood(ids, det_map, gt_map, mask_cache)

    # Categorize
    imprecise = [f for f in fn_analysis if f['category'] == 'imprecise_match']
    truly_missed = [f for f in fn_analysis if f['category'] == 'truly_missed']
    other = [f for f in fn_analysis if f['category'] not in ('imprecise_match', 'truly_missed')]
    print(f"  imprecise_match (nearest det <{D}px and is real): {len(imprecise)}")
    print(f"  truly_missed (no real det nearby): {len(truly_missed)}")
    print(f"  other: {len(other)}")

    # Of the imprecise_match: by distance
    imprecise_close = [f for f in imprecise if f['corner_d'] < 3]
    imprecise_5 = [f for f in imprecise if 3 <= f['corner_d'] < 5]
    print(f"    imprecise breakdown: <3px={len(imprecise_close)}, 3-5px={len(imprecise_5)}")

    # Step 2: Analyze 136 real_unlabeled
    print(f"\n[Step 2] Analyzing {real_unl} real_unlabeled detections...", flush=True)
    ru_analysis = analyze_real_unlabeled(ids, det_map, gt_map, mask_cache)
    n_isolated = sum(1 for r in ru_analysis if r['category'] == 'isolated_unannotated')
    n_extension = sum(1 for r in ru_analysis if r['category'] == 'gt_extension')
    print(f"  isolated_unannotated (new GT entries): {n_isolated}")
    print(f"  gt_extension (near existing GT): {n_extension}")

    # Step 3: Re-compute with completed GT
    print(f"\n[Step 3] Re-computing metrics with completed GT...", flush=True)
    base_m, aug_m, n_added_iso, n_added_ext = recompute_with_completion(
        fn_analysis, ru_analysis, ids, det_map, gt_map, mask_cache
    )

    # Step 4: Summary
    print("\n" + "=" * 78)
    print(f"GT LABEL COMPLETION ANALYSIS (D={D}, 500 imgs)")
    print("=" * 78)
    print(f"\npm_v5_1 baseline:")
    print(f"  TP={pm5_d5['tp']} FN={pm5_d5['fn']} trueFP={pm5_d5['true_fp']} "
          f"real_unl={pm5_d5['real_unlabeled']}")
    print(f"  recall={pm5_d5['tp']/pm5_d5['n_gt']*100:.1f}% "
          f"corr_f1=71.5%")
    print(f"\nFN neighborhood analysis (245 FN):")
    print(f"  imprecise_match (pm_v5 sees it nearby): {len(imprecise)} / 245")
    print(f"    <3px: {len(imprecise_close)} (would match at D=3)")
    print(f"    3-5px: {len(imprecise_5)} (match at D=5 already — count is in baseline)")
    print(f"  truly_missed (no real det nearby): {len(truly_missed)} / 245")
    print(f"  other: {len(other)}")
    print(f"\nreal_unlabeled analysis (136 boxes):")
    print(f"  isolated_unannotated (new GT entries): {n_isolated}")
    print(f"  gt_extension (near existing GT): {n_extension}")
    print(f"\nGT completion impact:")
    print(f"  Adding {n_isolated} isolated + {n_extension} extension to GT")
    print(f"  New GT size: {pm5_d5['n_gt']} + {n_isolated + n_extension} = "
          f"{pm5_d5['n_gt'] + n_isolated + n_extension}")
    print(f"\n  Per-D metrics with completed GT:")
    print(f"  {'D':>3s}  {'TP_base':>7s}  {'TP_aug':>6s}  {'recall_base':>11s}  "
          f"{'recall_aug':>10s}  {'f1_base':>7s}  {'f1_aug':>6s}")
    for dd in ['3', '5', '8', '12']:
        b = base_m[dd]
        a = aug_m[dd]
        print(f"  {dd:>3s}  {b['tp']:>7d}  {a['tp']:>6d}  "
              f"{b['recall']:>10.1f}%  {a['recall']:>9.1f}%  "
              f"{b['f1']:>6.1f}%  {a['f1']:>5.1f}%")
    print("=" * 78)

    # Save results
    results = {
        'pm5_d5': pm5_d5,
        'fn_analysis': {
            'total': len(fn_analysis),
            'imprecise_match': len(imprecise),
            'imprecise_close_3px': len(imprecise_close),
            'imprecise_3_to_5px': len(imprecise_5),
            'truly_missed': len(truly_missed),
            'other': len(other),
        },
        'real_unlabeled_analysis': {
            'total': len(ru_analysis),
            'isolated_unannotated': n_isolated,
            'gt_extension': n_extension,
        },
        'gt_completion': {
            'baseline_gt_size': pm5_d5['n_gt'],
            'added_isolated': n_added_iso,
            'added_extension': n_added_ext,
            'new_gt_size': pm5_d5['n_gt'] + n_added_iso + n_added_ext,
        },
        'metrics_base': base_m,
        'metrics_augmented': aug_m,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()