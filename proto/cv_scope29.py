"""LOOP-29 (Path-4): New quad extraction algorithms.

LOOP-24 found Stage 4 (base_quads) has the largest causal effect (|theta|*std=0.700).
LOOP-20 found 226/245 FN are truly missed — pipeline doesn't generate candidates.
The 3 existing methods (hull/edgefit/mar) are exhausted (LOOPs 7-12, 21-23).

New algorithms:
  A: hough_isect  — HoughLinesP on red mask, group into parallel pairs,
                    intersect to form quads. Works on broken contours.
  B: multi_eps    — Finer approxPolyDP epsilon grid + convex hull variants.
  C: split_curve  — Split large contours at high-curvature points,
                    fit quads to sub-contours. Handles merged boxes.

Each algorithm is added as a 4th union member to pm_v5_1.
GATE: TP > 502 AND corr_f1 > 71.5 at D=5.

Output:
  benchmark/cv_scope29.json
  tmp_sc29_<algo>/  (detection JSONs)
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected, iou_obb
from canny_quad import get_ids, to_json, denoise, _quad_adaptive, _side_ok, _order_quad
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES,
                       _edgefit_corners, _sector_corners, _fit_line_tls, _line_isect)
from cv_scope4 import HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import (load_boxes, match_boxes_point, is_real_unlabeled,
                             red_support)
from quad_ortho import is_orthogonal_quad

RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope29.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o1000_ba")
BASELINE_TP = 502
BASELINE_F1 = 71.5
D = 5.0


# ── Algorithm A: Hough-line intersection ──

def hough_isect_quads(img, p):
    """Detect quads by: HoughLinesP on red mask -> group parallel pairs -> intersect.

    Unlike cv_scope7.hough_quads (which bridges gaps then runs contour pipeline),
    this works directly on line segments to form quads from 4 intersecting lines.
    """
    H, W = img.shape[:2]
    mask = red_mask(img)
    if p['denoise_min_area'] > 0:
        mask = denoise(mask, p['denoise_min_area'])

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                            minLineLength=int(p['min_side']),
                            maxLineGap=15)
    if lines is None:
        return []

    segs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 2:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180
        segs.append({
            'p1': np.array([x1, y1], float),
            'p2': np.array([x2, y2], float),
            'mid': np.array([(x1 + x2) / 2, (y1 + y2) / 2], float),
            'dir': np.array([dx / L, dy / L]),
            'ang': ang,
            'length': L,
        })

    if len(segs) < 4:
        return []

    # Group lines by angle (within 8 degrees), keep top 15 longest per group
    segs.sort(key=lambda s: -s['length'])
    segs = segs[:80]  # limit total lines
    segs.sort(key=lambda s: s['ang'])
    groups = []
    current = [segs[0]]
    for s in segs[1:]:
        if abs(s['ang'] - current[-1]['ang']) < 8:
            current.append(s)
        else:
            if len(current) >= 1:
                groups.append(current[:15])
            current = [s]
    groups.append(current[:15])

    # For each pair of angle groups (roughly perpendicular), try to form quads
    quads = []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i:
                continue
            mean_ang1 = np.mean([s['ang'] for s in g1])
            mean_ang2 = np.mean([s['ang'] for s in g2])
            ang_diff = abs(mean_ang1 - mean_ang2)
            ang_diff = min(ang_diff, 180 - ang_diff)
            if not (70 <= ang_diff <= 110):
                continue

            # Try pairs within each group (parallel lines)
            for a1 in g1:
                for a2 in g1:
                    if a2 is a1:
                        continue
                    perp_dist_a = abs(a1['dir'][0] * (a2['mid'][1] - a1['p1'][1]) -
                                      a1['dir'][1] * (a2['mid'][0] - a1['p1'][0]))
                    if perp_dist_a < p['min_side'] or perp_dist_a > 300:
                        continue
                    for b1 in g2:
                        for b2 in g2:
                            if b2 is b1:
                                continue
                            perp_dist_b = abs(b1['dir'][0] * (b2['mid'][1] - b1['p1'][1]) -
                                              b1['dir'][1] * (b2['mid'][0] - b1['p1'][0]))
                            if perp_dist_b < p['min_side'] or perp_dist_b > 300:
                                continue

                            corners = []
                            for la, lb in [(a1, b1), (a1, b2), (a2, b2), (a2, b1)]:
                                m1, d1 = la['mid'], la['dir']
                                m2, d2 = lb['mid'], lb['dir']
                                pt = _line_isect(m1, d1, m2, d2)
                                if pt is None:
                                    corners = None
                                    break
                                corners.append(pt)
                            if corners is None or len(corners) != 4:
                                continue

                            q = np.array(corners, float)
                            if np.any(q[:, 0] < 0) or np.any(q[:, 0] >= W) or \
                               np.any(q[:, 1] < 0) or np.any(q[:, 1] >= H):
                                continue
                            qq = _pass(q, W, H, p)
                            if qq is not None:
                                quads.append(qq)
                                if len(quads) > 200:
                                    return dedup_keep_order(quads, p['dedup_iou'])

    return dedup_keep_order(quads, p['dedup_iou'])


# ── Algorithm B: Multi-epsilon approxPolyDP ──

def multi_eps_quads(img, p):
    """Try a finer epsilon grid + convex hull variants for quad extraction."""
    H, W = img.shape[:2]
    mask = red_mask(img)
    if p['denoise_min_area'] > 0:
        mask = denoise(mask, p['denoise_min_area'])
    cnts = outer_contours(mask, p)

    fine_eps = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
                0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28]
    quads = []

    for c in cnts:
        # Standard approxPolyDP with fine grid
        peri = cv2.arcLength(c, True)
        for eps in fine_eps:
            ap = cv2.approxPolyDP(c, eps * peri, True)
            if len(ap) == 4:
                q = _pass(ap.reshape(4, 2).astype(float), W, H, p)
                if q is not None:
                    quads.append(q)
                break

        # Convex hull with fine grid
        hull = cv2.convexHull(c)
        peri_h = cv2.arcLength(hull, True)
        for eps in fine_eps:
            ap = cv2.approxPolyDP(hull, eps * peri_h, True)
            if len(ap) == 4:
                q = _pass(ap.reshape(4, 2).astype(float), W, H, p)
                if q is not None:
                    quads.append(q)
                break

        # Also try approxPolyDP on the contour with 5-6 vertices, then take
        # the 4 closest to a rectangle
        for eps in fine_eps:
            ap = cv2.approxPolyDP(c, eps * peri, True)
            if len(ap) in (5, 6):
                pts = ap.reshape(-1, 2).astype(float)
                # Try all 4-vertex subsets
                from itertools import combinations
                for combo in combinations(range(len(pts)), 4):
                    q4 = pts[list(combo)]
                    q4 = _order_quad(q4)
                    qq = _pass(q4, W, H, p)
                    if qq is not None:
                        quads.append(qq)
                break

    return dedup_keep_order(quads, p['dedup_iou'])


# ── Algorithm C: Contour splitting at high curvature ──

def split_curve_quads(img, p):
    """Split large contours at high-curvature points, fit quads to sub-contours."""
    H, W = img.shape[:2]
    mask = red_mask(img)
    if p['denoise_min_area'] > 0:
        mask = denoise(mask, p['denoise_min_area'])
    cnts = outer_contours(mask, p)

    quads = []

    for c in cnts:
        # First try standard methods
        base_qh = _quad_adaptive(cv2.convexHull(c), p["approx_eps_list"])
        if base_qh is not None:
            q = _pass(base_qh, W, H, p)
            if q is not None:
                quads.append(q)

        # Check if contour is "large" (might contain multiple boxes)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if area < 5000 or peri < 200:
            continue

        # Compute curvature at each point
        pts = c.reshape(-1, 2).astype(float)
        n = len(pts)
        if n < 20:
            continue

        # Curvature via discrete second derivative
        k = 5  # window size
        curvature = np.zeros(n)
        for i in range(n):
            prev = pts[(i - k) % n]
            curr = pts[i]
            nxt = pts[(i + k) % n]
            v1 = curr - prev
            v2 = nxt - curr
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                curvature[i] = abs(cross) / (n1 * n2)

        # Find high-curvature points (corners)
        threshold = np.percentile(curvature, 90)
        high_curv = np.where(curvature > threshold)[0]
        if len(high_curv) < 4:
            continue

        # Cluster high-curvature points (merge nearby ones)
        clusters = []
        current = [high_curv[0]]
        for idx in high_curv[1:]:
            if idx - current[-1] < 10:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)

        if len(clusters) < 4:
            continue

        # If exactly 4 clusters, use their centroids as quad corners
        if len(clusters) == 4:
            corners = np.array([pts[np.mean(cl, dtype=int)] for cl in clusters])
            corners = _order_quad(corners)
            q = _pass(corners, W, H, p)
            if q is not None:
                quads.append(q)

        # If >4 clusters, try all 4-element subsets (limit to avoid explosion)
        elif len(clusters) <= 8:
            from itertools import combinations
            centroids = [pts[int(np.mean(cl))] for cl in clusters]
            for combo in combinations(range(len(centroids)), 4):
                corners = np.array([centroids[i] for i in combo])
                corners = _order_quad(corners)
                q = _pass(corners, W, H, p)
                if q is not None:
                    quads.append(q)

    return dedup_keep_order(quads, p['dedup_iou'])


ALGORITHMS = {
    'hough_isect': ('A: Hough-line intersection', hough_isect_quads),
    'multi_eps':   ('B: Multi-epsilon approxPolyDP', multi_eps_quads),
    'split_curve': ('C: Contour splitting', split_curve_quads),
}


def evaluate(ids, det_map, gt_map, mask_cache, exp, dist_thr):
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    for iid in ids:
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        gt = [g.astype(np.int32) for g in gt_map.get(iid, [])]
        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp; FN += len(gt) - tp
        raw_fp += len(unmatched)
        csv_n = exp.get(iid, 0)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_cache[iid]):
                real_unl += 1
            else:
                true_fp += 1
    n_gt = TP + FN
    n_det = TP + raw_fp
    recall = TP / n_gt * 100 if n_gt else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0
    return {
        'tp': TP, 'fn': FN, 'raw_fp': raw_fp, 'true_fp': true_fp,
        'real_unlabeled': real_unl, 'n_det': n_det,
        'recall': round(recall, 1), 'corrected_f1': round(f1_c, 1),
    }


def main():
    ids = get_ids()
    exp = load_expected()
    print(f"LOOP-29: New quad extraction algorithms on {len(ids)} real images", flush=True)
    print(f"  Baseline: pm_v5_1 TP={BASELINE_TP}, corr_f1={BASELINE_F1}", flush=True)

    # Load GT and pm_v5_1 detections
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    pm5_det = {iid: [q.astype(np.int32) for q in load_boxes(PM5_REF, iid)] for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"  GT boxes: {total_gt}", flush=True)
    print(f"  pm_v5_1 detections: {sum(len(v) for v in pm5_det.values())}", flush=True)

    # Build mask cache
    print(f"\nBuilding mask cache...", flush=True)
    mask_cache = {}
    img_cache = {}
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        img_cache[iid] = img
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask_cache[iid] = mask_d
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] cached", flush=True)
    print(f"  {len(mask_cache)} cached", flush=True)

    # Baseline metrics
    p = pm5_params()
    print(f"\nBaseline (pm_v5_1):", flush=True)
    base_results = {}
    for dd in DIST_VALUES:
        m = evaluate(ids, pm5_det, gt_map, mask_cache, exp, float(dd))
        base_results[str(dd)] = m
        print(f"  D={dd:2d}: TP={m['tp']:4d} recall={m['recall']:5.1f}% "
              f"corr_f1={m['corrected_f1']:5.1f}", flush=True)

    # Test each algorithm
    all_results = {'baseline': base_results}

    for algo_name, (label, algo_fn) in ALGORITHMS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"  {label}", flush=True)
        print(f"{'='*60}", flush=True)

        # Run algorithm
        algo_det = {}
        total_algo = 0
        t0 = time.time()
        out_dir = os.path.join(ROOT, f"tmp_sc29_{algo_name}")
        os.makedirs(out_dir, exist_ok=True)
        for n, iid in enumerate(ids):
            img = img_cache.get(iid)
            if img is None:
                algo_det[iid] = []
                continue
            quads = algo_fn(img, p)
            algo_det[iid] = quads
            total_algo += len(quads)
            with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, [np.asarray(q, np.int32) for q in quads]), f,
                          ensure_ascii=False)
            if (n + 1) % 100 == 0:
                print(f"  [{n+1}/{len(ids)}] det={total_algo}", flush=True)
        elapsed = time.time() - t0
        print(f"  {label}: {total_algo} detections in {elapsed:.1f}s", flush=True)

        # Evaluate algo alone
        algo_results = {}
        for dd in DIST_VALUES:
            m = evaluate(ids, algo_det, gt_map, mask_cache, exp, float(dd))
            algo_results[str(dd)] = m
        m5 = algo_results['5']
        print(f"  Algo only D=5: TP={m5['tp']} recall={m5['recall']}% "
              f"corr_f1={m5['corrected_f1']}", flush=True)

        # Union with pm_v5_1
        union_det = {}
        total_union = 0
        union_dir = os.path.join(ROOT, f"tmp_sc29_union_{algo_name}")
        os.makedirs(union_dir, exist_ok=True)
        for iid in ids:
            combined = list(pm5_det.get(iid, [])) + list(algo_det.get(iid, []))
            union_det[iid] = dedup_keep_order(combined, p['dedup_iou'])
            total_union += len(union_det[iid])
            with open(os.path.join(union_dir, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, [np.asarray(q, np.int32) for q in union_det[iid]]), f,
                          ensure_ascii=False)
        print(f"  Union: {total_union} detections", flush=True)

        # Evaluate union
        union_results = {}
        for dd in DIST_VALUES:
            m = evaluate(ids, union_det, gt_map, mask_cache, exp, float(dd))
            union_results[str(dd)] = m
        m5u = union_results['5']
        d_tp = m5u['tp'] - BASELINE_TP
        d_f1 = m5u['corrected_f1'] - BASELINE_F1
        verdict = "WIN" if m5u['tp'] > BASELINE_TP and m5u['corrected_f1'] > BASELINE_F1 else \
                  "TIE" if m5u['tp'] == BASELINE_TP and m5u['corrected_f1'] == BASELINE_F1 else \
                  "MARGINAL" if abs(d_tp) < 5 and abs(d_f1) < 1.0 else "DEAD"
        print(f"  Union D=5: TP={m5u['tp']} ({d_tp:+d}) recall={m5u['recall']}% "
              f"corr_f1={m5u['corrected_f1']} ({d_f1:+.1f}) -> {verdict}", flush=True)

        all_results[algo_name] = {
            'label': label,
            'algo_only': algo_results,
            'union': union_results,
            'n_algo_det': total_algo,
            'n_union_det': total_union,
            'verdict': verdict,
        }

    # Summary
    print(f"\n{'='*90}")
    print(f"LOOP-29 SUMMARY: New quad extraction algorithms")
    print(f"{'='*90}")
    print(f"\n{'Algorithm':30s} {'algo_det':>8s} {'union_det':>9s}  "
          f"{'TP@5':>5s} {'dTP':>4s}  {'corr_f1':>7s} {'df1':>5s}  {'verdict':>8s}")
    b5 = base_results['5']
    print(f"{'baseline (pm_v5_1)':30s} {'':>8s} {sum(len(v) for v in pm5_det.values()):>9d}  "
          f"{b5['tp']:>5d} {'':>4s}  {b5['corrected_f1']:>6.1f}% {'':>5s}  {'':>8s}")
    for algo_name, (label, _) in ALGORITHMS.items():
        r = all_results[algo_name]
        u5 = r['union']['5']
        d_tp = u5['tp'] - BASELINE_TP
        d_f1 = u5['corrected_f1'] - BASELINE_F1
        print(f"{label:30s} {r['n_algo_det']:>8d} {r['n_union_det']:>9d}  "
              f"{u5['tp']:>5d} {d_tp:>+4d}  {u5['corrected_f1']:>6.1f}% {d_f1:>+4.1f}  "
              f"{r['verdict']:>8s}")
    print(f"{'='*90}")

    # Save
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()