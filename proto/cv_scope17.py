"""LOOP-17: Fragment-joining assembly — pair disconnected red components to form quads.

Hypothesis: at brokenness >= 0.05, the outline splits into 2-4 disconnected
fragments. Each fragment is a thin elongated rectangle (1-2px thick, ~50-80px
long). If we pair PARALLEL fragments, we can form a candidate quad that
spans the gap.

Algorithm:
  1. Find red connected components (not contours — raw components).
  2. For each component, compute its minAreaRect -> fragment info
     (cx, cy, length, thickness, angle).
  3. For each pair of fragments:
     a. Check parallel: |angle1 - angle2| within tolerance
     b. Check thickness: both should be thin (< 8px)
     c. Check distance: perpendicular distance = box's "short dim"
     d. Check overlap: along-axis overlap = box's "long dim"
     e. Form candidate quad = (f1, f2, perpendicular closures)
  4. Apply admission filter (ortho, boundary, aspect) -> keep valid quads
  5. Add to pm_v5 hybrid as 4th family (after hull/edgefit/mar)

Tested variants:
  - base: pm_v5 only (baseline)
  - fjoin: pm_v5 + fragment-joining (parallel pairs)
  - fjoin_loose: pm_v5 + fragment-joining with looser angle/distance tol
  - fjoin_strict: pm_v5 + fragment-joining with stricter angle/distance tol

Usage: python proto/cv_scope17.py [--variants VARIANTS] [--max-pairs N]
"""
import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import obb_corners, draw_obb, compute_gt_minarearect
from cv_scope16 import gen_dataset
from eval_pointmatch import match_boxes_point

BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
DET_DIR = os.path.join(ROOT, "tmp_sc17_det")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope17.json")
D = 5.0
N_BG = 50
N_PER_VALUE = 4
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 42
DEFAULT_OBB = {'box_w': 80, 'box_h': 50, 'angle': 15, 'stroke_w': 2}


# ── Fragment-joining core ──

def get_red_components(mask, min_area=40, connectivity=8):
    """Return list of {cx, cy, length, thickness, angle, area, contour}."""
    n, lbls, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    components = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp_mask = (lbls == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        rect = cv2.minAreaRect(cnts[0])
        (cx, cy), (rw, rh), angle = rect
        # Normalize: long axis is the larger one, angle in [0, 180)
        if rw < rh:
            rw, rh = rh, rw
            angle += 90
        angle = angle % 180
        if rw < 1:
            rw = 1
        if rh < 1:
            rh = 1
        components.append({
            'cx': float(cx), 'cy': float(cy),
            'length': float(max(rw, rh)),
            'thickness': float(min(rw, rh)),
            'angle': float(angle),
            'area': float(area),
            'contour': cnts[0],
        })
    return components


def parallel_pair_quad(c1, c2, p, max_aspect_extra=2.0):
    """Form a candidate quad from 2 parallel fragments.

    Returns (corners, box_dims) or None.
    """
    angle_rad = np.radians(c1['angle'])
    d_long = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
    d_perp = np.array([-d_long[1], d_long[0]], dtype=np.float32)

    p1 = np.array([c1['cx'], c1['cy']], dtype=np.float32)
    p2 = np.array([c2['cx'], c2['cy']], dtype=np.float32)

    # Perpendicular distance (box's "short dim")
    d_perp_dist = abs(float(np.dot(p2 - p1, d_perp)))
    if d_perp_dist < p['min_side'] or d_perp_dist > p['max_side']:
        return None

    # Projection along long axis (box's "long dim")
    proj1 = float(np.dot(p1, d_long))
    proj2 = float(np.dot(p2, d_long))
    half1 = c1['length'] / 2
    half2 = c2['length'] / 2
    long_start = min(proj1 - half1, proj2 - half2)
    long_end = max(proj1 + half1, proj2 + half2)
    long_len = long_end - long_start
    if long_len < p['min_side'] or long_len > p['max_side']:
        return None

    # Aspect ratio
    aspect = max(long_len, d_perp_dist) / max(1e-6, min(long_len, d_perp_dist))
    if aspect > p['max_aspect'] * max_aspect_extra:
        return None

    # Form the box: 2 long edges = the parallel fragments
    # Perpendicular axis center: (p1 + p2) / 2 projected onto d_perp
    perp_center = float(np.dot((p1 + p2) / 2, d_perp))
    long_center = (long_start + long_end) / 2

    # 4 corners
    corners = []
    for sgn_d in [-1, 1]:
        for sgn_l in [-1, 1]:
            corner = (long_center + sgn_l * long_len / 2) * d_long + \
                      (perp_center + sgn_d * d_perp_dist / 2) * d_perp
            corners.append(corner)
    return np.array(corners, dtype=np.float32), (long_len, d_perp_dist)


def perpendicular_pair_quad(c1, c2, p, max_distance=200.0):
    """Form a candidate quad from 2 perpendicular fragments sharing a corner.

    Returns quad or None.
    """
    # Normalize so that c1's angle is 0 (horizontal) by rotating both
    angle1 = c1['angle']
    angle2 = c2['angle']
    # Perpendicular check
    diff = abs(angle1 - angle2)
    if abs(diff - 90) > 15 and abs(diff - 180) > 15:
        # Not perpendicular within tolerance
        return None
    # ... (more complex; skip for now)
    return None


def fragment_joining_quads(mask, p, parallel_tol=10.0, max_thickness=8.0,
                              min_length=15.0, max_pairs=300,
                              max_distance=200.0, max_aspect_extra=2.0):
    """Find candidate quads via parallel-fragment-pairing.

    Also tries perpendicular pairs (fragments forming an L).
    """
    components = get_red_components(mask, min_area=40)
    if len(components) < 2:
        return []

    # Filter to thin elongated fragments
    thin = [c for c in components
            if c['thickness'] < max_thickness and c['length'] > min_length]
    if len(thin) < 2:
        return []

    quads = []
    pair_count = 0
    n = len(thin)
    # Sort by length to prioritize long fragments
    indices = sorted(range(n), key=lambda i: -thin[i]['length'])

    for ii in range(n):
        i = indices[ii]
        c1 = thin[i]
        for jj in range(ii + 1, n):
            j = indices[jj]
            c2 = thin[j]

            # Skip if centers too far apart
            dist = np.sqrt((c1['cx'] - c2['cx']) ** 2 + (c1['cy'] - c2['cy']) ** 2)
            if dist > max_distance:
                continue

            # Check parallel
            angle_diff = abs(c1['angle'] - c2['angle'])
            if angle_diff > parallel_tol and abs(angle_diff - 180) > parallel_tol:
                # Try perpendicular (skip for now, complex)
                continue

            # Form quad
            result = parallel_pair_quad(c1, c2, p, max_aspect_extra)
            if result is None:
                continue
            quad, _ = result
            quads.append(quad)
            pair_count += 1
            if pair_count >= max_pairs:
                return quads
    return quads


# ── Pipeline integration ──

def run_variant(name, dataset, p, mode='baseline',
                parallel_tol=10.0, max_thickness=8.0,
                min_length=15.0, max_pairs=300, max_distance=200.0,
                max_aspect_extra=2.0):
    """Run pm_v5 with optional fragment-joining."""
    det_map = {}
    total = 0
    t0 = time.time()
    for item in dataset:
        img = item['img']
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
        # Add fragment-joining quads
        if mode != 'baseline':
            fjoin_quads = fragment_joining_quads(
                mask, p,
                parallel_tol=parallel_tol,
                max_thickness=max_thickness,
                min_length=min_length,
                max_pairs=max_pairs,
                max_distance=max_distance,
                max_aspect_extra=max_aspect_extra,
            )
            # Apply admission filter
            for q in fjoin_quads:
                qf = _pass(q, W, H, p)
                if qf is not None:
                    hy_quads.append(qf)
        try:
            hy = dedup_keep_order(hy_quads, p['dedup_iou'])
        except cv2.error:
            hy = dedup_keep_order(
                [q for q in hy_quads if q is not None and len(q) == 4],
                p['dedup_iou'],
            )
        det_map[item['iid']] = hy
        total += len(hy)
    dt = time.time() - t0
    print(f"  {name}: {total} boxes in {dt:.1f}s "
          f"({total / len(dataset):.2f} dets/img)", flush=True)
    return det_map


def evaluate(det_map, dataset, dist_thr=5.0):
    metrics = {'total': {'tp': 0, 'fn': 0, 'fp': 0, 'n_gt': 0, 'n_det': 0}}
    for item in dataset:
        b = item['brokenness']
        if b not in metrics:
            metrics[b] = {'tp': 0, 'fn': 0, 'fp': 0, 'n_gt': 0, 'n_det': 0}
        gt = item['gt_quads']
        det = det_map.get(item['iid'], [])
        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            dist_thr,
        )
        metrics[b]['tp'] += tp
        metrics[b]['fn'] += len(gt) - tp
        metrics[b]['fp'] += len(unmatched)
        metrics[b]['n_gt'] += len(gt)
        metrics[b]['n_det'] += len(det)
        metrics['total']['tp'] += tp
        metrics['total']['fn'] += len(gt) - tp
        metrics['total']['fp'] += len(unmatched)
        metrics['total']['n_gt'] += len(gt)
        metrics['total']['n_det'] += len(det)
    summary = {}
    for k, m in metrics.items():
        recall = m['tp'] / m['n_gt'] * 100 if m['n_gt'] else 0
        prec = m['tp'] / m['n_det'] * 100 if m['n_det'] else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        summary[str(k)] = {**m, 'recall': round(recall, 2),
                            'precision': round(prec, 2), 'f1': round(f1, 2)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', type=str,
                    default='baseline,fjoin,fjoin_strict,fjoin_loose',
                    help='Comma-separated variants')
    args = ap.parse_args()

    p = pm5_params()
    print(f"LOOP-17: Fragment-joining assembly on synth X-ray (D={D})", flush=True)
    print(f"  pm5: min_area={p['min_area']:.0f} denoise={p['denoise_min_area']:.0f} "
          f"min_side={p['min_side']:.0f} max_aspect={p['max_aspect']:.0f}", flush=True)

    print(f"\nGenerating dataset (n_bg={N_BG}, n_per_value={N_PER_VALUE})...", flush=True)
    t0 = time.time()
    dataset = gen_dataset()
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)

    all_results = {'pm5_params': {k: v for k, v in p.items()
                                   if not isinstance(v, (list, tuple))},
                   'brokenness_values': BROKENNESS_VALUES,
                   'n_images': len(dataset),
                   'dist_thr': D,
                   'methods': {}}

    # Define variants
    variant_configs = {
        'baseline':      {'mode': 'baseline'},
        'fjoin':         {'mode': 'fjoin',
                           'parallel_tol': 10.0, 'max_thickness': 8.0,
                           'min_length': 15.0, 'max_pairs': 300,
                           'max_distance': 200.0, 'max_aspect_extra': 2.0},
        'fjoin_strict':  {'mode': 'fjoin',
                           'parallel_tol': 5.0, 'max_thickness': 5.0,
                           'min_length': 20.0, 'max_pairs': 200,
                           'max_distance': 150.0, 'max_aspect_extra': 1.5},
        'fjoin_loose':   {'mode': 'fjoin',
                           'parallel_tol': 15.0, 'max_thickness': 12.0,
                           'min_length': 10.0, 'max_pairs': 500,
                           'max_distance': 300.0, 'max_aspect_extra': 3.0},
    }

    # Run baseline first
    bl_cfg = variant_configs['baseline']
    print(f"\n[Baseline] pm_v5 only...", flush=True)
    t0 = time.time()
    det_base = run_variant('baseline', dataset, p, **bl_cfg)
    dt = time.time() - t0
    m_base = evaluate(det_base, dataset, D)
    all_results['methods']['baseline'] = m_base
    bl = m_base['total']
    print(f"  baseline: TP={bl['tp']} FP={bl['fp']} FN={bl['fn']}  "
          f"recall={bl['recall']:.1f}% f1={bl['f1']:.1f}%", flush=True)
    for b in BROKENNESS_VALUES:
        bm = m_base[str(b)]
        print(f"    b={b:.2f}: TP={bm['tp']:>3d}/{bm['n_gt']:>3d}  "
              f"recall={bm['recall']:>5.1f}%  det={bm['n_det']}", flush=True)

    # Run each variant
    for vname in args.variants.split(','):
        if vname == 'baseline':
            continue
        if vname not in variant_configs:
            print(f"  WARNING: unknown variant {vname}, skipping", flush=True)
            continue
        cfg = variant_configs[vname]
        print(f"\n[Variant: {vname}] parallel_tol={cfg['parallel_tol']} "
              f"max_thickness={cfg['max_thickness']} min_length={cfg['min_length']} "
              f"max_pairs={cfg['max_pairs']} max_dist={cfg['max_distance']} "
              f"aspect_extra={cfg['max_aspect_extra']}...", flush=True)
        t0 = time.time()
        det_v = run_variant(vname, dataset, p, **cfg)
        dt = time.time() - t0
        m_v = evaluate(det_v, dataset, D)
        all_results['methods'][vname] = m_v
        v_total = m_v['total']
        delta_tp = v_total['tp'] - bl['tp']
        delta_fp = v_total['fp'] - bl['fp']
        delta_f1 = v_total['f1'] - bl['f1']
        if v_total['tp'] > bl['tp'] and v_total['fp'] <= bl['fp'] + 10:
            verdict = "WIN"
        elif v_total['tp'] >= bl['tp'] and v_total['fp'] <= bl['fp'] + 20:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"  {vname}: TP={v_total['tp']} FP={v_total['fp']} FN={v_total['fn']}  "
              f"recall={v_total['recall']:.1f}% f1={v_total['f1']:.1f}%  "
              f"(ΔTP={delta_tp:+d} ΔFP={delta_fp:+d} Δf1={delta_f1:+.1f}) [{verdict}]", flush=True)
        for b in BROKENNESS_VALUES:
            bm = m_v[str(b)]
            bbm = m_base[str(b)]
            dtp = bm['tp'] - bbm['tp']
            print(f"    b={b:.2f}: TP={bm['tp']:>3d}/{bm['n_gt']:>3d}  "
                  f"recall={bm['recall']:>5.1f}%  det={bm['n_det']}  (ΔTP={dtp:+d})", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, {len(dataset)} synth images)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':>16s}  {'TP':>5s}  {'FP':>5s}  {'recall':>6s}  "
          f"{'f1':>6s}  {'ΔTP':>5s}  {'ΔFP':>5s}  {'Δf1':>5s}  Verdict", flush=True)
    print(f"{'pm_v5 base':>16s}  {bl['tp']:>5d}  {bl['fp']:>5d}  "
          f"{bl['recall']:>5.1f}%  {bl['f1']:>5.1f}%  {'--':>5s}  {'--':>5s}  {'--':>5s}  baseline", flush=True)
    for vname in args.variants.split(','):
        if vname == 'baseline' or vname not in all_results['methods']:
            continue
        m = all_results['methods'][vname]['total']
        dtp = m['tp'] - bl['tp']
        dfp = m['fp'] - bl['fp']
        df1 = m['f1'] - bl['f1']
        if m['tp'] > bl['tp'] and m['fp'] <= bl['fp'] + 10:
            verdict = "WIN"
        elif m['tp'] >= bl['tp'] and m['fp'] <= bl['fp'] + 20:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"{vname:>16s}  {m['tp']:>5d}  {m['fp']:>5d}  "
              f"{m['recall']:>5.1f}%  {m['f1']:>5.1f}%  "
              f"{dtp:+5d}  {dfp:+5d}  {df1:+5.1f}  {verdict}", flush=True)
    print("=" * 78, flush=True)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()