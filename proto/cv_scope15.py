"""LOOP-15: Single-axis parameter optimization + orthogonal combination.

Workflow:
  Step 1: For each parameter axis, sweep values on the same realistic X-ray
          dataset. Dataset GT is rebuilt in real-time via compute_gt_minarearect
          (minAreaRect of rendered pixels for each injected OBB).
  Step 2: Pick the best value per axis.
  Step 3: Orthogonally combine all best values into a candidate config.
  Step 4: Evaluate the candidate on the realistic + 500-real benchmark.

Axes swept:
  smin           (HSV saturation min)        [60, 70, 80, 90, 100]
  vmin           (HSV value min)             [90, 100, 110, 120, 130]
  min_area       (min contour area)          [400, 700, 900, 1200, 2000]
  denoise_min    (denoise min area)          [20, 40, 60, 100]
  min_side       (min edge length)           [10, 15, 20, 25]
  max_aspect     (max aspect ratio)          [6, 8, 10, 12, 15]
  ortho_tol      (orthogonality tolerance)   [10, 15, 20]
  boundary_band  (boundary proximity band)   [5, 8, 12, 20]

Usage: python proto/cv_scope15.py [--n-bg N] [--n-per-value N] [--seed S]
"""
import sys, os, json, time, glob, copy, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask, LAB, IMG
from canny_quad import denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                        _edgefit_corners, _sector_corners, evaluate, ROOT, DIST_VALUES)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import obb_corners, draw_obb, compute_gt_minarearect
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids, to_json
from cv_hybrid import FIT_DEFAULT

from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
OUT_ROOT = os.path.join(ROOT, "tmp_sc15")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope15.json")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]


def list_bg_images(n_bg=30, seed=42):
    files = sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))
    rng = np.random.RandomState(seed)
    if n_bg >= len(files):
        return files
    idxs = rng.choice(len(files), n_bg, replace=False)
    return [files[i] for i in idxs]


def gen_dataset(n_bg, brokenness_values, n_per_value, seed=42, diverse=True):
    """Generate realistic dataset: inject 1 OBB per image into real X-ray bg.

    Each image rebuilds GT via compute_gt_minarearect (minAreaRect of rendered
    pixels in the box region).

    diverse=True: randomize box size, angle, position (including near boundary)
                  to make parameter sweeps meaningful.
    """
    bg_files = list_bg_images(n_bg, seed=seed)
    rng = np.random.RandomState(seed + 1)
    default = {
        'box_w': 80, 'box_h': 50, 'angle': 15,
        'stroke_w': 2, 'brokenness': 0,
    }
    dataset = []
    for b in brokenness_values:
        for k in range(n_per_value):
            bg_file = bg_files[k % len(bg_files)]
            bg = cv2.imread(bg_file)
            if bg is None:
                continue
            H, W = bg.shape[:2]
            if diverse:
                # Random size (40-200), random angle (0-90)
                bw = rng.randint(40, 200)
                bh = rng.randint(30, 120)
                ang = rng.randint(0, 90)
                # Random position: 50% mid-image, 30% near boundary, 20% off-center
                pos_kind = rng.random()
                if pos_kind < 0.5:
                    cx = rng.randint(100, max(101, W - 100))
                    cy = rng.randint(80, max(81, H - 80))
                elif pos_kind < 0.8:
                    # Near boundary (within 30px of edge)
                    side = rng.choice(['top', 'bottom', 'left', 'right'])
                    if side == 'top':
                        cy = rng.randint(0, 30)
                        cx = rng.randint(0, W)
                    elif side == 'bottom':
                        cy = rng.randint(max(0, H - 30), H)
                        cx = rng.randint(0, W)
                    elif side == 'left':
                        cx = rng.randint(0, 30)
                        cy = rng.randint(0, H)
                    else:
                        cx = rng.randint(max(0, W - 30), W)
                        cy = rng.randint(0, H)
                else:
                    cx = rng.randint(30, W - 30)
                    cy = rng.randint(30, H - 30)
            else:
                bw, bh, ang = default['box_w'], default['box_h'], default['angle']
                cx = rng.randint(80, max(81, W - 80))
                cy = rng.randint(60, max(61, H - 60))
            bp = dict(default)
            bp['cx'] = cx
            bp['cy'] = cy
            bp['box_w'] = bw
            bp['box_h'] = bh
            bp['angle'] = ang
            bp['brokenness'] = b
            img = bg.copy()
            corners = obb_corners(bp['cx'], bp['cy'], bp['box_w'], bp['box_h'],
                                   bp['angle'])
            draw_obb(img, corners, brokenness=bp['brokenness'],
                     stroke_w=bp['stroke_w'], color=(0, 0, 255), rng=rng,
                     brokenness_mode='corner')
            gt = compute_gt_minarearect(img, corners, padding=10)
            dataset.append({
                'img': img, 'gt_quads': [gt], 'bp': bp,
                'bg_file': os.path.basename(bg_file),
                'param_name': 'brokenness', 'param_value': b,
                'idx': len(dataset),
            })
    return dataset


def red_mask_with_hsv(img, smin, vmin):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def run_pipeline(dataset, p, smin, vmin, denoise_min):
    """Run pm_v5-like pipeline with custom HSV + denoise, using p for other params."""
    det_map = {}
    for item in dataset:
        img = item['img']
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
        det_map[item['idx']] = hy
    return det_map


def evaluate_det_map(det_map, dataset, dist_thr=5.0):
    TP = FN = FP = 0
    n_gt = 0
    n_det = 0
    for item in dataset:
        gt = item['gt_quads']
        det = det_map.get(item['idx'], [])
        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            dist_thr,
        )
        TP += tp
        FN += len(gt) - tp
        FP += len(unmatched)
        n_gt += len(gt)
        n_det += len(det)
    recall = TP / n_gt * 100 if n_gt else 0
    precision = TP / n_det * 100 if n_det else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'tp': TP, 'fn': FN, 'fp': FP,
        'n_gt': n_gt, 'n_det': n_det,
        'recall': round(recall, 2),
        'precision': round(precision, 2),
        'f1': round(f1, 2),
    }


def axis_sweep(name, values, dataset, base_p, base_smin, base_vmin, base_denoise):
    """Sweep a single parameter axis; return (best_value, best_metrics, all_results)."""
    p_base = dict(base_p)
    results = {}
    for v in values:
        p2 = dict(p_base)
        p2[name] = v
        # Re-route ortho_tol and boundary_band into the PM5 dict
        if name in ('ortho_tol', 'boundary_band', 'max_aspect', 'min_side', 'min_area'):
            for k in ('ortho_tol', 'boundary_band', 'max_aspect', 'min_side', 'min_area'):
                if k in p2:
                    p2[k] = p2[k]
        # Set smin/vmin/denoise separately
        det_map = run_pipeline(dataset, p2, base_smin, base_vmin, base_denoise)
        m = evaluate_det_map(det_map, dataset, D)
        results[v] = m
        print(f"    {name}={v}: TP={m['tp']:>3d}  recall={m['recall']:>5.1f}%  "
              f"FP={m['fp']:>3d}  prec={m['precision']:>5.1f}%  F1={m['f1']:>5.1f}", flush=True)
    # Pick best: max TP, then min FP
    best_v = max(results.keys(),
                 key=lambda v: (results[v]['tp'], -results[v]['fp']))
    print(f"    → BEST: {name}={best_v} (TP={results[best_v]['tp']}, FP={results[best_v]['fp']})", flush=True)
    return best_v, results[best_v], results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-bg', type=int, default=30)
    ap.add_argument('--n-per-value', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-combined', action='store_true')
    args = ap.parse_args()

    base_p = pm5_params()
    print(f"LOOP-15: Single-axis sweep + orthogonal combination (D={D})", flush=True)
    print(f"  pm5 baseline: {base_p}", flush=True)

    # Generate dataset ONCE, used for all axis sweeps
    print(f"\nGenerating realistic X-ray dataset (n_bg={args.n_bg}, n_per_value={args.n_per_value})...", flush=True)
    t0 = time.time()
    dataset = gen_dataset(args.n_bg, [0.0], args.n_per_value, seed=args.seed)
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s (brokenness=0.0 only, for fast iteration)", flush=True)

    # Baseline metrics
    print(f"\n[Baseline] pm_v5 default params...", flush=True)
    base_metrics = evaluate_det_map(
        run_pipeline(dataset, base_p, 80, 110, 40), dataset, D)
    print(f"  baseline: TP={base_metrics['tp']}  recall={base_metrics['recall']:.1f}%  "
          f"FP={base_metrics['fp']}  F1={base_metrics['f1']:.1f}", flush=True)

    # Define axes
    # Note: smin, vmin, denoise are passed directly to run_pipeline
    # ortho_tol, boundary_band, max_aspect, min_side, min_area go into base_p
    axes = {
        'smin':          ([60, 70, 80, 90, 100],  'hsv'),
        'vmin':          ([90, 100, 110, 120, 130], 'hsv'),
        'denoise_min':   ([20, 40, 60, 100, 160], 'hsv'),
        'min_area':      ([400, 700, 900, 1200, 2000], 'p'),
        'min_side':      ([10, 15, 20, 25, 30], 'p'),
        'max_aspect':    ([6, 8, 10, 12, 15, 20], 'p'),
        'ortho_tol':     ([10, 15, 20, 25], 'p'),
        'boundary_band': ([5, 8, 12, 20, 30], 'p'),
    }

    best_per_axis = {}
    all_results = {'baseline': base_metrics, 'axes': {}}

    for axis_name, (values, kind) in axes.items():
        print(f"\n[Axis: {axis_name}] values={values}", flush=True)
        t0 = time.time()
        if kind == 'hsv':
            # Sweep hsv/denoise param; other hsv params at baseline
            best_v, best_m, all_m = {}, {}, {}
            for v in values:
                if axis_name == 'smin':
                    smin = v
                    vmin = base_p.get('smin', 80) if False else 110
                elif axis_name == 'vmin':
                    smin = 80
                    vmin = v
                else:  # denoise
                    smin = 80
                    vmin = 110
                denoise_min = v if axis_name == 'denoise_min' else 40
                det_map = run_pipeline(dataset, base_p, smin, vmin, denoise_min)
                m = evaluate_det_map(det_map, dataset, D)
                all_m[v] = m
                print(f"    {axis_name}={v}: TP={m['tp']:>3d}  recall={m['recall']:>5.1f}%  "
                      f"FP={m['fp']:>3d}  F1={m['f1']:>5.1f}", flush=True)
            best_v = max(all_m.keys(), key=lambda x: (all_m[x]['tp'], -all_m[x]['fp']))
            best_m = all_m[best_v]
            print(f"    → BEST: {axis_name}={best_v} (TP={best_m['tp']}, FP={best_m['fp']})", flush=True)
            best_per_axis[axis_name] = best_v
            all_results['axes'][axis_name] = {str(k): v for k, v in all_m.items()}
        else:
            # Sweep p param
            best_v, best_m, all_m = {}, {}, {}
            for v in values:
                p2 = dict(base_p)
                p2[axis_name] = v
                det_map = run_pipeline(dataset, p2, 80, 110, 40)
                m = evaluate_det_map(det_map, dataset, D)
                all_m[v] = m
                print(f"    {axis_name}={v}: TP={m['tp']:>3d}  recall={m['recall']:>5.1f}%  "
                      f"FP={m['fp']:>3d}  F1={m['f1']:>5.1f}", flush=True)
            best_v = max(all_m.keys(), key=lambda x: (all_m[x]['tp'], -all_m[x]['fp']))
            best_m = all_m[best_v]
            print(f"    → BEST: {axis_name}={best_v} (TP={best_m['tp']}, FP={best_m['fp']})", flush=True)
            best_per_axis[axis_name] = best_v
            all_results['axes'][axis_name] = {str(k): v for k, v in all_m.items()}
        dt = time.time() - t0
        print(f"    [axis {axis_name} done in {dt:.1f}s]", flush=True)

    print(f"\n" + "=" * 78, flush=True)
    print(f"AXIS SWEEP SUMMARY (D={D}, {len(dataset)} imgs)", flush=True)
    print("=" * 78, flush=True)
    print(f"  baseline: TP={base_metrics['tp']}  FP={base_metrics['fp']}  "
          f"recall={base_metrics['recall']:.1f}%  F1={base_metrics['f1']:.1f}", flush=True)
    for axis_name in axes.keys():
        if axis_name in best_per_axis:
            v = best_per_axis[axis_name]
            m = all_results['axes'][axis_name][str(v)]
            print(f"  {axis_name}={v}: TP={m['tp']:>3d}  FP={m['fp']:>3d}  "
                  f"recall={m['recall']:.1f}%  F1={m['f1']:.1f}", flush=True)
    print("=" * 78, flush=True)

    # Step 2: Orthogonal combination
    if args.skip_combined:
        print(f"\n[Step 2] Skipped (--skip-combined)", flush=True)
    else:
        print(f"\n[Step 2] Orthogonal combination: applying all best-axis values together", flush=True)
        combined_p = dict(base_p)
        for k, v in best_per_axis.items():
            if k in ('smin', 'vmin', 'denoise_min'):
                continue  # hsv params passed separately
            combined_p[k] = v
        combined_smin = best_per_axis.get('smin', 80)
        combined_vmin = best_per_axis.get('vmin', 110)
        combined_denoise = best_per_axis.get('denoise_min', 40)
        print(f"  combined params: {dict(combined_p)}", flush=True)
        print(f"  combined HSV: smin={combined_smin} vmin={combined_vmin} denoise={combined_denoise}", flush=True)
        t0 = time.time()
        det_map_c = run_pipeline(dataset, combined_p, combined_smin, combined_vmin, combined_denoise)
        dt = time.time() - t0
        m_c = evaluate_det_map(det_map_c, dataset, D)
        print(f"  combined: TP={m_c['tp']}  FP={m_c['fp']}  recall={m_c['recall']:.1f}%  F1={m_c['f1']:.1f}  "
              f"[{dt:.1f}s]", flush=True)
        all_results['combined'] = {
            'best_per_axis': {k: (best_per_axis[k] if k in best_per_axis else None) for k in axes.keys()},
            'combined_p': dict(combined_p),
            'combined_hsv': {'smin': combined_smin, 'vmin': combined_vmin, 'denoise': combined_denoise},
            'metrics': m_c,
        }
        # Verdict
        delta_tp = m_c['tp'] - base_metrics['tp']
        delta_fp = m_c['fp'] - base_metrics['fp']
        if m_c['tp'] > base_metrics['tp'] and m_c['fp'] <= base_metrics['fp'] + 5:
            verdict = "WIN"
        elif m_c['tp'] >= base_metrics['tp'] and m_c['fp'] <= base_metrics['fp'] + 10:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"  VERDICT: {verdict} (ΔTP={delta_tp:+d}, ΔFP={delta_fp:+d})", flush=True)
        all_results['combined']['verdict'] = verdict

    # Save
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()