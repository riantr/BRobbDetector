"""LOOP-21: Optimize pm_v5_1 parameters on synthetic (corner-gap model).

Algorithm: pm_v5_1 = hybrid(hull->edgefit->mar) + admission filter + min_area=1000.
This is the current frozen baseline (real: TP=502, corr_f1=71.5; synth: f1=46.8%).

Methodology: axis-by-axis sweep (one dimension at a time).
  1. Start from pm_v5_1 baseline.
  2. Sweep each axis independently, pick best value (max f1, tie-break max TP).
  3. Combine all best values -> final candidate.
  4. Compare final candidate vs baseline.

Axes swept:
  min_area:       [600, 800, 900, 1000, 1200, 1500]   (current=1000)
  min_side:       [8, 12, 15, 20, 25]                  (current=15)
  max_aspect:     [8, 10, 12, 15, 20]                   (current=12)
  ortho_tol:      [8, 12, 15, 20, 25]                   (current=15)
  boundary_band:  [6, 10, 12, 16, 20]                   (current=12)
  denoise_min_area: [0, 20, 40, 60, 80]                 (current=40)
  fit_band:       [3, 5, 6, 8, 10]                      (current=6)

Usage: python proto/cv_scope21.py
"""
import sys, os, json, time, glob, argparse, copy
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import obb_corners, draw_obb, compute_gt_minarearect
from cv_scope16 import gen_dataset
from eval_pointmatch import match_boxes_point

RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope21.json")
D = 5.0
N_BG = 50
N_PER_VALUE = 4
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 42
DEFAULT_OBB = {'box_w': 80, 'box_h': 50, 'angle': 15, 'stroke_w': 2}

AXES = {
    'min_area':         [600, 800, 900, 1000, 1200, 1500],
    'min_side':         [8, 12, 15, 20, 25],
    'max_aspect':       [8, 10, 12, 15, 20],
    'ortho_tol':        [8, 12, 15, 20, 25],
    'boundary_band':    [6, 10, 12, 16, 20],
    'denoise_min_area': [0, 20, 40, 60, 80],
    'fit_band':         [3, 5, 6, 8, 10],
}


def run_pm_v5_1(dataset, p):
    """Run pm_v5_1 hybrid pipeline, return det_map = {iid: [quads]}."""
    det_map = {}
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
        try:
            hy = dedup_keep_order(hy_quads, p['dedup_iou'])
        except cv2.error:
            hy = dedup_keep_order(
                [q for q in hy_quads if q is not None and len(q) == 4],
                p['dedup_iou'],
            )
        det_map[item['iid']] = hy
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
        for k in ['tp', 'fn', 'fp', 'n_gt', 'n_det']:
            metrics[b][k] += 0
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


def run_and_eval(dataset, p, label, verbose=True):
    t0 = time.time()
    det = run_pm_v5_1(dataset, p)
    dt = time.time() - t0
    m = evaluate(det, dataset, D)
    t = m['total']
    if verbose:
        print(f"  {label:30s}: TP={t['tp']:4d} FP={t['fp']:4d} "
              f"recall={t['recall']:5.1f}% f1={t['f1']:5.1f}%  "
              f"({dt:.1f}s)", flush=True)
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-build', action='store_true')
    args = ap.parse_args()

    p_base = pm5_params()
    p_base['min_area'] = 1000  # pm_v5_1

    print(f"LOOP-21: Optimize pm_v5_1 on synthetic (corner-gap, D={D})", flush=True)
    print(f"  baseline: min_area={p_base['min_area']:.0f} min_side={p_base['min_side']:.0f} "
          f"max_aspect={p_base['max_aspect']:.0f} ortho_tol={p_base['ortho_tol']:.0f} "
          f"bnd={p_base['boundary_band']:.0f} denoise={p_base['denoise_min_area']:.0f} "
          f"fit_band={p_base['fit_band']:.0f}", flush=True)

    print(f"\nGenerating dataset (n_bg={N_BG}, n_per_value={N_PER_VALUE})...", flush=True)
    t0 = time.time()
    dataset = gen_dataset()
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)

    # Baseline
    print(f"\n[Baseline] pm_v5_1...", flush=True)
    m_base, t_base = run_and_eval(dataset, p_base, "pm_v5_1 baseline")

    all_results = {
        'baseline_params': {k: v for k, v in p_base.items()
                            if not isinstance(v, (list, tuple))},
        'baseline_metrics': m_base,
        'axes': {},
        'best_per_axis': {},
        'combined': {},
        'n_images': len(dataset),
        'dist_thr': D,
    }

    # Axis-by-axis sweep
    best_params = dict(p_base)
    for axis_name, values in AXES.items():
        print(f"\n[Axis: {axis_name}] values={values}", flush=True)
        axis_results = {}
        best_val = p_base[axis_name]
        best_f1 = t_base['f1']
        best_tp = t_base['tp']
        for val in values:
            p_test = dict(best_params)
            p_test[axis_name] = val
            label = f"{axis_name}={val}"
            m, t = run_and_eval(dataset, p_test, label)
            axis_results[str(val)] = t
            # Pick best: max f1, tie-break max TP
            if t['f1'] > best_f1 or (t['f1'] == best_f1 and t['tp'] > best_tp):
                best_f1 = t['f1']
                best_tp = t['tp']
                best_val = val
        all_results['axes'][axis_name] = axis_results
        all_results['best_per_axis'][axis_name] = {
            'best_val': best_val,
            'best_f1': best_f1,
            'best_tp': best_tp,
            'baseline_val': p_base[axis_name],
            'baseline_f1': t_base['f1'],
            'baseline_tp': t_base['tp'],
        }
        delta_f1 = best_f1 - t_base['f1']
        delta_tp = best_tp - t_base['tp']
        print(f"  -> BEST: {axis_name}={best_val}  "
              f"f1={best_f1:.1f}% (Δf1={delta_f1:+.1f})  "
              f"TP={best_tp} (ΔTP={delta_tp:+d})", flush=True)
        # Update best_params for next axis (greedy)
        best_params[axis_name] = best_val

    # Combined candidate
    print(f"\n[Combined] all best values...", flush=True)
    print(f"  params: " + ", ".join(f"{k}={best_params[k]}" for k in AXES), flush=True)
    m_comb, t_comb = run_and_eval(dataset, best_params, "combined")
    all_results['combined'] = {
        'params': {k: best_params[k] for k in AXES},
        'metrics': m_comb,
    }

    delta_f1 = t_comb['f1'] - t_base['f1']
    delta_tp = t_comb['tp'] - t_base['tp']
    if t_comb['f1'] > t_base['f1']:
        verdict = "WIN"
    elif t_comb['f1'] == t_base['f1']:
        verdict = "TIE"
    else:
        verdict = "DEAD"
    print(f"\n{'='*78}", flush=True)
    print(f"SUMMARY (D={D}, {len(dataset)} synth images)", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'Method':>30s}  {'TP':>5s}  {'FP':>5s}  {'f1':>6s}  {'ΔTP':>5s}  {'Δf1':>5s}  Verdict", flush=True)
    print(f"{'pm_v5_1 baseline':>30s}  {t_base['tp']:>5d}  {t_base['fp']:>5d}  "
          f"{t_base['f1']:>5.1f}%  {'--':>5s}  {'--':>5s}  baseline", flush=True)
    print(f"{'combined':>30s}  {t_comb['tp']:>5d}  {t_comb['fp']:>5d}  "
          f"{t_comb['f1']:>5.1f}%  {delta_tp:+5d}  {delta_f1:+5.1f}  {verdict}", flush=True)
    print(f"{'='*78}", flush=True)

    # Per-brokenness for combined
    print(f"\nPer-brokenness (baseline vs combined):", flush=True)
    for b in BROKENNESS_VALUES:
        bm = m_base[str(b)]
        cm = m_comb[str(b)]
        print(f"  b={b:.2f}: base TP={bm['tp']:>3d}/{bm['n_gt']:>3d} ({bm['recall']:5.1f}%)  "
              f"comb TP={cm['tp']:>3d}/{cm['n_gt']:>3d} ({cm['recall']:5.1f}%)  "
              f"ΔTP={cm['tp']-bm['tp']:+d}", flush=True)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()