"""LOOP-23: Optimize fragment-joining on synthetic (corner-gap model).

LOOP-17 tested fragment-joining with 3 fixed configs (fjoin, fjoin_strict,
fjoin_loose) — all gave 0 new TP. But parameters were not swept systematically.

On the corner-gap model, each box yields 4 clean edges with 1-4 corner gaps.
At brokenness >= 0.05, the outline may split into 2-4 disconnected fragments.
Fragment-joining pairs parallel fragments to form candidate quads.

Axes swept (axis-by-axis from fjoin baseline):
  parallel_tol:    [5, 10, 15, 20, 25]       (current=10)
  max_thickness:   [4, 6, 8, 10, 15]          (current=8)
  min_length:      [10, 15, 20, 30, 40]       (current=15)
  max_distance:    [100, 150, 200, 300]       (current=200)
  max_aspect_extra:[1.0, 1.5, 2.0, 3.0]       (current=2.0)
  min_area:        [600, 900, 1000, 1200]     (admission filter interaction)

Usage: python proto/cv_scope23.py
"""
import sys, os, json, time, argparse, copy
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope16 import gen_dataset
from cv_scope17 import fragment_joining_quads
from eval_pointmatch import match_boxes_point

RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope23.json")
D = 5.0
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]

AXES = {
    'parallel_tol':    [5.0, 10.0, 15.0, 20.0, 25.0],
    'max_thickness':   [4.0, 6.0, 8.0, 10.0, 15.0],
    'min_length':      [10.0, 15.0, 20.0, 30.0, 40.0],
    'max_distance':    [100.0, 150.0, 200.0, 300.0],
    'max_aspect_extra': [1.0, 1.5, 2.0, 3.0],
    'min_area':        [600, 900, 1000, 1200],
}

BASE_FJOIN = {
    'parallel_tol': 10.0,
    'max_thickness': 8.0,
    'min_length': 15.0,
    'max_pairs': 300,
    'max_distance': 200.0,
    'max_aspect_extra': 2.0,
}


def run_fjoin(dataset, p, fcfg):
    """Run pm_v5_1 + fragment-joining."""
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
        # Fragment-joining
        fjoin_quads = fragment_joining_quads(mask, p, **fcfg)
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


def run_and_eval(dataset, p, fcfg, label, verbose=True):
    t0 = time.time()
    det = run_fjoin(dataset, p, fcfg)
    dt = time.time() - t0
    m = evaluate(det, dataset, D)
    t = m['total']
    if verbose:
        print(f"  {label:45s}: TP={t['tp']:4d} FP={t['fp']:4d} "
              f"recall={t['recall']:5.1f}% f1={t['f1']:5.1f}%  "
              f"({dt:.1f}s)", flush=True)
    return m, t


def main():
    p_base = pm5_params()
    p_base['min_area'] = 1000  # pm_v5_1

    print(f"LOOP-23: Optimize fragment-joining on synthetic (corner-gap, D={D})", flush=True)
    print(f"  pm_v5_1 baseline: min_area={p_base['min_area']:.0f}", flush=True)
    print(f"  fjoin baseline: {BASE_FJOIN}", flush=True)

    print(f"\nGenerating dataset...", flush=True)
    t0 = time.time()
    dataset = gen_dataset()
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)

    # Baseline: pm_v5_1 without fragment-joining
    print(f"\n[Baseline] pm_v5_1 (no fjoin)...", flush=True)
    t0 = time.time()
    det_base = {}
    for item in dataset:
        img = item['img']
        H, W = img.shape[:2]
        mask = red_mask(img)
        if p_base['denoise_min_area'] > 0:
            mask = denoise(mask, p_base['denoise_min_area'])
        cnts = outer_contours(mask, p_base)
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p_base)
            for m in HY_ORDER:
                q = _pass(base.get(m), W, H, p_base)
                if q is not None:
                    hy_quads.append(q)
        try:
            hy = dedup_keep_order(hy_quads, p_base['dedup_iou'])
        except cv2.error:
            hy = dedup_keep_order([q for q in hy_quads if q is not None and len(q) == 4],
                                  p_base['dedup_iou'])
        det_base[item['iid']] = hy
    dt = time.time() - t0
    m_base = evaluate(det_base, dataset, D)
    t_base = m_base['total']
    print(f"  pm_v5_1 (no fjoin)               : TP={t_base['tp']:4d} FP={t_base['fp']:4d} "
          f"f1={t_base['f1']:5.1f}%  ({dt:.1f}s)", flush=True)

    # fjoin baseline
    print(f"\n[fjoin baseline]...", flush=True)
    m_fj, t_fj = run_and_eval(dataset, p_base, BASE_FJOIN, "fjoin baseline")

    all_results = {
        'baseline_nofjoin': t_base,
        'baseline_fjoin': t_fj,
        'axes': {},
        'best_per_axis': {},
        'combined': {},
        'n_images': len(dataset),
        'dist_thr': D,
    }

    # Axis-by-axis sweep
    best_cfg = dict(BASE_FJOIN)
    best_p = dict(p_base)

    for axis_name, values in AXES.items():
        print(f"\n[Axis: {axis_name}] values={values}", flush=True)
        axis_results = {}
        if axis_name == 'min_area':
            current_val = p_base['min_area']
        else:
            current_val = best_cfg[axis_name]
        best_val = current_val
        best_f1 = t_fj['f1']
        best_tp = t_fj['tp']
        for val in values:
            if axis_name == 'min_area':
                p_test = dict(best_p)
                p_test['min_area'] = val
                fcfg_test = dict(best_cfg)
            else:
                p_test = dict(best_p)
                fcfg_test = dict(best_cfg)
                fcfg_test[axis_name] = val
            label = f"{axis_name}={val}"
            m, t = run_and_eval(dataset, p_test, fcfg_test, label)
            axis_results[str(val)] = t
            if t['f1'] > best_f1 or (t['f1'] == best_f1 and t['tp'] > best_tp):
                best_f1 = t['f1']
                best_tp = t['tp']
                best_val = val
        all_results['axes'][axis_name] = axis_results
        all_results['best_per_axis'][axis_name] = {
            'best': best_val, 'f1': best_f1, 'tp': best_tp,
            'baseline': current_val,
        }
        delta_f1 = best_f1 - t_fj['f1']
        delta_tp = best_tp - t_fj['tp']
        print(f"  -> BEST: {axis_name}={best_val}  "
              f"f1={best_f1:.1f}% (Δf1={delta_f1:+.1f})  "
              f"TP={best_tp} (ΔTP={delta_tp:+d})", flush=True)
        if axis_name == 'min_area':
            best_p['min_area'] = best_val
        else:
            best_cfg[axis_name] = best_val

    # Combined
    print(f"\n[Combined] best_cfg={best_cfg}  min_area={best_p['min_area']}", flush=True)
    m_comb, t_comb = run_and_eval(dataset, best_p, best_cfg, "combined")
    all_results['combined'] = {
        'fcfg': best_cfg,
        'min_area': best_p['min_area'],
        'metrics': m_comb,
    }

    # Verdict vs no-fjoin baseline
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
    print(f"{'Method':>45s}  {'TP':>5s}  {'FP':>5s}  {'f1':>6s}  {'ΔTP':>5s}  {'Δf1':>5s}  V", flush=True)
    print(f"{'pm_v5_1 (no fjoin)':>45s}  {t_base['tp']:>5d}  {t_base['fp']:>5d}  "
          f"{t_base['f1']:>5.1f}%  {'--':>5s}  {'--':>5s}  base", flush=True)
    print(f"{'fjoin baseline':>45s}  {t_fj['tp']:>5d}  {t_fj['fp']:>5d}  "
          f"{t_fj['f1']:>5.1f}%  {t_fj['tp']-t_base['tp']:+5d}  {t_fj['f1']-t_base['f1']:+5.1f}  -", flush=True)
    print(f"{'combined':>45s}  {t_comb['tp']:>5d}  {t_comb['fp']:>5d}  "
          f"{t_comb['f1']:>5.1f}%  {delta_tp:+5d}  {delta_f1:+5.1f}  {verdict}", flush=True)
    print(f"{'='*78}", flush=True)

    print(f"\nPer-brokenness (no-fjoin vs combined):", flush=True)
    for b in BROKENNESS_VALUES:
        bm = m_base[str(b)]
        cm = m_comb[str(b)]
        print(f"  b={b:.2f}: nofjoin TP={bm['tp']:>3d}/{bm['n_gt']:>3d} ({bm['recall']:5.1f}%)  "
              f"comb TP={cm['tp']:>3d}/{cm['n_gt']:>3d} ({cm['recall']:5.1f}%)  "
              f"ΔTP={cm['tp']-bm['tp']:+d}", flush=True)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()