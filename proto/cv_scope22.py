"""LOOP-22: Optimize morphological close on synthetic (corner-gap model).

LOOP-16 tested close_3/5/7 on the OLD per-edge model (baseline TP=131).
Now re-test on the NEW corner-gap model (baseline TP=383) and sweep more axes.

Hypothesis: morphological close bridges broken-outline gaps before findContours,
recovering detections at brokenness >= 0.05. On the corner-gap model, each box
yields 4 clean edges with 1-4 corner gaps, so close may bridge them more
effectively than on the per-edge model.

Axes swept:
  kernel_size:  [2, 3, 4, 5, 7, 9]       (LOOP-16 tested 3, 5, 7)
  kernel_shape: [RECT, ELLIPSE, CROSS]    (LOOP-16 only RECT)
  iterations:   [1, 2]                    (LOOP-16 only 1)
  min_area:     [600, 900, 1000, 1200]    (interaction with close)

Methodology: axis-by-axis sweep from close_3 baseline, then combine best.

Usage: python proto/cv_scope22.py
"""
import sys, os, json, time, argparse, copy
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope16 import gen_dataset, apply_preprocess
from eval_pointmatch import match_boxes_point

RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope22.json")
D = 5.0
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]

SHAPE_MAP = {
    'RECT': cv2.MORPH_RECT,
    'ELLIPSE': cv2.MORPH_ELLIPSE,
    'CROSS': cv2.MORPH_CROSS,
}


def apply_close(mask, ksize, shape='RECT', iterations=1):
    kernel = cv2.getStructuringElement(SHAPE_MAP[shape], (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def run_close(dataset, p, ksize, shape, iterations):
    det_map = {}
    for item in dataset:
        img = item['img']
        H, W = img.shape[:2]
        mask = red_mask(img)
        mask = apply_close(mask, ksize, shape, iterations)
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


def run_and_eval(dataset, p, ksize, shape, iterations, label, verbose=True):
    t0 = time.time()
    det = run_close(dataset, p, ksize, shape, iterations)
    dt = time.time() - t0
    m = evaluate(det, dataset, D)
    t = m['total']
    if verbose:
        print(f"  {label:40s}: TP={t['tp']:4d} FP={t['fp']:4d} "
              f"recall={t['recall']:5.1f}% f1={t['f1']:5.1f}%  "
              f"({dt:.1f}s)", flush=True)
    return m, t


def main():
    p_base = pm5_params()
    p_base['min_area'] = 1000  # pm_v5_1

    # Baseline config: close_3 RECT 1 iter, min_area=1000
    base_cfg = {'ksize': 3, 'shape': 'RECT', 'iterations': 1, 'min_area': 1000}

    print(f"LOOP-22: Optimize morphological close on synthetic (corner-gap, D={D})", flush=True)
    print(f"  pm_v5_1 baseline (no close): min_area={p_base['min_area']:.0f}", flush=True)
    print(f"  close baseline: ksize={base_cfg['ksize']} shape={base_cfg['shape']} "
          f"iter={base_cfg['iterations']} min_area={base_cfg['min_area']}", flush=True)

    print(f"\nGenerating dataset...", flush=True)
    t0 = time.time()
    dataset = gen_dataset()
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)

    # No-close baseline (pm_v5_1)
    print(f"\n[Baseline] pm_v5_1 (no close)...", flush=True)
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
    m_noclose = evaluate(det_base, dataset, D)
    t_noclose = m_noclose['total']
    print(f"  pm_v5_1 (no close)            : TP={t_noclose['tp']:4d} FP={t_noclose['fp']:4d} "
          f"f1={t_noclose['f1']:5.1f}%  ({dt:.1f}s)", flush=True)

    # close_3 baseline
    print(f"\n[close_3 baseline]...", flush=True)
    m_c3, t_c3 = run_and_eval(dataset, p_base, 3, 'RECT', 1, "close_3 RECT iter=1")

    all_results = {
        'baseline_noclose': t_noclose,
        'baseline_close3': t_c3,
        'axes': {},
        'best_per_axis': {},
        'combined': {},
        'n_images': len(dataset),
        'dist_thr': D,
    }

    # Axis-by-axis sweep from close_3 baseline
    best_cfg = dict(base_cfg)

    # Axis 1: kernel_size
    print(f"\n[Axis: kernel_size] values=[2,3,4,5,7,9]", flush=True)
    axis_results = {}
    best_val, best_f1, best_tp = 3, t_c3['f1'], t_c3['tp']
    for k in [2, 3, 4, 5, 7, 9]:
        p_test = dict(p_base)
        p_test['min_area'] = best_cfg['min_area']
        m, t = run_and_eval(dataset, p_test, k, best_cfg['shape'],
                            best_cfg['iterations'], f"ksize={k}")
        axis_results[str(k)] = t
        if t['f1'] > best_f1 or (t['f1'] == best_f1 and t['tp'] > best_tp):
            best_f1, best_tp, best_val = t['f1'], t['tp'], k
    all_results['axes']['kernel_size'] = axis_results
    all_results['best_per_axis']['kernel_size'] = {'best': best_val, 'f1': best_f1, 'tp': best_tp}
    print(f"  -> BEST: ksize={best_val} f1={best_f1:.1f}% TP={best_tp}", flush=True)
    best_cfg['ksize'] = best_val

    # Axis 2: kernel_shape
    print(f"\n[Axis: kernel_shape] values=[RECT,ELLIPSE,CROSS]", flush=True)
    axis_results = {}
    best_val_s, best_f1, best_tp = best_cfg['shape'], t_c3['f1'], t_c3['tp']
    for s in ['RECT', 'ELLIPSE', 'CROSS']:
        p_test = dict(p_base)
        p_test['min_area'] = best_cfg['min_area']
        m, t = run_and_eval(dataset, p_test, best_cfg['ksize'], s,
                            best_cfg['iterations'], f"shape={s}")
        axis_results[s] = t
        if t['f1'] > best_f1 or (t['f1'] == best_f1 and t['tp'] > best_tp):
            best_f1, best_tp, best_val_s = t['f1'], t['tp'], s
    all_results['axes']['kernel_shape'] = axis_results
    all_results['best_per_axis']['kernel_shape'] = {'best': best_val_s, 'f1': best_f1, 'tp': best_tp}
    print(f"  -> BEST: shape={best_val_s} f1={best_f1:.1f}% TP={best_tp}", flush=True)
    best_cfg['shape'] = best_val_s

    # Axis 3: iterations
    print(f"\n[Axis: iterations] values=[1,2]", flush=True)
    axis_results = {}
    best_val_i, best_f1, best_tp = best_cfg['iterations'], t_c3['f1'], t_c3['tp']
    for it in [1, 2]:
        p_test = dict(p_base)
        p_test['min_area'] = best_cfg['min_area']
        m, t = run_and_eval(dataset, p_test, best_cfg['ksize'], best_cfg['shape'],
                            it, f"iter={it}")
        axis_results[str(it)] = t
        if t['f1'] > best_f1 or (t['f1'] == best_f1 and t['tp'] > best_tp):
            best_f1, best_tp, best_val_i = t['f1'], t['tp'], it
    all_results['axes']['iterations'] = axis_results
    all_results['best_per_axis']['iterations'] = {'best': best_val_i, 'f1': best_f1, 'tp': best_tp}
    print(f"  -> BEST: iter={best_val_i} f1={best_f1:.1f}% TP={best_tp}", flush=True)
    best_cfg['iterations'] = best_val_i

    # Axis 4: min_area interaction
    print(f"\n[Axis: min_area] values=[600,900,1000,1200]", flush=True)
    axis_results = {}
    best_val_ma, best_f1, best_tp = best_cfg['min_area'], t_c3['f1'], t_c3['tp']
    for ma in [600, 900, 1000, 1200]:
        p_test = dict(p_base)
        p_test['min_area'] = ma
        m, t = run_and_eval(dataset, p_test, best_cfg['ksize'], best_cfg['shape'],
                            best_cfg['iterations'], f"min_area={ma}")
        axis_results[str(ma)] = t
        if t['f1'] > best_f1 or (t['f1'] == best_f1 and t['tp'] > best_tp):
            best_f1, best_tp, best_val_ma = t['f1'], t['tp'], ma
    all_results['axes']['min_area'] = axis_results
    all_results['best_per_axis']['min_area'] = {'best': best_val_ma, 'f1': best_f1, 'tp': best_tp}
    print(f"  -> BEST: min_area={best_val_ma} f1={best_f1:.1f}% TP={best_tp}", flush=True)
    best_cfg['min_area'] = best_val_ma

    # Combined
    print(f"\n[Combined] best: ksize={best_cfg['ksize']} shape={best_cfg['shape']} "
          f"iter={best_cfg['iterations']} min_area={best_cfg['min_area']}", flush=True)
    p_comb = dict(p_base)
    p_comb['min_area'] = best_cfg['min_area']
    m_comb, t_comb = run_and_eval(dataset, p_comb, best_cfg['ksize'],
                                  best_cfg['shape'], best_cfg['iterations'], "combined")
    all_results['combined'] = {'cfg': best_cfg, 'metrics': m_comb}

    # Verdict vs no-close baseline
    delta_f1 = t_comb['f1'] - t_noclose['f1']
    delta_tp = t_comb['tp'] - t_noclose['tp']
    if t_comb['f1'] > t_noclose['f1']:
        verdict = "WIN"
    elif t_comb['f1'] == t_noclose['f1']:
        verdict = "TIE"
    else:
        verdict = "DEAD"

    print(f"\n{'='*78}", flush=True)
    print(f"SUMMARY (D={D}, {len(dataset)} synth images)", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'Method':>40s}  {'TP':>5s}  {'FP':>5s}  {'f1':>6s}  {'ΔTP':>5s}  {'Δf1':>5s}  V", flush=True)
    print(f"{'pm_v5_1 (no close)':>40s}  {t_noclose['tp']:>5d}  {t_noclose['fp']:>5d}  "
          f"{t_noclose['f1']:>5.1f}%  {'--':>5s}  {'--':>5s}  base", flush=True)
    print(f"{'close_3 baseline':>40s}  {t_c3['tp']:>5d}  {t_c3['fp']:>5d}  "
          f"{t_c3['f1']:>5.1f}%  {t_c3['tp']-t_noclose['tp']:+5d}  {t_c3['f1']-t_noclose['f1']:+5.1f}  -", flush=True)
    print(f"{'combined':>40s}  {t_comb['tp']:>5d}  {t_comb['fp']:>5d}  "
          f"{t_comb['f1']:>5.1f}%  {delta_tp:+5d}  {delta_f1:+5.1f}  {verdict}", flush=True)
    print(f"{'='*78}", flush=True)

    print(f"\nPer-brokenness (no-close vs combined):", flush=True)
    for b in BROKENNESS_VALUES:
        bm = m_noclose[str(b)]
        cm = m_comb[str(b)]
        print(f"  b={b:.2f}: noclose TP={bm['tp']:>3d}/{bm['n_gt']:>3d} ({bm['recall']:5.1f}%)  "
              f"comb TP={cm['tp']:>3d}/{cm['n_gt']:>3d} ({cm['recall']:5.1f}%)  "
              f"ΔTP={cm['tp']-bm['tp']:+d}", flush=True)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()