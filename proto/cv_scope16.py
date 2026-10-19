"""LOOP-16: Mask pre-processing — morphological close/dilate to bridge broken outlines.

Hypothesis (from LOOP-14 visualization):
  At brokenness >= 0.05, the outline splits into 2-4 disconnected fragments,
  each below min_area=900. outer_contours returns nothing -> 0% recall.

Test: apply morphological close/dilate to the red mask BEFORE findContours
to bridge small gaps. Variants:
  baseline:   no extra pre-processing
  close_3:    cv2.morphologyEx(mask, MORPH_CLOSE, 3x3)
  close_5:    cv2.morphologyEx(mask, MORPH_CLOSE, 5x5)
  close_7:    cv2.morphologyEx(mask, MORPH_CLOSE, 7x7)
  dilate_3:   cv2.dilate(mask, 3x3)
  dilate_5:   cv2.dilate(mask, 5x5)
  dilate_7:   cv2.dilate(mask, 7x7)
  open_close: cv2.morphologyEx(mask, MORPH_OPEN, 3x3) then MORPH_CLOSE 5x5

Validate on the synthetic X-ray benchmark (1000 imgs, 5 brokenness levels).
Any WIN must also pass the 500-image real benchmark (see cv_scope16_real.py).

Usage: python proto/cv_scope16.py [--variants VARIANTS]
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
from cv_scope15 import red_mask_with_hsv
from eval_pointmatch import match_boxes_point

BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
DET_DIR = os.path.join(ROOT, "tmp_sc16_det")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope16.json")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]
N_BG = 50
N_PER_VALUE = 4
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 42
DEFAULT_OBB = {'box_w': 80, 'box_h': 50, 'angle': 15, 'stroke_w': 2}


def get_bg_files_sorted():
    """Use sorted order to match the eval_synth_benchmark baseline."""
    return sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))[:N_BG]


def gen_dataset():
    """Build the same 1000-image synthetic dataset as eval_synth_benchmark."""
    bg_files = get_bg_files_sorted()
    rng = np.random.RandomState(SEED)
    dataset = []
    for bi, bg_file in enumerate(bg_files):
        bg = cv2.imread(bg_file)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        for brokenness in BROKENNESS_VALUES:
            for k in range(N_PER_VALUE):
                cx = int(rng.randint(80, max(81, W - 80)))
                cy = int(rng.randint(60, max(61, H - 60)))
                bp = dict(DEFAULT_OBB)
                bp['cx'] = cx
                bp['cy'] = cy
                bp['brokenness'] = brokenness
                img = bg.copy()
                corners = obb_corners(cx, cy, bp['box_w'], bp['box_h'], bp['angle'])
                draw_obb(img, corners, brokenness=brokenness,
                         stroke_w=bp['stroke_w'], color=(0, 0, 255), rng=rng,
                         brokenness_mode='corner')
                gt = compute_gt_minarearect(img, corners, padding=10)
                iid = f"synth_{bi:03d}_b{brokenness:.2f}_{k}"
                dataset.append({
                    'iid': iid, 'img': img, 'gt_quads': [gt],
                    'bp': bp, 'brokenness': brokenness,
                })
    return dataset


def apply_preprocess(mask, variant):
    """Apply morphological pre-processing to the red mask."""
    if variant == 'baseline':
        return mask
    elif variant.startswith('close_'):
        k = int(variant.split('_')[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif variant.startswith('dilate_'):
        k = int(variant.split('_')[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.dilate(mask, kernel)
    elif variant == 'open_close':
        k1, k2 = 3, 5
        k1_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))
        k2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1_kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k2_kernel)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_variant(name, dataset, p, variant):
    """Run pm_v5 with custom mask pre-processing."""
    det_map = {}
    total = 0
    t0 = time.time()
    for item in dataset:
        img = item['img']
        H, W = img.shape[:2]
        mask = red_mask(img)
        mask = apply_preprocess(mask, variant)
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
        total += len(hy)
    dt = time.time() - t0
    print(f"  {name}: {total} boxes in {dt:.1f}s "
          f"({total / len(dataset):.2f} dets/img)", flush=True)
    return det_map


def evaluate(det_map, dataset, dist_thr=5.0):
    """Per-brokenness + total metrics."""
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
        summary[str(k)] = {
            **m,
            'recall': round(recall, 2),
            'precision': round(prec, 2),
            'f1': round(f1, 2),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', type=str,
                    default='baseline,close_3,close_5,close_7,dilate_3,dilate_5,dilate_7,open_close',
                    help='Comma-separated variants to test')
    args = ap.parse_args()

    p = pm5_params()
    print(f"LOOP-16: Mask pre-processing (morphological close/dilate) on synth X-ray (D={D})", flush=True)
    print(f"  pm5 baseline: min_area={p['min_area']:.0f} denoise={p['denoise_min_area']:.0f}", flush=True)

    print(f"\nGenerating dataset (n_bg={N_BG}, n_per_value={N_PER_VALUE}, brokenness={BROKENNESS_VALUES})...", flush=True)
    t0 = time.time()
    dataset = gen_dataset()
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)

    variants = args.variants.split(',')
    all_results = {'pm5_params': {k: v for k, v in p.items()
                                   if not isinstance(v, (list, tuple))},
                   'brokenness_values': BROKENNESS_VALUES,
                   'n_images': len(dataset),
                   'dist_thr': D,
                   'methods': {}}

    # Run baseline first
    print(f"\n[Baseline] pm_v5 with no extra pre-processing...", flush=True)
    t0 = time.time()
    det_base = run_variant('baseline', dataset, p, 'baseline')
    dt = time.time() - t0
    m_base = evaluate(det_base, dataset, D)
    all_results['methods']['baseline'] = m_base
    bl = m_base['total']
    print(f"  baseline: TP={bl['tp']} FP={bl['fp']} FN={bl['fn']}  "
          f"recall={bl['recall']:.1f}% prec={bl['precision']:.1f}% f1={bl['f1']:.1f}%", flush=True)
    for b in BROKENNESS_VALUES:
        bm = m_base[str(b)]
        print(f"    b={b:.2f}: TP={bm['tp']:>3d}/{bm['n_gt']:>3d}  recall={bm['recall']:>5.1f}%  "
              f"det={bm['n_det']}", flush=True)

    # Run each variant
    for v in variants:
        if v == 'baseline':
            continue
        print(f"\n[Variant: {v}] mask pre-processing...", flush=True)
        t0 = time.time()
        det_v = run_variant(v, dataset, p, v)
        dt = time.time() - t0
        m_v = evaluate(det_v, dataset, D)
        all_results['methods'][v] = m_v
        v_total = m_v['total']
        delta_tp = v_total['tp'] - bl['tp']
        delta_fp = v_total['fp'] - bl['fp']
        delta_f1 = v_total['f1'] - bl['f1']
        # Verdict
        if v_total['tp'] > bl['tp'] and v_total['fp'] <= bl['fp'] + 5:
            verdict = "WIN"
        elif v_total['tp'] >= bl['tp'] and v_total['fp'] <= bl['fp'] + 10:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"  {v}: TP={v_total['tp']} FP={v_total['fp']} FN={v_total['fn']}  "
              f"recall={v_total['recall']:.1f}% f1={v_total['f1']:.1f}%  "
              f"(ΔTP={delta_tp:+d} ΔFP={delta_fp:+d} Δf1={delta_f1:+.1f}) [{verdict}]", flush=True)
        for b in BROKENNESS_VALUES:
            bm = m_v[str(b)]
            bbm = m_base[str(b)]
            dtp = bm['tp'] - bbm['tp']
            print(f"    b={b:.2f}: TP={bm['tp']:>3d}/{bm['n_gt']:>3d}  recall={bm['recall']:>5.1f}%  "
                  f"det={bm['n_det']}  (ΔTP={dtp:+d})", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, {len(dataset)} synth images, pm_v5 baseline)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':>12s}  {'TP':>5s}  {'FP':>5s}  {'recall':>6s}  "
          f"{'f1':>6s}  {'ΔTP':>5s}  {'ΔFP':>5s}  {'Δf1':>5s}  Verdict", flush=True)
    print(f"{'pm_v5 base':>12s}  {bl['tp']:>5d}  {bl['fp']:>5d}  "
          f"{bl['recall']:>5.1f}%  {bl['f1']:>5.1f}%  {'--':>5s}  {'--':>5s}  {'--':>5s}  baseline", flush=True)
    for v in variants:
        if v == 'baseline':
            continue
        m = all_results['methods'][v]['total']
        dtp = m['tp'] - bl['tp']
        dfp = m['fp'] - bl['fp']
        df1 = m['f1'] - bl['f1']
        # Same verdict logic
        if m['tp'] > bl['tp'] and m['fp'] <= bl['fp'] + 5:
            verdict = "WIN"
        elif m['tp'] >= bl['tp'] and m['fp'] <= bl['fp'] + 10:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"{v:>12s}  {m['tp']:>5d}  {m['fp']:>5d}  "
              f"{m['recall']:>5.1f}%  {m['f1']:>5.1f}%  "
              f"{dtp:+5d}  {dfp:+5d}  {df1:+5.1f}  {verdict}", flush=True)
    print("=" * 78, flush=True)

    # Save
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()