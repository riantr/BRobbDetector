"""LOOP-14: Realistic synthetic data using X-ray backgrounds.

Builds synthetic OBBs ON TOP of real X-ray images (from
data\\Image@Xray_CSHH_Cargo_TF_P01_A), then runs pm_v5 to measure
performance in a realistic noise/texture setting.

Key changes from LOOP-13:
  - Real X-ray backgrounds (instead of clean dark gray + Gaussian noise)
  - Existing red content in the background (real distribution of red objects)
  - More realistic test of pm_v5's performance

GT strategy: per request, GT = minAreaRect of the rendered red pixels
near the injected OBB (NOT analytical). This matches real-data annotation
philosophy: the annotator fits a quad to the visible outline.

Variants tested:
  - pm_v5 baseline (min_area=900)
  - Targeted: lower min_area (200, 100, 50)

Usage: python proto/cv_scope14.py [--n-images N] [--n-bg BG]
"""
import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise
from cv_hybrid import outer_contours, _pass, dedup_keep_order
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import (obb_corners, draw_obb, compute_gt_minarearect)
from cv_hybrid import ROOT
from eval_pointmatch import match_boxes_point

BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
OUT_ROOT = os.path.join(ROOT, "tmp_sc14")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope14.json")
D = 5.0


def list_bg_images(n_bg=50, seed=42):
    """Pick n_bg random X-ray images."""
    files = sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))
    rng = np.random.RandomState(seed)
    if n_bg >= len(files):
        return files
    idxs = rng.choice(len(files), n_bg, replace=False)
    return [files[i] for i in idxs]


def inject_obb(bg, bp, rng):
    """Inject a synthetic OBB on top of bg image. Returns (img, gt).

    bp: {cx, cy, box_w, box_h, angle, brokenness, stroke_w}
    gt: minAreaRect of red pixels in the box region
    """
    img = bg.copy()
    corners = obb_corners(bp['cx'], bp['cy'], bp['box_w'], bp['box_h'], bp['angle'])
    draw_obb(img, corners, brokenness=bp.get('brokenness', 0),
             stroke_w=bp.get('stroke_w', 2), color=(0, 0, 255), rng=rng,
             brokenness_mode=bp.get('brokenness_mode', 'corner'))
    gt = compute_gt_minarearect(img, corners, padding=10)
    return img, gt


def gen_dataset(n_bg, brokenness_values, n_per_value, seed=42):
    """Generate dataset: n_per_value images per brokenness value, using n_bg backgrounds.

    Each image has 1 injected OBB at random position with the given brokenness.
    """
    bg_files = list_bg_images(n_bg, seed=seed)
    rng = np.random.RandomState(seed)

    # Default OBB params (typical real-data box)
    default = {
        'box_w': 80, 'box_h': 50, 'angle': 15,
        'stroke_w': 2, 'brokenness': 0,
    }

    dataset = []
    for b in brokenness_values:
        for k in range(n_per_value):
            # Pick a random background
            bg_file = bg_files[k % len(bg_files)]
            bg = cv2.imread(bg_file)
            if bg is None:
                continue
            H, W = bg.shape[:2]
            # Random OBB position
            cx = rng.randint(80, max(81, W - 80))
            cy = rng.randint(60, max(61, H - 60))

            bp = dict(default)
            bp['cx'] = cx
            bp['cy'] = cy
            bp['brokenness'] = b

            img, gt = inject_obb(bg, bp, rng)
            dataset.append({
                'img': img, 'gt_quads': [gt],
                'bp': bp, 'bg_file': os.path.basename(bg_file),
                'param_name': 'brokenness', 'param_value': b,
                'idx': len(dataset),
            })
    return dataset


def run_pm5_on_dataset(dataset, p, min_area_override=None):
    """Run pm_v5 on each image, return det_map."""
    p2 = dict(p)
    if min_area_override is not None:
        p2['min_area'] = min_area_override
    det_map = {}
    for item in dataset:
        img = item['img']
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
        det_map[item['idx']] = hy
    return det_map


def evaluate(det_map, dataset, dist_thr=5.0):
    """Per-bucket TP/recall/FP/precision."""
    metrics = {}
    for item in dataset:
        b = item['param_value']
        if b not in metrics:
            metrics[b] = {'tp': 0, 'fn': 0, 'fp': 0, 'n_gt': 0, 'n_det': 0}
        gt = item['gt_quads']
        det = det_map.get(item['idx'], [])
        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            dist_thr,
        )
        metrics[b]['tp'] += tp
        metrics[b]['fn'] += len(gt) - tp
        metrics[b]['fp'] += len(det) - tp
        metrics[b]['n_gt'] += len(gt)
        metrics[b]['n_det'] += len(det)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-bg', type=int, default=30)
    ap.add_argument('--n-per-value', type=int, default=20)
    ap.add_argument('--brokenness', type=str, default='0,0.05,0.10,0.20,0.30,0.50',
                    help='Comma-separated brokenness values')
    ap.add_argument('--min-areas', type=str, default='900,500,200,100',
                    help='Comma-separated min_area values to test')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-images', action='store_true')
    args = ap.parse_args()

    p = pm5_params()
    print(f"LOOP-14: Realistic synthetic data (X-ray bg) + pm_v5 optimization (D={D})", flush=True)
    print(f"  pm5_params: ortho={p['ortho_tol']} bnd={p['boundary_band']} "
          f"aspect={p['max_aspect']} min_area={p['min_area']}", flush=True)
    print(f"  Background dir: {BG_DIR}", flush=True)

    if not os.path.isdir(BG_DIR):
        print(f"ERROR: background dir not found: {BG_DIR}", flush=True)
        return

    brokenness_values = [float(x) for x in args.brokenness.split(',')]
    min_areas = [int(x) for x in args.min_areas.split(',')]

    print(f"\nGenerating realistic synthetic dataset "
          f"(n_bg={args.n_bg}, n_per_value={args.n_per_value})...", flush=True)
    t0 = time.time()
    dataset = gen_dataset(args.n_bg, brokenness_values, args.n_per_value, seed=args.seed)
    dt = time.time() - t0
    print(f"  {len(dataset)} images generated in {dt:.1f}s", flush=True)
    print(f"  Image size: {dataset[0]['img'].shape}", flush=True)

    if args.save_images:
        sample_dir = os.path.join(ROOT, "tmp_sc14_samples")
        os.makedirs(sample_dir, exist_ok=True)
        for k, item in enumerate(dataset[:6]):
            cv2.imwrite(os.path.join(sample_dir, f"sample_{k}_b{item['param_value']}.png"),
                        item['img'])
        print(f"  Saved 6 sample images to {sample_dir}", flush=True)

    print(f"\n[Step 1] Running pm_v5 baseline (min_area={p['min_area']:.0f})...", flush=True)
    t0 = time.time()
    det_base = run_pm5_on_dataset(dataset, p)
    dt = time.time() - t0
    n_det = sum(len(v) for v in det_base.values())
    print(f"  {n_det} detections in {dt:.1f}s ({n_det/len(dataset):.2f} dets/img)", flush=True)
    metrics_base = evaluate(det_base, dataset, D)
    total_tp = sum(m['tp'] for m in metrics_base.values())
    total_n = sum(m['n_gt'] for m in metrics_base.values())
    total_det = sum(m['n_det'] for m in metrics_base.values())
    overall_recall = total_tp / total_n * 100 if total_n else 0
    print(f"  baseline: TP={total_tp}/{total_n}  recall={overall_recall:.1f}%  det={total_det}", flush=True)

    print(f"\n[Step 1 report] Per-brokenness pm_v5 performance (D={D}):", flush=True)
    print(f"  {'brokenness':>10s}  {'n_gt':>5s}  {'TP':>4s}  {'recall':>7s}  "
          f"{'det':>4s}  {'FP':>4s}", flush=True)
    for b in brokenness_values:
        m = metrics_base[b]
        recall = m['tp'] / m['n_gt'] * 100 if m['n_gt'] else 0
        print(f"  {b:>10.2f}  {m['n_gt']:>5d}  {m['tp']:>4d}  {recall:>6.1f}%  "
              f"{m['n_det']:>4d}  {m['fp']:>4d}", flush=True)

    # Targeted sweep
    print(f"\n[Step 2] Targeted: lower min_area for broken-outline recovery...", flush=True)
    targeted = {}
    for ma in min_areas:
        t0 = time.time()
        det_t = run_pm5_on_dataset(dataset, p, min_area_override=ma)
        dt = time.time() - t0
        m_t = evaluate(det_t, dataset, D)
        tt = sum(m['tp'] for m in m_t.values())
        tn = sum(m['n_gt'] for m in m_t.values())
        tdet = sum(m['n_det'] for m in m_t.values())
        tfp = sum(m['fp'] for m in m_t.values())
        t_recall = tt / tn * 100 if tn else 0
        targeted[ma] = {
            'tp': tt, 'n_gt': tn, 'det': tdet, 'fp': tfp,
            'recall': t_recall,
            'per_brokenness': {b: m_t[b] for b in brokenness_values},
        }
        delta_tp = tt - total_tp
        delta_det = tdet - total_det
        delta_fp = tfp - sum(m['fp'] for m in metrics_base.values())
        print(f"  min_area={ma:>4d}: TP={tt:>4d}/{tn}  recall={t_recall:>5.1f}%  "
              f"det={tdet:>4d}  FP={tfp:>4d}  "
              f"(ΔTP={delta_tp:+d}  ΔFP={delta_fp:+d})  [{dt:.1f}s]", flush=True)

    # Per-brokenness breakdown for each min_area
    print(f"\n  Per-brokenness recall (%):", flush=True)
    print(f"  {'min_area':>9s}  " + "  ".join(f"b={b:>4.2f}" for b in brokenness_values), flush=True)
    for ma in [p['min_area']] + min_areas:
        if ma in targeted:
            r = targeted[ma]
        else:
            r = {'per_brokenness': {b: {'tp': metrics_base[b]['tp'], 'n_gt': metrics_base[b]['n_gt']} for b in brokenness_values}}
        line = f"  {ma:>9.0f}  "
        for b in brokenness_values:
            m = r['per_brokenness'][b]
            rec = m['tp'] / max(1, m['n_gt']) * 100
            line += f"  {rec:>5.1f}%"
        print(line, flush=True)

    # Save
    out_data = {
        'dist_thr': D,
        'n_bg': args.n_bg,
        'n_per_value': args.n_per_value,
        'n_images': len(dataset),
        'pm5_params': {k: v for k, v in p.items() if not isinstance(v, tuple)},
        'baseline': {
            'tp': total_tp, 'n_gt': total_n, 'det': total_det,
            'recall': overall_recall,
            'per_brokenness': {b: metrics_base[b] for b in brokenness_values},
        },
        'targeted_min_area_sweep': {
            str(ma): targeted[ma] for ma in min_areas
        },
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()