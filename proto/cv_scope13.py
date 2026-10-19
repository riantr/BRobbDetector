"""LOOP-13: Synthetic OpenCV data + CV validation + orthogonal optimization.

GATE: D=5 TP > 502 AND corr_f1 > 71.3 on REAL data.
       (synthetic data is the validation/optimization tool, not the final metric)

Strategy:
  1. Generate synthetic images with parameterized OBBs (angle, brokenness,
     stroke width, aspect ratio). 4-axis sweep, 1 box per image.
  2. Run pm_v5 on synthetic data; measure per-parameter TP/recall.
  3. Identify the worst failure modes (which parameter × value combinations
     drop the recall).
  4. Design targeted CV methods for the worst failure modes (next iter).
  5. Validate on real data.

Usage: python proto/cv_scope13.py [--n-per-value N] [--dist D] [--save-images]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise
from cv_hybrid import outer_contours, _pass, dedup_keep_order
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import match_boxes_point


# ── Synthetic data generator ──

def obb_corners(cx, cy, w, h, angle_deg):
    """Compute 4 corners of a rotated rectangle, in CW order starting at +x."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    hwx, hwy = (w / 2) * cos_a, (w / 2) * sin_a
    hhx, hhy = -(h / 2) * sin_a, (h / 2) * cos_a
    return np.array([
        [cx + hwx + hhx, cy + hwy + hhy],  # +x +y
        [cx - hwx + hhx, cy - hwy + hhy],  # -x +y
        [cx - hwx - hhx, cy - hwy - hhy],  # -x -y
        [cx + hwx - hhx, cy + hwy - hhy],  # +x -y
    ], dtype=np.float32)


def draw_obb(img, corners, brokenness=0, stroke_w=2, color=(0, 0, 255),
              bg_color=(30, 30, 30), rng=None, line_aa=True,
              brokenness_mode='corner'):
    """Draw OBB outline with controlled brokenness.

    brokenness_mode:
      'corner'  : place gaps at corners (preserves 4-edge structure)
                  with probability brokenness per corner. The gap is a
                  small region around the corner erased in bg_color,
                  disconnecting the 2 edges meeting at that corner.
                  Result: up to 4 disconnected edge fragments, each
                  clearly recognizable as a box edge.
      'per_edge' : place a gap of brokenness*edge_length at a random
                  position on EACH edge (legacy model; can split edges
                  into 2 sub-fragments).
      'single'   : place ONE gap of brokenness*perimeter at a random
                  position on a random edge (only 1 edge is broken).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    line_type = cv2.LINE_AA if line_aa else cv2.LINE_8
    bg_color_arr = np.array(bg_color, dtype=np.uint8)

    if brokenness <= 0:
        cv2.polylines(img, [corners.astype(np.int32)], True, color, stroke_w, line_type)
        return

    if brokenness_mode == 'corner':
        # First draw the full outline, then erase corner regions
        cv2.polylines(img, [corners.astype(np.int32)], True, color, stroke_w, line_type)
        # For each corner, with probability brokenness, erase a small region
        for i in range(4):
            if rng.random() < brokenness:
                p_corner = corners[i].astype(int)
                # Gap radius scales with brokenness
                gap_radius = max(3, int(stroke_w * 2 + brokenness * 8))
                cv2.circle(img, tuple(p_corner), gap_radius, bg_color_arr.tolist(), -1, line_type)
    elif brokenness_mode == 'per_edge':
        n = 4
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            gap = brokenness
            max_start = max(0.0, 1.0 - gap - 1e-3)
            t_start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
            t_end = t_start + gap
            if t_start > 0.005:
                q1 = p1 + t_start * (p2 - p1)
                cv2.line(img, tuple(p1.astype(int)), tuple(q1.astype(int)),
                         color, stroke_w, line_type)
            if t_end < 0.995:
                q2 = p1 + t_end * (p2 - p1)
                cv2.line(img, tuple(q2.astype(int)), tuple(p2.astype(int)),
                         color, stroke_w, line_type)
    elif brokenness_mode == 'single':
        # Place ONE gap of brokenness*perimeter at a random position on a random edge
        perim = 0
        for i in range(4):
            perim += float(np.linalg.norm(corners[(i+1) % 4] - corners[i]))
        gap = brokenness * perim
        # Pick a random edge
        edge_idx = int(rng.randint(0, 4))
        p1 = corners[edge_idx]
        p2 = corners[(edge_idx + 1) % 4]
        edge_len = float(np.linalg.norm(p2 - p1))
        if gap < edge_len:
            max_start = max(0.0, edge_len - gap - 1e-3)
            t_start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
            t_end = t_start + gap
            for j in range(4):
                a, b = corners[j], corners[(j+1) % 4]
                if j != edge_idx:
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                             color, stroke_w, line_type)
                else:
                    if t_start > 0.5:
                        q1 = a + (t_start / edge_len) * (b - a)
                        cv2.line(img, tuple(a.astype(int)), tuple(q1.astype(int)),
                                 color, stroke_w, line_type)
                    if t_end < edge_len - 0.5:
                        q2 = a + (t_end / edge_len) * (b - a)
                        cv2.line(img, tuple(q2.astype(int)), tuple(b.astype(int)),
                                 color, stroke_w, line_type)
    else:
        raise ValueError(f"Unknown brokenness_mode: {brokenness_mode}")


def gen_synthetic_image(H, W, boxes_params, bg_color=(30, 30, 30), bg_noise=10, rng=None):
    """Generate one synthetic image with multiple boxes (BGR)."""
    if rng is None:
        rng = np.random.RandomState(0)
    img = np.full((H, W, 3), bg_color, dtype=np.uint8)
    if bg_noise > 0:
        noise = rng.normal(0, bg_noise, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    gt_quads = []
    for bp in boxes_params:
        corners = obb_corners(bp['cx'], bp['cy'], bp['w'], bp['h'], bp['angle'])
        draw_obb(img, corners, brokenness=bp.get('brokenness', 0),
                  stroke_w=bp.get('stroke_w', 2), color=(0, 0, 255),
                  bg_color=bg_color, rng=rng,
                  brokenness_mode=bp.get('brokenness_mode', 'per_edge'))
        gt_quads.append(corners)
    return img, gt_quads


def compute_gt_minarearect(img, original_corners, padding=10,
                            smin=80, vmin=110):
    """Compute minAreaRect of red pixels in the region near the original box.

    This treats GT as 'the best-fit rectangle of what's actually drawn',
    matching the philosophy of real-data annotation (annotator fits a quad
    to the visible outline).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    x, y, w, h = cv2.boundingRect(original_corners.astype(np.int32))
    x_lo = max(0, x - padding)
    y_lo = max(0, y - padding)
    x_hi = min(img.shape[1], x + w + padding)
    y_hi = min(img.shape[0], y + h + padding)
    if x_hi <= x_lo or y_hi <= y_lo:
        return original_corners
    region = mask[y_lo:y_hi, x_lo:x_hi]
    ys, xs = np.where(region > 0)
    if len(xs) < 4:
        return original_corners
    pts = np.stack([xs + x_lo, ys + y_lo], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def gen_param_sweep_dataset(n_per_value, rng_seed=42, gt_mode='minarearect'):
    """Generate dataset sweeping one parameter at a time (other params at default).

    gt_mode: 'analytical' uses obb_corners (intended rectangle).
             'minarearect' uses minAreaRect of the rendered red pixels.
    """
    H, W = 256, 384

    # Default params (representing typical real data)
    default = {
        'box_w': 80, 'box_h': 50, 'angle': 15, 'brokenness': 0,
        'stroke_w': 2, 'bg_noise': 10, 'bg_color': (30, 30, 30),
    }

    sweeps = {
        'angle':      [0, 15, 30, 45, 60, 75],
        'brokenness': [0, 0.05, 0.10, 0.20, 0.30, 0.50],
        'stroke_w':   [1, 2, 3, 4, 5],
        'aspect':     [1, 2, 3, 5, 8],
    }

    rng = np.random.RandomState(rng_seed)
    dataset = []
    for param_name, values in sweeps.items():
        for value in values:
            for k in range(n_per_value):
                params = dict(default)
                if param_name == 'angle':
                    params['angle'] = value
                elif param_name == 'brokenness':
                    params['brokenness'] = value
                elif param_name == 'stroke_w':
                    params['stroke_w'] = value
                elif param_name == 'aspect':
                    area = default['box_w'] * default['box_h']
                    params['box_w'] = int(np.sqrt(area * value))
                    params['box_h'] = max(1, int(np.sqrt(area / value)))

                cx = rng.randint(60, W - 60)
                cy = rng.randint(40, H - 40)

                bp = {
                    'cx': cx, 'cy': cy,
                    'w': params['box_w'], 'h': params['box_h'],
                    'angle': params['angle'],
                    'brokenness': params['brokenness'],
                    'stroke_w': params['stroke_w'],
                }
                img, gt_analytical = gen_synthetic_image(
                    H, W, [bp],
                    bg_color=default['bg_color'],
                    bg_noise=default['bg_noise'],
                    rng=rng,
                )
                if gt_mode == 'minarearect':
                    gt = [compute_gt_minarearect(img, c, padding=10)
                          for c in gt_analytical]
                else:
                    gt = gt_analytical
                dataset.append({
                    'img': img, 'gt_quads': gt,
                    'gt_analytical': gt_analytical,
                    'param_name': param_name, 'param_value': value,
                    'idx': len(dataset),
                })
    return dataset


# ── pm_v5 runner ──

def run_pm5_on_dataset(dataset, p, min_area_override=None):
    """Run pm_v5 pipeline (hybrid h+e+m + o15_ba) on each image.

    min_area_override: if set, override p['min_area'] with this value
    (used to test broken-outline recovery by lowering the min-area filter).
    """
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


def evaluate_synthetic(det_map, dataset, dist_thr=5.0):
    """Per-parameter TP/recall/precision."""
    metrics = {}
    for item in dataset:
        pname = item['param_name']
        pval = item['param_value']
        gt = item['gt_quads']
        det = det_map.get(item['idx'], [])

        if pname not in metrics:
            metrics[pname] = {}
        if pval not in metrics[pname]:
            metrics[pname][pval] = {'tp': 0, 'fn': 0, 'det': 0, 'n_gt': 0}

        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            dist_thr,
        )
        metrics[pname][pval]['tp'] += tp
        metrics[pname][pval]['fn'] += len(gt) - tp
        metrics[pname][pval]['det'] += len(det)
        metrics[pname][pval]['n_gt'] += len(gt)
    return metrics


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-value', type=int, default=25)
    ap.add_argument('--dist', type=float, default=5.0)
    ap.add_argument('--save-images', action='store_true',
                    help='Save a few synthetic images to tmp_sc13_samples/ for inspection')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-targeted', action='store_true',
                    help='Skip the targeted-method (min_area sweep) step')
    ap.add_argument('--min-areas', type=str, default='200,100,50,0',
                    help='Comma-separated min_area values to test for broken-outline recovery')
    ap.add_argument('--gt-mode', choices=['analytical', 'minarearect'],
                    default='minarearect',
                    help="GT generation: 'analytical' (intended rectangle) or "
                         "'minarearect' (best-fit rectangle of rendered pixels)")
    args = ap.parse_args()

    p = pm5_params()
    print(f"LOOP-13: Synthetic data + pm_v5 validation + targeted optimization (D={args.dist})", flush=True)
    print(f"  pm5_params: ortho={p['ortho_tol']} bnd={p['boundary_band']} "
          f"aspect={p['max_aspect']} min_side={p['min_side']} "
          f"min_area={p['min_area']}", flush=True)
    print(f"  gt_mode: {args.gt_mode}", flush=True)

    print(f"\nGenerating synthetic dataset (n_per_value={args.n_per_value}, seed={args.seed})...", flush=True)
    t0 = time.time()
    dataset = gen_param_sweep_dataset(args.n_per_value, rng_seed=args.seed, gt_mode=args.gt_mode)
    dt = time.time() - t0
    print(f"  {len(dataset)} images generated in {dt:.1f}s", flush=True)

    if args.save_images:
        sample_dir = os.path.join(os.path.dirname(__file__), '..', 'tmp_sc13_samples')
        os.makedirs(sample_dir, exist_ok=True)
        for item in dataset[:6]:
            cv2.imwrite(os.path.join(sample_dir, f"sample_{item['idx']}.png"),
                        item['img'])
        print(f"  Saved 6 sample images to {sample_dir}", flush=True)

    # ── Step 1: pm_v5 baseline on synthetic data ──
    print(f"\n[Step 1] Running pm_v5 baseline on synthetic dataset...", flush=True)
    t0 = time.time()
    det_map_base = run_pm5_on_dataset(dataset, p)
    dt = time.time() - t0
    n_det = sum(len(v) for v in det_map_base.values())
    print(f"  {n_det} detections in {dt:.1f}s "
          f"({n_det / len(dataset):.2f} dets/img)", flush=True)
    metrics_base = evaluate_synthetic(det_map_base, dataset, args.dist)
    total_tp = sum(m['tp'] for pvals in metrics_base.values() for m in pvals.values())
    total_n = sum(m['n_gt'] for pvals in metrics_base.values() for m in pvals.values())
    total_det = sum(m['det'] for pvals in metrics_base.values() for m in pvals.values())
    overall_recall = total_tp / total_n * 100 if total_n else 0
    print(f"  baseline: TP={total_tp}/{total_n}  recall={overall_recall:.1f}%  det={total_det}", flush=True)

    # Per-parameter report (baseline)
    print("\n" + "=" * 78, flush=True)
    print(f"BASELINE pm_v5 PER-PARAMETER (D={args.dist})", flush=True)
    print("=" * 78, flush=True)
    for pname in sorted(metrics_base.keys()):
        print(f"\n  [{pname}]", flush=True)
        print(f"    {'value':>10s}  {'n_gt':>5s}  {'TP':>4s}  {'recall':>7s}  "
              f"{'det':>4s}", flush=True)
        for pval in sorted(metrics_base[pname].keys()):
            m = metrics_base[pname][pval]
            recall = m['tp'] / m['n_gt'] * 100 if m['n_gt'] else 0
            print(f"    {str(pval):>10s}  {m['n_gt']:>5d}  {m['tp']:>4d}  "
                  f"{recall:>6.1f}%  {m['det']:>4d}", flush=True)

    # ── Step 2: Targeted method — lower min_area for broken-outline recovery ──
    if args.skip_targeted:
        print("\n[Step 2] Skipped (--skip-targeted)", flush=True)
        targeted_results = {}
    else:
        min_areas = [int(x) for x in args.min_areas.split(',')]
        targeted_results = {}
        print("\n" + "=" * 78, flush=True)
        print(f"[Step 2] Targeted method: lower min_area for broken-outline recovery", flush=True)
        print("=" * 78, flush=True)
        print(f"  pm_v5 baseline min_area = {p['min_area']:.0f}", flush=True)
        for ma in min_areas:
            t0 = time.time()
            det_map_t = run_pm5_on_dataset(dataset, p, min_area_override=ma)
            dt = time.time() - t0
            metrics_t = evaluate_synthetic(det_map_t, dataset, args.dist)
            tt = sum(m['tp'] for pvals in metrics_t.values() for m in pvals.values())
            tn = sum(m['n_gt'] for pvals in metrics_t.values() for m in pvals.values())
            tdet = sum(m['det'] for pvals in metrics_t.values() for m in pvals.values())
            t_recall = tt / tn * 100 if tn else 0
            targeted_results[ma] = {
                'tp': tt, 'n_gt': tn, 'det': tdet,
                'recall': t_recall,
                'brokenness_recall': {
                    str(b): metrics_t['brokenness'][b]['tp'] / max(1, metrics_t['brokenness'][b]['n_gt']) * 100
                    for b in sorted(metrics_t['brokenness'].keys())
                },
                'metrics': {pname: {str(pval): m for pval, m in pvals.items()}
                            for pname, pvals in metrics_t.items()},
            }
            delta_tp = tt - total_tp
            delta_det = tdet - total_det
            print(f"  min_area={ma:>4d}: TP={tt:>4d}/{tn}  recall={t_recall:>5.1f}%  "
                  f"det={tdet:>4d}  (ΔTP={delta_tp:+d}  Δdet={delta_det:+d})  "
                  f"[{dt:.1f}s]", flush=True)

        # Per-brokenness breakdown
        print(f"\n  Per-brokenness breakdown:", flush=True)
        for ma in min_areas:
            r = targeted_results[ma]
            print(f"    min_area={ma:>4d}:  ", end="", flush=True)
            for b in sorted(r['brokenness_recall'].keys(), key=float):
                print(f"b={b}: {r['brokenness_recall'][b]:>5.1f}%  ", end="", flush=True)
            print(flush=True)

    # ── Save results ──
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmark')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cv_scope13.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dist_thr': args.dist,
            'n_per_value': args.n_per_value,
            'n_images': len(dataset),
            'pm5_params': {k: v for k, v in p.items() if not isinstance(v, tuple)},
            'baseline': {
                'tp': total_tp, 'n_gt': total_n, 'det': total_det,
                'recall': overall_recall,
                'metrics': {pname: {str(pval): m for pval, m in pvals.items()}
                            for pname, pvals in metrics_base.items()},
            },
            'targeted_min_area_sweep': targeted_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == '__main__':
    main()