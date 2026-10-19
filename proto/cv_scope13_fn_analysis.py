"""LOOP-13 diagnostic: characterize the 244 FN on real data.

For each unmatched GT box, compute:
  - Corner distance to nearest detection (if any)
  - Aspect ratio
  - Rotation angle
  - Box size (in pixels)
  - Red pixel coverage inside the GT box (proxy for "outline presence")
  - Red pixel density in the surrounding region

Then save diagnostic images showing the GT box (green) vs nearby detections (red),
group FN cases by their characteristics, and save a sample of each group.

Output:
  benchmark/cv_scope13_fn_analysis.json  — per-FN statistics
  tmp_sc13_fn_diag/                       — sample diagnostic images
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask
from canny_quad import get_ids
from cv_scope7 import pm5_params
from cv_hybrid import ROOT
from eval_pointmatch import load_boxes, match_boxes_point

PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
OUT_DIR = os.path.join(ROOT, "tmp_sc13_fn_diag")
OUT_JSON = os.path.join(ROOT, "benchmark", "cv_scope13_fn_analysis.json")
D = 5.0


def quad_corners_to_box(quad):
    """Convert 4-corner quad to (cx, cy, w, h, angle)."""
    quad = np.asarray(quad, np.float32)
    rect = cv2.minAreaRect(quad)
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90
    angle = angle % 180
    return cx, cy, w, h, angle


def min_corner_dist(gt_quad, det_quads):
    """Minimum per-corner L2 distance from GT quad to any detection quad."""
    if not det_quads:
        return float('inf')
    gt = np.asarray(gt_quad, np.float32)
    best = float('inf')
    for d in det_quads:
        d = np.asarray(d, np.float32)
        # Hungarian-style 4-to-4: try all 24 permutations, take min sum
        from itertools import permutations
        perms = list(permutations(range(4)))
        best_perm = float('inf')
        for p in perms:
            d_perm = d[list(p)]
            dist = np.linalg.norm(gt - d_perm, axis=1).sum()
            if dist < best_perm:
                best_perm = dist
        if best_perm < best:
            best = best_perm
    return best / 4.0  # per-corner mean


def analyze_fns(ids, dist_thr=5.0):
    """Find all FN on real data, characterize each, save diagnostic images."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fns = []
    per_image_counts = {}

    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes(LAB, iid)
        det = [d.astype(np.int32) for d in load_boxes(PM5_REF, iid)]
        if not gt:
            continue

        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        if len(gt) - tp > 0:
            per_image_counts[iid] = len(gt) - tp
        fn_idxs = set(unmatched)

        for gi, g in enumerate(gt):
            if gi not in fn_idxs:
                continue
            # Found an FN
            cx, cy, w, h, angle = quad_corners_to_box(g)
            corner_d = min_corner_dist(g, det)
            # Red mask analysis
            mask = red_mask(img)
            mask_inside = np.zeros_like(mask)
            cv2.fillPoly(mask_inside, [g.astype(np.int32)], 255)
            inside_red = (mask & mask_inside).sum() / max(1, mask_inside.sum() / 255)
            inside_count = int((mask & mask_inside).sum() / 255)
            inside_area = int(mask_inside.sum() / 255)
            # Surrounding red density
            dil = cv2.dilate(mask_inside, np.ones((20, 20), np.uint8), iterations=1)
            surround = ((dil > 0) & (mask_inside == 0)).sum() / 255
            surround_red = int(((dil > 0) & (mask_inside == 0) & (mask > 0)).sum() / 255)
            surround_density = surround_red / max(1, surround)

            fns.append({
                'iid': iid,
                'gt_idx': gi,
                'cx': float(cx), 'cy': float(cy),
                'w': float(w), 'h': float(h), 'angle': float(angle),
                'aspect': float(max(w, h) / max(1, min(w, h))),
                'min_corner_dist': float(corner_d),
                'inside_red_px': inside_count,
                'inside_area': inside_area,
                'inside_red_frac': float(inside_red),
                'surround_red_frac': float(surround_density),
                'has_close_det': corner_d < 20.0,
            })

    # Save diagnostic images for ~30 sample FNs (stratified by corner_dist bucket)
    print(f"  Saving diagnostic images for {len(fns)} FNs...", flush=True)
    fns_sorted = sorted(fns, key=lambda x: x['min_corner_dist'])
    # Stratify: 10 close (<5px), 10 medium (5-20px), 10 far (>20px)
    buckets = {'close': [], 'medium': [], 'far': []}
    for fn in fns_sorted:
        d = fn['min_corner_dist']
        if d < 5:
            buckets['close'].append(fn)
        elif d < 20:
            buckets['medium'].append(fn)
        else:
            buckets['far'].append(fn)
    saved = 0
    for bname, fn_list in buckets.items():
        for fn in fn_list[:10]:
            iid = fn['iid']
            gi = fn['gt_idx']
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            H, W = img.shape[:2]
            vis = img.copy()
            gt = load_boxes(LAB, iid)
            det = [d.astype(np.int32) for d in load_boxes(PM5_REF, iid)]
            cv2.polylines(vis, [gt[gi].astype(np.int32)], True, (0, 255, 0), 2)
            for d in det:
                cv2.polylines(vis, [d.astype(np.int32)], True, (0, 0, 255), 1)
            # Add text
            label = (f"FN #{saved} corner_d={fn['min_corner_dist']:.1f}px "
                     f"w={fn['w']:.0f} h={fn['h']:.0f} ang={fn['angle']:.0f} "
                     f"inside_red={fn['inside_red_frac']*100:.0f}%")
            cv2.putText(vis, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUT_DIR, f"fn_{bname}_{saved:03d}.png"), vis)
            saved += 1
    print(f"  Saved {saved} diagnostic images to {OUT_DIR}/", flush=True)

    return fns, per_image_counts


def main():
    ids = get_ids()
    print(f"LOOP-13 FN analysis on {len(ids)} real images, D={D}", flush=True)
    print(f"  pm_v5 reference: {PM5_REF}", flush=True)

    t0 = time.time()
    fns, per_image = analyze_fns(ids, D)
    dt = time.time() - t0
    print(f"  Analyzed {len(fns)} FN in {dt:.1f}s", flush=True)

    # Summary statistics
    print("\n" + "=" * 78, flush=True)
    print(f"FN CHARACTERIZATION (D={D}, 500 imgs, {len(fns)} FN total)", flush=True)
    print("=" * 78, flush=True)

    # Bucket by corner distance
    close = [fn for fn in fns if fn['min_corner_dist'] < 5]
    medium = [fn for fn in fns if 5 <= fn['min_corner_dist'] < 20]
    far = [fn for fn in fns if fn['min_corner_dist'] >= 20]
    print(f"\n  By min corner distance to nearest detection:", flush=True)
    print(f"    close   (<5px):   {len(close):>3d}  ({100*len(close)/len(fns):.1f}%)  "
          f"— could be matched with relaxed threshold", flush=True)
    print(f"    medium  (5-20px): {len(medium):>3d}  ({100*len(medium)/len(fns):.1f}%)  "
          f"— detectable but imprecise corners", flush=True)
    print(f"    far     (>=20px): {len(far):>3d}  ({100*len(far)/len(fns):.1f}%)  "
          f"— completely missed or no nearby det", flush=True)

    # Bucket by inside_red_frac
    has_outline = [fn for fn in fns if fn['inside_red_frac'] > 0.01]
    no_outline = [fn for fn in fns if fn['inside_red_frac'] <= 0.01]
    print(f"\n  By red pixel presence inside GT box:", flush=True)
    print(f"    has outline (inside_red>1%):   {len(has_outline):>3d}  "
          f"({100*len(has_outline)/len(fns):.1f}%)", flush=True)
    print(f"    no outline  (inside_red<=1%):  {len(no_outline):>3d}  "
          f"({100*len(no_outline)/len(fns):.1f}%)  — likely annotation noise or very faded", flush=True)

    # Bucket by aspect
    high_aspect = [fn for fn in fns if fn['aspect'] > 4]
    low_aspect = [fn for fn in fns if fn['aspect'] <= 4]
    print(f"\n  By aspect ratio (w/h):", flush=True)
    print(f"    high aspect (>4): {len(high_aspect):>3d}  ({100*len(high_aspect)/len(fns):.1f}%)", flush=True)
    print(f"    low aspect  (<=4): {len(low_aspect):>3d}  ({100*len(low_aspect)/len(fns):.1f}%)", flush=True)

    # Bucket by rotation
    high_rot = [fn for fn in fns if abs(fn['angle'] - 45) > 30 and abs(fn['angle'] - 135) > 30]
    low_rot = [fn for fn in fns if not (abs(fn['angle'] - 45) > 30 and abs(fn['angle'] - 135) > 30)]
    print(f"\n  By rotation angle:", flush=True)
    print(f"    high rotation (|ang-45|>30 and |ang-135|>30): "
          f"{len(high_rot):>3d}  ({100*len(high_rot)/len(fns):.1f}%)", flush=True)
    print(f"    low rotation:  {len(low_rot):>3d}  ({100*len(low_rot)/len(fns):.1f}%)", flush=True)

    # Cross-tabulation: close corner dist + has outline = recoverable?
    recoverable = [fn for fn in fns if fn['min_corner_dist'] < 5 and fn['inside_red_frac'] > 0.01]
    print(f"\n  POTENTIALLY RECOVERABLE (close+has outline): "
          f"{len(recoverable):>3d}  ({100*len(recoverable)/len(fns):.1f}%)", flush=True)
    print(f"  These have a nearby detection (<5px) AND red pixels inside the GT box.", flush=True)
    print(f"  → A corner-refinement method (already tried in LOOP-4, DEAD) MIGHT help.", flush=True)
    print(f"  → Or a relaxed-threshold metric (e.g. D=8 or D=12) catches them.", flush=True)

    # Images with most FN
    if per_image:
        worst = sorted(per_image.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 images with most FN:", flush=True)
        for iid, n in worst:
            print(f"    {iid}: {n} FN", flush=True)

    print("=" * 78, flush=True)

    # Save full FN analysis
    out_data = {
        'dist_thr': D,
        'n_total_fn': len(fns),
        'n_images_with_fn': len(per_image),
        'per_image_fn_counts': per_image,
        'fn_list': [{k: (bool(v) if isinstance(v, np.bool_) else
                         float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in fn.items()} for fn in fns],
        'buckets': {
            'close': len(close), 'medium': len(medium), 'far': len(far),
            'has_outline': len(has_outline), 'no_outline': len(no_outline),
            'high_aspect': len(high_aspect), 'low_aspect': len(low_aspect),
            'high_rot': len(high_rot), 'low_rot': len(low_rot),
            'recoverable_close_with_outline': len(recoverable),
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_JSON}", flush=True)
    print(f"Diagnostic images: {OUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()