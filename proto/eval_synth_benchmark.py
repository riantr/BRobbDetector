"""Synthetic X-ray benchmark — uses real X-ray backgrounds + injected OBBs
with minAreaRect GT (real-time per image).

Pattern matches `proto/eval_benchmark.py`: --save <name> and --compare <name>
flags, machine-readable JSON + human-readable MD baselines.

Dataset (deterministic, seed=42):
  - 50 unique X-ray backgrounds from data\Image@Xray_CSHH_Cargo_TF_P01_A
  - 5 brokenness values (0, 0.05, 0.10, 0.20, 0.30) × 4 images each
  - 1 OBB per image (default 80x50 at 15°), position random in interior
  - GT = minAreaRect of rendered red pixels in the box region (rebuilt per image)
  - Total: 200 images

Methods compared:
  pm_v5       = hybrid(h+e+m)+filter, min_area=900 (frozen baseline)
  pm_v5_1     = pm_v5 with min_area=1000 (LOOP-15c WIN)

Usage:
  python proto/eval_synth_benchmark.py --save synth_baseline
  python proto/eval_synth_benchmark.py --method pm_v5_1 --save synth_minarea1000
  python proto/eval_synth_benchmark.py --method pm_v5_1 --compare synth_baseline
"""
import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import obb_corners, draw_obb, compute_gt_minarearect
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
DET_DIR = os.path.join(ROOT, "tmp_synth_det")
GT_DIR = os.path.join(ROOT, "tmp_synth_gt")
BENCH_DIR = os.path.join(ROOT, "benchmark")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]
N_BG = 50
N_PER_VALUE = 4
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 42
DEFAULT_OBB = {'box_w': 80, 'box_h': 50, 'angle': 15, 'stroke_w': 2}


def list_bg_images():
    return sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))


def gen_synth_dataset():
    """Build a deterministic synthetic dataset: each item = {img, gt_quads, iid, meta}.

    GT is rebuilt per-image via compute_gt_minarearect.
    """
    bg_files = list_bg_images()
    if N_BG > len(bg_files):
        bg_files = bg_files
    else:
        bg_files = bg_files[:N_BG]

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
                    'bp': bp, 'bg_file': os.path.basename(bg_file),
                    'brokenness': brokenness,
                })
    return dataset


def write_gt(dataset):
    """Write GT to GT_DIR as JSON per image."""
    os.makedirs(GT_DIR, exist_ok=True)
    for item in dataset:
        out = {
            "image_id": item['iid'],
            "infos": [
                {"points": ";".join(f"{int(x)},{int(y)}" for x, y in q)}
                for q in item['gt_quads']
            ],
        }
        path = os.path.join(GT_DIR, item['iid'] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def write_img(dataset, img_dir):
    """Write images to img_dir as JPG."""
    os.makedirs(img_dir, exist_ok=True)
    for item in dataset:
        cv2.imwrite(os.path.join(img_dir, item['iid'] + ".jpg"), item['img'])


def parse_points(s):
    pts = []
    for p in s.split(';'):
        x, y = p.split(',')
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)


def load_gt(iid):
    path = os.path.join(GT_DIR, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]


def load_det(iid):
    path = os.path.join(DET_DIR, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]


def run_detector(dataset, p):
    """Run pm_v5 hybrid(h+e+m)+filter on each image, save det JSONs."""
    os.makedirs(DET_DIR, exist_ok=True)
    for f in os.listdir(DET_DIR):
        os.remove(os.path.join(DET_DIR, f))
    t0 = time.time()
    n_boxes = 0
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
        n_boxes += len(hy)
        out = {
            "image_id": item['iid'],
            "infos": [
                {"points": ";".join(f"{int(x)},{int(y)}" for x, y in q)}
                for q in hy
            ],
        }
        with open(os.path.join(DET_DIR, item['iid'] + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
    dt = time.time() - t0
    return n_boxes, dt


def compute_metrics(dataset, dist_thr=5.0):
    """Compute point-match metrics with brokenness breakdown."""
    TP = FN = FP = 0
    per_image = []
    by_brokenness = {}
    for item in dataset:
        gt = item['gt_quads']
        det = load_det(item['iid'])
        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            dist_thr,
        )
        b = item['brokenness']
        if b not in by_brokenness:
            by_brokenness[b] = {'tp': 0, 'fn': 0, 'fp': 0, 'n_gt': 0, 'n_det': 0}
        by_brokenness[b]['tp'] += tp
        by_brokenness[b]['fn'] += len(gt) - tp
        by_brokenness[b]['fp'] += len(unmatched)
        by_brokenness[b]['n_gt'] += len(gt)
        by_brokenness[b]['n_det'] += len(det)
        TP += tp
        FN += len(gt) - tp
        FP += len(unmatched)
        per_image.append({
            "id": item['iid'],
            "brokenness": b,
            "gt": len(gt), "det": len(det),
            "tp": tp, "fp": len(unmatched), "fn": len(gt) - tp,
        })
    n = len(dataset)
    total_gt = TP + FN
    total_det = TP + FP
    recall = TP / total_gt * 100 if total_gt else 0
    precision = TP / total_det * 100 if total_det else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    for b, m in by_brokenness.items():
        m['recall'] = round(m['tp'] / m['n_gt'] * 100, 2) if m['n_gt'] else 0
        m['precision'] = round(m['tp'] / m['n_det'] * 100, 2) if m['n_det'] else 0
    return {
        "config": {
            "n_bg": N_BG,
            "brokenness_values": BROKENNESS_VALUES,
            "n_per_value": N_PER_VALUE,
            "total_images": n,
            "dist_thr": dist_thr,
        },
        "box_level": {
            "tp": TP, "fp": FP, "fn": FN,
            "total_gt": total_gt, "total_det": total_det,
            "recall": round(recall, 2),
            "precision": round(precision, 2),
            "f1": round(f1, 2),
        },
        "by_brokenness": {str(b): by_brokenness[b] for b in sorted(by_brokenness.keys())},
        "per_image": per_image,
    }


def multi_d_compute_metrics(dataset, dist_values):
    out = {}
    for dd in dist_values:
        out[f"D{dd}"] = compute_metrics(dataset, float(dd))["box_level"]
    return out


def save_baseline(name, method, params, results, multi_d):
    base_path = os.path.join(BENCH_DIR, f"{name}.json")
    payload = {
        "name": name,
        "method": method,
        "params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in params.items() if not isinstance(v, (list, tuple))},
        "results": results,
        "multi_d": multi_d,
    }
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    md_path = os.path.join(BENCH_DIR, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        bl = results["box_level"]
        f.write(f"# Synthetic benchmark: {name}\n\n")
        f.write(f"**Method**: {method}  \n")
        f.write(f"**Dataset**: {results['config']['total_images']} images  \n")
        f.write(f"  - {results['config']['n_bg']} unique X-ray backgrounds  \n")
        f.write(f"  - brokenness: {results['config']['brokenness_values']}  \n")
        f.write(f"  - GT: minAreaRect of rendered red pixels (rebuilt per image)  \n")
        f.write(f"**Distance threshold**: D={results['config']['dist_thr']}  \n\n")
        f.write(f"## Box-level metrics (D={results['config']['dist_thr']})\n\n")
        f.write(f"| TP | FP | FN | total_gt | total_det | recall | precision | f1 |\n")
        f.write(f"|----|----|----|----------|-----------|--------|-----------|----|\n")
        f.write(f"| {bl['tp']} | {bl['fp']} | {bl['fn']} | {bl['total_gt']} | "
                f"{bl['total_det']} | {bl['recall']:.1f}% | {bl['precision']:.1f}% | "
                f"{bl['f1']:.1f}% |\n\n")
        f.write(f"## Per-brokenness recall (D={results['config']['dist_thr']})\n\n")
        f.write(f"| brokenness | n_gt | TP | recall | det | precision |\n")
        f.write(f"|------------|------|----|--------|-----|-----------|\n")
        for b, m in sorted(results["by_brokenness"].items(), key=lambda x: float(x[0])):
            f.write(f"| {float(b):.2f} | {m['n_gt']} | {m['tp']} | {m['recall']:.1f}% | "
                    f"{m['n_det']} | {m['precision']:.1f}% |\n")
        f.write(f"\n## Multi-D performance\n\n")
        f.write(f"| D | TP | recall | precision | f1 |\n")
        f.write(f"|---|----|--------|-----------|----|\n")
        for d, m in multi_d.items():
            f.write(f"| {d[1:]} | {m['tp']} | {m['recall']:.1f}% | "
                    f"{m['precision']:.1f}% | {m['f1']:.1f}% |\n")
    print(f"Saved: {base_path}")
    print(f"Saved: {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='pm_v5', choices=['pm_v5', 'pm_v5_1'])
    ap.add_argument('--save', default=None, help='save baseline name')
    ap.add_argument('--compare', default=None, help='compare against baseline name')
    ap.add_argument('--no-build', action='store_true', help='skip dataset rebuild')
    ap.add_argument('--no-run', action='store_true', help='skip detector run')
    args = ap.parse_args()

    # Build params
    p = pm5_params()
    if args.method == 'pm_v5_1':
        p['min_area'] = 1000  # LOOP-15c WIN

    print(f"Synthetic X-ray benchmark: method={args.method}, min_area={p['min_area']}", flush=True)
    print(f"  Backgrounds: {N_BG}, brokenness: {BROKENNESS_VALUES}, n_per_value: {N_PER_VALUE}", flush=True)

    if not args.no_build:
        print(f"\nBuilding synthetic dataset (real-time GT)...", flush=True)
        t0 = time.time()
        dataset = gen_synth_dataset()
        dt = time.time() - t0
        print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)
        write_gt(dataset)
        print(f"  GT written to {GT_DIR}/", flush=True)
    else:
        print(f"\nSkipping dataset build (--no-build), reloading GT...", flush=True)
        # Reload from GT_DIR
        dataset = []
        for iid in sorted(os.listdir(GT_DIR)):
            if not iid.endswith(".json"):
                continue
            iid_clean = iid[:-5]
            # parse iid format: synth_BBB_bBB.BB_K
            parts = iid_clean.split("_")
            try:
                bi = int(parts[1])
                b = float(parts[2][1:])
                k = int(parts[3])
            except (IndexError, ValueError):
                continue
            # Re-derive img from bg_files (deterministic)
            # For --no-build, we just skip — need a saved dataset
            print(f"  --no-build requires saving dataset first; abort", flush=True)
            return

    if not args.no_run:
        print(f"\nRunning detector...", flush=True)
        t0 = time.time()
        n_boxes, dt = run_detector(dataset, p)
        print(f"  {n_boxes} boxes in {dt:.1f}s", flush=True)

    # Compute metrics
    print(f"\nComputing metrics (D={D})...", flush=True)
    results = compute_metrics(dataset, D)
    bl = results["box_level"]
    print(f"  TP={bl['tp']} FP={bl['fp']} FN={bl['fn']}  "
          f"recall={bl['recall']:.1f}% precision={bl['precision']:.1f}% f1={bl['f1']:.1f}%", flush=True)

    print(f"\nPer-brokenness:", flush=True)
    for b, m in sorted(results["by_brokenness"].items(), key=lambda x: float(x[0])):
        print(f"  b={float(b):.2f}: TP={m['tp']:>3d}/{m['n_gt']:>3d}  "
              f"recall={m['recall']:.1f}%  det={m['n_det']}  prec={m['precision']:.1f}%", flush=True)

    print(f"\nMulti-D performance...", flush=True)
    multi_d = multi_d_compute_metrics(dataset, DIST_VALUES_FULL)
    for d, m in multi_d.items():
        print(f"  {d}: TP={m['tp']} recall={m['recall']:.1f}% "
              f"prec={m['precision']:.1f}% f1={m['f1']:.1f}%", flush=True)

    if args.save:
        save_baseline(args.save, args.method, p, results, multi_d)

    if args.compare:
        comp_path = os.path.join(BENCH_DIR, f"{args.compare}.json")
        if not os.path.exists(comp_path):
            print(f"ERROR: compare baseline not found: {comp_path}", flush=True)
            return
        with open(comp_path, encoding="utf-8") as f:
            other = json.load(f)
        print(f"\n" + "=" * 78, flush=True)
        print(f"COMPARE: {args.method} (current) vs {other['name']} ({other['method']})", flush=True)
        print("=" * 78, flush=True)
        for k in ['tp', 'fp', 'fn', 'recall', 'precision', 'f1']:
            cur = bl[k]
            oth = other["results"]["box_level"][k]
            d = cur - oth if isinstance(cur, (int, float)) else "N/A"
            print(f"  {k:>12s}: current={cur}  baseline={oth}  Δ={d}", flush=True)
        # By-brokenness comparison
        print(f"\nPer-brokenness recall (current vs baseline):", flush=True)
        for b in sorted(results["by_brokenness"].keys(), key=float):
            cur = results["by_brokenness"][b]
            oth = other["results"]["by_brokenness"].get(b, {})
            d = cur.get("tp", 0) - oth.get("tp", 0)
            print(f"  b={float(b):.2f}: cur_TP={cur.get('tp', 0)}/{cur.get('n_gt', 0)}  "
                  f"base_TP={oth.get('tp', 0)}/{oth.get('n_gt', 0)}  Δ={d:+d}", flush=True)


if __name__ == "__main__":
    main()