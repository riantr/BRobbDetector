"""Visualize constructed synthetic data — red = simulated OBB, black = GT (minAreaRect).

Generates N diverse samples from the synthetic benchmark dataset, draws both
the simulated red OBB and the GT (loaded from tmp_synth_gt/*.json) on the
synthetic image, and saves them to tmp_synth_viz/ for inspection.

Usage: python proto/visualize_synth.py [--n N] [--out OUT_DIR]
"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from cv_scope13 import obb_corners
from cv_scope14 import list_bg_images, BG_DIR
from cv_hybrid import ROOT
from cv_scope15 import BG_DIR as BG_DIR_15  # ensure import

GT_DIR = os.path.join(ROOT, "tmp_synth_gt")
OUT_DIR = os.path.join(ROOT, "tmp_synth_viz")
DEFAULT_OBB = {'box_w': 80, 'box_h': 50, 'angle': 15, 'stroke_w': 2}
BROKENNESS_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEED = 42


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


def gen_for_viz(n_per_value, seed=SEED):
    """Regenerate the same synthetic dataset as eval_synth_benchmark.

    IMPORTANT: must use sorted(glob.glob(...))[:50] to match the benchmark's
    image ordering. The cv_scope14.list_bg_images() does random sampling
    which produces a DIFFERENT image set.
    """
    from cv_scope13 import draw_obb, compute_gt_minarearect
    bg_files = sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))[:50]
    rng = np.random.RandomState(seed)
    dataset = []
    for bi, bg_file in enumerate(bg_files):
        bg = cv2.imread(bg_file)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        for brokenness in BROKENNESS_VALUES:
            for k in range(n_per_value):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--n-per-value', type=int, default=4)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--with-detection', action='store_true',
                    help='Overlay pm_v5 detection (blue) on top of red/black')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # First ensure GT exists; if not, run benchmark
    if not os.path.isdir(GT_DIR) or len(os.listdir(GT_DIR)) < 10:
        print(f"GT dir empty; running benchmark first to populate {GT_DIR}/...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                         "eval_synth_benchmark.py"),
                         "--method", "pm_v5", "--save", "synth_pm_v5"],
                        check=False)

    print(f"Generating visualization dataset...")
    dataset = gen_for_viz(args.n_per_value, args.seed)

    by_b = {}
    for item in dataset:
        by_b.setdefault(item['brokenness'], []).append(item)
    samples = []
    for b in sorted(by_b.keys()):
        samples.extend(by_b[b][:max(1, args.n // len(by_b))])
    samples = samples[:args.n]

    # Optionally load detection results
    det_dir = os.path.join(ROOT, "tmp_synth_det")
    det_map = {}
    if args.with_detection and os.path.isdir(det_dir):
        for fn in os.listdir(det_dir):
            if not fn.endswith(".json"):
                continue
            iid = fn[:-5]
            with open(os.path.join(det_dir, fn), encoding="utf-8") as f:
                data = json.load(f)
            det_map[iid] = [parse_points(info["points"]) for info in data.get("infos", [])]
        print(f"  Loaded detections for {len(det_map)} images from {det_dir}/")
    elif args.with_detection:
        print(f"  WARNING: --with-detection requested but {det_dir}/ not found; run benchmark first")

    legend = (f"  RED = simulated OBB  |  BLACK = GT (from JSON)  "
              f"|  BLUE = pm_v5 detection" if args.with_detection else
              f"  RED = simulated OBB  |  BLACK = GT (from JSON)")
    print(f"Saving {len(samples)} visualizations to {args.out}/...")
    print(legend + "\n")

    for k, item in enumerate(samples):
        img = item['img'].copy()
        bp = item['bp']
        b = item['brokenness']
        iid = item['iid']

        sim_corners = obb_corners(bp['cx'], bp['cy'], bp['box_w'], bp['box_h'], bp['angle'])
        cv2.polylines(img, [sim_corners.astype(np.int32)], True, (0, 0, 255), 2)

        gt_quads = load_gt(iid)
        for gq in gt_quads:
            cv2.polylines(img, [gq], True, (0, 0, 0), 3)

        det_count = 0
        if args.with_detection and iid in det_map:
            for dq in det_map[iid]:
                cv2.polylines(img, [dq], True, (255, 0, 0), 2)
                det_count += 1

        if gt_quads:
            mean_d = float(np.mean([np.linalg.norm(sc - gc)
                                    for gq in gt_quads for sc, gc in zip(sim_corners, gq)]))
        else:
            mean_d = -1

        label = (f"{iid}  b={b:.2f}  ang={bp['angle']:.0f}  "
                 f"w={bp['box_w']}x{bp['box_h']}  bp=({bp['cx']},{bp['cy']})  "
                 f"sim-vs-GT={mean_d:.1f}px  dets={det_count}")
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)

        out_path = os.path.join(args.out, f"viz_{k:02d}_b{b:.2f}.png")
        cv2.imwrite(out_path, img)
        print(f"  [{k:2d}] b={b:.2f}  sim-vs-GT={mean_d:.1f}px  dets={det_count}  {iid}  -> {out_path}")


if __name__ == "__main__":
    main()