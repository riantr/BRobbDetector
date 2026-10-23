"""LOOP-75: how do GT boxes relate to the image border?

The new rule ("a rectangle with a vertex on the image edge is invalid") only
makes sense as a *consistent* rule if the ground truth obeys it too.  This
script measures, for the 500-image benchmark:

  * how many GT boxes have a vertex exactly on the border (x==0 / x==W-1 /
    y==0 / y==H-1) -> the human drew a box that was clipped by the image,
  * how many GT boxes merely come *close* to the border (<= 1/2/6 px),
  * the same statistics for the current detections.

The verdict decides whether the edge rule should be applied to the GT as well
(consistent) or only to the detections (filter-only).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, load_all

BANDS = [0, 1, 2, 6]


def on_edge(quad, W, H, band):
    q = np.asarray(quad, float)
    for i in range(4):
        x, y = q[i]
        if x <= band or x >= W - band or y <= band or y >= H - band:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/scope75_edge_gt.json")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()

    gt_stats = {b: 0 for b in BANDS}
    det_stats = {b: 0 for b in BANDS}
    exact = {"x0": 0, "xW1": 0, "y0": 0, "yH1": 0, "any": 0}
    n_gt = 0
    n_det = 0
    dist_hist = {}
    examples = []

    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        for g in gt_map[iid]:
            n_gt += 1
            q = np.asarray(g, float)
            d = min(min(q[:, 0]), min(q[:, 1]), W - 1 - max(q[:, 0]), H - 1 - max(q[:, 1]))
            dist_hist[int(max(d, 0))] = dist_hist.get(int(max(d, 0)), 0) + 1
            for b in BANDS:
                if on_edge(g, W, H, b):
                    gt_stats[b] += 1
            hit = False
            for i in range(4):
                x, y = q[i]
                if abs(x - 0) < 0.5:
                    exact["x0"] += 1
                    hit = True
                if abs(x - (W - 1)) < 0.5:
                    exact["xW1"] += 1
                    hit = True
                if abs(y - 0) < 0.5:
                    exact["y0"] += 1
                    hit = True
                if abs(y - (H - 1)) < 0.5:
                    exact["yH1"] += 1
                    hit = True
            if hit:
                exact["any"] += 1
                if len(examples) < 12:
                    examples.append({"id": iid, "W": W, "H": H,
                                     "quad": [list(map(int, p)) for p in q]})

        for q in detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT):
            n_det += 1
            for b in BANDS:
                if on_edge(q, W, H, b):
                    det_stats[b] += 1

    print(f"GT boxes: {n_gt}   detections: {n_det}")
    print("GT vertex on border (exact):", exact)
    for b in BANDS:
        print(f"  band {b}: GT {gt_stats[b]:4d} ({gt_stats[b]/max(n_gt,1)*100:5.1f}%)  "
              f"det {det_stats[b]:4d} ({det_stats[b]/max(n_det,1)*100:5.1f}%)")
    print("min-distance-to-border histogram (GT):",
          dict(sorted(dist_hist.items())[:12]))
    print("examples:")
    for e in examples:
        print("   ", e)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_gt": n_gt, "n_det": n_det, "gt_stats": gt_stats,
                   "det_stats": det_stats, "exact": exact,
                   "dist_hist": dist_hist, "examples": examples},
                  f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()