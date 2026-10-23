"""LOOP-63: calibrate the corner geometry of the hole quad.

`corner_match_distance` is the MAX distance over the 4 corners, so a
systematic corner bias directly caps the TP rate.  For every detection of the
hole pipeline this script measures, per corner, the signed offset of the GT
corner along the OUTWARD bisector of the fitted quad (positive = the GT is
further outside than the fit).

If the mean offset is a constant != 0 the fit can be corrected by growing the
quad along its bisectors.

Usage: python proto/scope63_calib.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance, PERMS_4
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, detect_holes, detect_edgefit)
from cv_hybrid import dedup_keep_order
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)


def best_perm(det, gt):
    best = None
    for perm in PERMS_4:
        mx = 0.0
        for i in range(4):
            d = det[i] - gt[perm[i]]
            mx = max(mx, float(d @ d))
        if best is None or mx < best[0]:
            best = (mx ** 0.5, perm)
    return best


def bisector(q, i):
    """Outward bisector direction at corner i of a quad."""
    a = np.asarray(q[(i - 1) % 4], float)
    b = np.asarray(q[i], float)
    c = np.asarray(q[(i + 1) % 4], float)
    u1 = b - a
    u2 = c - b
    n1 = np.array([-u1[1], u1[0]])
    n2 = np.array([-u2[1], u2[0]])
    ctr = np.asarray(q, float).mean(axis=0)
    if np.dot(n1, ctr - b) > 0:
        n1 = -n1
    if np.dot(n2, ctr - b) > 0:
        n2 = -n2
    v = n1 + n2
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-9 else n1


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    stats = collections.Counter()
    off_all = []
    off_near = []
    off_far = []
    maxd = collections.Counter()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        dets = detect_holes(img, P, CFG, PIPE) + detect_edgefit(img, P, CFG, PIPE)
        dets = dedup_keep_order(dets, PIPE["dedup_iou"])
        for d in dets:
            d = np.asarray(d, float)
            best = None
            for g in gt:
                dd, perm = best_perm(d, np.asarray(g, float))
                if best is None or dd < best[0]:
                    best = (dd, perm, g)
            if best is None or best[0] > 20:
                continue
            dd, perm, g = best
            maxd[min(int(dd // 2) * 2, 20)] += 1
            for i in range(4):
                b = bisector(d, i)
                off = float((np.asarray(g[perm[i]], float) - d[i]) @ b)
                off_all.append(off)
                (off_near if dd <= 5 else off_far).append(off)
    for name, arr in (("all", off_all), ("d<=5", off_near), ("d>5", off_far)):
        if not arr:
            continue
        a = np.array(arr)
        print(f"{name:6s} n={len(a):6d}  mean={a.mean():+6.2f}  std={a.std():5.2f}  "
              f"median={np.median(a):+6.2f}  |mean| per corner "
              f"{[round(float(x),2) for x in a.reshape(-1,4).mean(0)] if len(a)%4==0 else ''}")
    print("\nmax corner distance histogram (2px buckets):",
          sorted(maxd.items()))
    out = os.path.join(ROOT, "benchmark", "scope63_calib.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mean_all": float(np.mean(off_all)) if off_all else 0,
                   "std_all": float(np.std(off_all)) if off_all else 0,
                   "maxd_hist": {str(k): v for k, v in maxd.items()}},
                  f, ensure_ascii=False, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()