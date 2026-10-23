"""Render the geometric near-misses: GTs that ARE detected but off by >5 px.

For those boxes the pipeline finds the object but localises it too coarsely to
count as a TP at D=5.  This script shows the GT (yellow), the detection
(red) and the red mask (green) so that the failure mode is visible.

Usage:
    python proto/viz_geom_miss.py --n 16 --lo 5 --hi 30
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect
from eval_pointmatch import corner_match_distance
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lo", type=float, default=5.0)
    ap.add_argument("--hi", type=float, default=1e9)
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp_geom_miss.png"))
    args = ap.parse_args()

    ids = get_ids()
    cases = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        for g in gts:
            if not dets:
                continue
            ds = [corner_match_distance(d, g) for d in dets]
            k = int(np.argmin(ds))
            if args.lo < ds[k] <= args.hi:
                cases.append((ds[k], iid, img, g, dets[k]))
    cases.sort(key=lambda t: -t[0])
    print("near-miss cases: %d" % len(cases))

    tiles = []
    for cd, iid, img, g, d in cases[:args.n]:
        H, W = img.shape[:2]
        q = np.asarray(g, float)
        pad = 25
        x0 = int(max(0, q[:, 0].min() - pad))
        x1 = int(min(W, q[:, 0].max() + pad))
        y0 = int(max(0, q[:, 1].min() - pad))
        y1 = int(min(H, q[:, 1].max() + pad))
        c = img[y0:y1, x0:x1].copy()
        m = red_mask(c, P["sat_min"], P["val_min"], P["hue_max"])
        m = denoise(m, P["denoise_min_area"])
        c[m > 0] = (0, 255, 0)
        gg = np.asarray(g, float) - np.array([x0, y0])
        cv2.polylines(c, [gg.astype(np.int32)], True, (0, 255, 255), 2)
        dd = np.asarray(d, float) - np.array([x0, y0])
        cv2.polylines(c, [dd.astype(np.int32)], True, (0, 0, 255), 2)
        c = cv2.resize(c, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, "%s cd=%.0f" % (iid[-6:], cd), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(c)

    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    cw = max(t.shape[1] for t in tiles)
    canvas = np.zeros((rows * (h + 4), cols * (cw + 4), 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c_ = divmod(k, cols)
        canvas[r * (h + 4):r * (h + 4) + t.shape[0],
               c_ * (cw + 4):c_ * (cw + 4) + t.shape[1]] = t
    cv2.imwrite(args.out, canvas)
    print("wrote %s (%d tiles)" % (args.out, len(tiles)))


if __name__ == "__main__":
    main()