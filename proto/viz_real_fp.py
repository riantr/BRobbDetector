"""Render the real false positives of the production pipeline.

Colours:  green  = red mask (the burned-in annotation)
          yellow = ground truth rectangle
          red    = the false positive detection

Usage:
    python proto/viz_real_fp.py --n 12 --sort area
    python proto/viz_real_fp.py --n 12 --kind lonely
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect)
from eval_pointmatch import is_real_unlabeled, corner_match_distance
from quad_ortho import quad_on_boundary
from analyze_real_fp import sides_of, matched_index

ROOT = r"D:\src\CodeArts\BRobbDetector"
D = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sort", default="area", choices=["area", "min_side"])
    ap.add_argument("--kind", default="all",
                    choices=["all", "lonely", "near"])
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp_real_fp.png"))
    args = ap.parse_args()

    ids = get_ids()
    exp = load_expected()
    fps = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        used = matched_index(dets, gts, D)
        wide = cv2.dilate(red_mask(img, 80, 110, 10),
                          np.ones((3, 3), np.uint8), iterations=3) > 0
        for di, d in enumerate(dets):
            if di in used:
                continue
            if is_real_unlabeled(d, exp.get(iid, 0), len(gts), wide):
                continue
            cd = min([corner_match_distance(d, g) for g in gts], default=1e9)
            kind = "lonely" if cd > 25 else "near"
            if args.kind != "all" and kind != args.kind:
                continue
            mn, mx = sides_of(np.asarray(d, float))
            fps.append((mx if args.sort == "area" else mn, iid, img, d, gts))
    fps.sort(key=lambda t: -t[0])
    print("trueFP shown: %d" % len(fps))

    tiles = []
    for mside, iid, img, d, gts in fps[:args.n]:
        H, W = img.shape[:2]
        q = np.asarray(d, float)
        pad = 30
        x0 = int(max(0, q[:, 0].min() - pad))
        x1 = int(min(W, q[:, 0].max() + pad))
        y0 = int(max(0, q[:, 1].min() - pad))
        y1 = int(min(H, q[:, 1].max() + pad))
        c = img[y0:y1, x0:x1].copy()
        m = red_mask(c, P["sat_min"], P["val_min"], P["hue_max"])
        m = denoise(m, P["denoise_min_area"])
        c[m > 0] = (0, 255, 0)
        for g in gts:
            gg = np.asarray(g, float) - np.array([x0, y0])
            if (gg[:, 0].min() > -60 and gg[:, 0].max() < c.shape[1] + 60
                    and gg[:, 1].min() > -60 and gg[:, 1].max() < c.shape[0] + 60):
                cv2.polylines(c, [gg.astype(np.int32)], True, (0, 255, 255), 1)
        dd = np.asarray(d, float) - np.array([x0, y0])
        cv2.polylines(c, [dd.astype(np.int32)], True, (0, 0, 255), 2)
        c = cv2.resize(c, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, "%s ms=%.0f" % (iid[-6:], mside), (4, 16),
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
        y = r * (h + 4)
        x = c_ * (cw + 4)
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    cv2.imwrite(args.out, canvas)
    print("wrote %s (%d tiles)" % (args.out, len(tiles)))


if __name__ == "__main__":
    main()