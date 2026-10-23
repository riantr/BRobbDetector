"""Render real GT boxes with a low red support (TP and FN) side by side.

Each crop shows the annotated image with
  blue  = the hand drawn GT rectangle
  green = the detection of the production pipeline (if any)

Usage: python proto/viz_real_degrade.py [--out tmp_real_degrade.png] [--n 6]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def side_support(mask, g):
    H, W = mask.shape
    sups = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        L = float(np.linalg.norm(b - a))
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        sups.append(float((mask[ys, xs] > 0).mean()))
    return sups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp_real_degrade.png"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--imgs", type=int, default=60)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.9)
    args = ap.parse_args()

    ids = get_ids()[:args.imgs]
    tiles = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = [g for g in load_boxes_any(RAW, iid) if not quad_on_boundary(g, W, H, 0.0)]
        if not gts:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        dets = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)
        for g in gts:
            sups = side_support(mask, g)
            if not (args.lo <= min(sups) < args.hi):
                continue
            best, bd = 1e9, None
            for d in dets:
                dd = corner_match_distance(np.asarray(d, np.int32), np.asarray(g, np.int32))
                if dd < best:
                    best, bd = dd, d
            g = np.asarray(g, np.int32)
            c = g.mean(axis=0)
            rad = int(max(60, np.linalg.norm(g - c[None, :], axis=1).max() + 40))
            x0, y0 = int(c[0]) - rad, int(c[1]) - rad
            x1, y1 = int(c[0]) + rad, int(c[1]) + rad
            px0, py0 = max(0, x0), max(0, y0)
            px1, py1 = min(W, x1), min(H, y1)
            crop = img[py0:py1, px0:px1].copy()
            gm = g - [px0, py0]
            cv2.polylines(crop, [gm], True, (255, 0, 0), 1)
            if bd is not None and best <= 5.0:
                cv2.polylines(crop, [np.asarray(bd, np.int32) - [px0, py0]], True,
                              (0, 255, 0), 1)
            s = 2 if max(crop.shape[:2]) < 320 else 1
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            lbl = "%s sup=%.2f %s" % (iid[-6:], min(sups),
                                      "TP" if best <= 5.0 else "FN")
            cv2.putText(crop, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
            if len(tiles) >= args.n:
                break
        if len(tiles) >= args.n:
            break

    if not tiles:
        print("no tiles")
        return 1
    hh = max(t.shape[0] for t in tiles)
    ww = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), 4):
        row = []
        for t in tiles[i:i + 4]:
            pad = np.zeros((hh, ww, 3), np.uint8)
            pad[:t.shape[0], :t.shape[1]] = t
            row.append(pad)
        while len(row) < 4:
            row.append(np.zeros((hh, ww, 3), np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(args.out, grid)
    print("wrote %s  (%d tiles, %dx%d)" % (args.out, len(tiles), grid.shape[1], grid.shape[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())