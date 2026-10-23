"""Render the real GT boxes whose side is "broken" (the red is not within
+-8 px of the ideal line over the whole side), TP and FN side by side.

The real data shows a very steep drop: a box with 0 broken sides is found in
96.9% of the cases, with 1 broken side only in 42.9%.  The emulation has to
reproduce that, so we need to SEE what the red of such a box looks like.

Usage: python proto/viz_real_broken.py [--n 5] [--out tmp_real_broken.png]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import denoise
from scope45_holes import P, RAW
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def n_broken(r):
    return sum(1 for c in r["band"] if c[0] < 0.999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=os.path.join(ROOT, "benchmark",
                                                  "real_degrade_full.json"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp_real_broken.png"))
    args = ap.parse_args()

    rows = json.load(open(args.rows, encoding="utf-8"))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    tiles = []
    for want_tp in (False, True):
        k = 0
        for iid, rs in by_id.items():
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            H, W = img.shape[:2]
            gts = [np.asarray(g, float) for g in load_boxes_any(RAW, iid)
                   if not quad_on_boundary(g, W, H, 0.0)]
            if len(gts) != len(rs):
                continue
            for r, g in zip(rs, gts):
                if n_broken(r) != 1 or r["tp"] != want_tp:
                    continue
                tiles.append((r, img, g, iid))
                k += 1
                if k >= args.n:
                    break
            if k >= args.n:
                break

    out = []
    for r, img, g, iid in tiles:
        H, W = img.shape[:2]
        pad = 14
        x0 = int(max(0, g[:, 0].min() - pad))
        x1 = int(min(W, g[:, 0].max() + pad))
        y0 = int(max(0, g[:, 1].min() - pad))
        y1 = int(min(H, g[:, 1].max() + pad))
        c = img[y0:y1, x0:x1].copy()
        m = red_mask(c, P["sat_min"], P["val_min"], P["hue_max"])
        m = denoise(m, P["denoise_min_area"])
        c[m > 0] = (0, 255, 0)
        gg = g - np.array([x0, y0])
        cv2.polylines(c, [gg.astype(np.int32)], True, (0, 255, 255), 1)
        out.append((c, "%s tp=%d d=%s" % (iid[-6:], r["tp"], r["dist"])))

    if not out:
        print("no tiles")
        return
    maxh = max(c.shape[0] for c, _ in out)
    total = sum(c.shape[1] for c, _ in out)
    canvas = np.zeros((maxh + 20, total, 3), np.uint8)
    x = 0
    for c, txt in out:
        canvas[0:c.shape[0], x:x + c.shape[1]] = c
        cv2.putText(canvas, txt, (x + 3, maxh + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
        x += c.shape[1]
    cv2.imwrite(args.out, canvas)
    print("wrote %s (%d tiles)" % (args.out, len(out)))


if __name__ == "__main__":
    main()