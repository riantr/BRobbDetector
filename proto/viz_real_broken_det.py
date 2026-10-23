"""Render the real GT boxes with the pipeline's DETECTIONS, for the boxes whose
side is "broken" (no red within +-8 px of the ideal line over the whole side).

The broken-side boxes are found in 42.9% of the cases (a box without a broken
side in 96.9%).  The question is what the detector does with them: the
nearest detection of a missed box is ~36 px away, so the pipeline finds a
*wrong* rectangle rather than nothing.

Usage: python proto/viz_real_broken_det.py [--n 4] [--out tmp_real_broken_det.png]
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
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def n_broken(r):
    return sum(1 for c in r["band"] if c[0] < 0.999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=os.path.join(ROOT, "benchmark",
                                                  "real_degrade_full.json"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp_real_broken_det.png"))
    args = ap.parse_args()

    rows = json.load(open(args.rows, encoding="utf-8"))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    tiles = []
    for iid, rs in by_id.items():
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = [np.asarray(g, float) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        if len(gts) != len(rs):
            continue
        mask = denoise(red_mask(img, P["sat_min"], P["val_min"], P["hue_max"]),
                       P["denoise_min_area"])
        dets = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask)
        for r, g in zip(rs, gts):
            if n_broken(r) != 1:
                continue
            if r["tp"]:
                continue
            tiles.append((r, img, g, dets, iid))
            if len(tiles) >= args.n:
                break
        if len(tiles) >= args.n:
            break

    out = []
    for r, img, g, dets, iid in tiles:
        H, W = img.shape[:2]
        pad = 10
        x0 = int(max(0, g[:, 0].min() - pad))
        x1 = int(min(W, g[:, 0].max() + pad))
        y0 = int(max(0, g[:, 1].min() - pad))
        y1 = int(min(H, g[:, 1].max() + pad))
        c = img[y0:y1, x0:x1].copy()
        m = red_mask(c, P["sat_min"], P["val_min"], P["hue_max"])
        m = denoise(m, P["denoise_min_area"])
        c[m > 0] = (0, 255, 0)
        for d in dets:
            dd = np.asarray(d, float) - np.array([x0, y0])
            cv2.polylines(c, [dd.astype(np.int32)], True, (255, 200, 0), 1)
        gg = g - np.array([x0, y0])
        cv2.polylines(c, [gg.astype(np.int32)], True, (0, 255, 255), 1)
        c = cv2.resize(c, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, "%s nd=%s ndet=%d" % (iid[-6:], r["dist"], len(dets)),
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out.append(c)

    if not out:
        print("no tiles")
        return
    h = max(o.shape[0] for o in out)
    w = sum(o.shape[1] for o in out)
    canvas = np.zeros((h, w, 3), np.uint8)
    x = 0
    for o in out:
        canvas[:o.shape[0], x:x + o.shape[1]] = o
        x += o.shape[1]
    cv2.imwrite(args.out, canvas)
    print("wrote %s (%d tiles)" % (args.out, len(out)))


if __name__ == "__main__":
    main()