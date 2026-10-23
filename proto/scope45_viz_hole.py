"""LOOP-45 viz: why is the hole only a strip inside the GT box?

Usage: python proto/scope45_viz_hole.py 20260912_112325_000 [20260907_091937_197 ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise, _order_quad
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
OUT = os.path.join(ROOT, "tmp_hole_viz")


def main():
    ids = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(HOLE_DEFAULT)
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        gt = load_boxes_any(RAW, iid)
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        den = denoise(mask, P["denoise_min_area"])
        vis = img.copy()
        for g in gt:
            cv2.polylines(vis, [g.astype(np.int32)], True, (255, 0, 255), 2)
        holes = _holes_of(den, cfg)
        for h in holes:
            cv2.drawContours(vis, [h], -1, (0, 255, 255), 1)
            r = cv2.minAreaRect(h)
            cv2.drawContours(vis, [cv2.boxPoints(r).astype(np.int32)], -1, (0, 200, 0), 1)
        cv2.imwrite(os.path.join(OUT, f"{iid}.jpg"),
                    np.hstack([img, vis, cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)]))
        print(f"{iid}: {len(gt)} GT, {len(holes)} holes -> tmp_hole_viz/{iid}.jpg")


if __name__ == "__main__":
    main()