"""LOOP-47 viz: why is there no hole inside this GT box?

Usage: python proto/scope47_viz_nohole.py 20260911_200451_013:0 20260905_193507_874:0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
OUT = os.path.join(ROOT, "tmp_nohole_viz")


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(HOLE_DEFAULT)
    for spec in sys.argv[1:]:
        iid, gi = spec.split(":")
        gi = int(gi)
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        gt = load_boxes_any(RAW, iid)
        g = gt[gi].astype(np.int32)
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        den = denoise(mask, P["denoise_min_area"])

        x, y, w, h = cv2.boundingRect(g)
        pad = 30
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])

        # how much of the GT INTERIOR is red?
        poly = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(poly, [g], 255)
        inner = poly > 0
        # erode to drop the stroke itself
        inner_shrunk = cv2.erode(inner.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        frac_red = float((den[inner_shrunk] > 0).mean()) if inner_shrunk.any() else -1.0
        print(f"{iid} gt#{gi}: interior(eroded) red fraction = {frac_red:.3f} "
              f"({inner_shrunk.sum()} px)")

        vis = img.copy()
        cv2.polylines(vis, [g], True, (255, 0, 255), 2)
        holes = _holes_of(den, cfg)
        for h in holes:
            cv2.drawContours(vis, [h], -1, (0, 255, 255), 1)
        tile = np.hstack([img[y0:y1, x0:x1], vis[y0:y1, x0:x1],
                          cv2.cvtColor(den[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(os.path.join(OUT, f"{iid}_{gi}.jpg"), tile)
        print(f"   -> tmp_nohole_viz/{iid}_{gi}.jpg  ({len(holes)} holes in image)")


if __name__ == "__main__":
    main()