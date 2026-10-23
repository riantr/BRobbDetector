"""LOOP-47 diagnostic: is the GT interior really enclosed by red?

Flood-fills the background from the image border and checks whether the
eroded GT interior is reachable from outside.  If it is, the red outline has
a GAP somewhere - find where by looking at the GT perimeter profile.

Usage: python proto/scope47_flood.py 20260911_200451_013:0 ...
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _side_geom
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    cfg = dict(HOLE_DEFAULT)
    for spec in sys.argv[1:]:
        iid, gi = spec.split(":")
        gi = int(gi)
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        gt = load_boxes_any(RAW, iid)
        g = gt[gi].astype(np.int32)
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        den = denoise(mask, P["denoise_min_area"])

        poly = np.zeros(den.shape, np.uint8)
        cv2.fillPoly(poly, [g], 255)
        inner = cv2.erode(poly, np.ones((9, 9), np.uint8))
        # background flood fill from the image border
        bg = np.zeros((den.shape[0] + 2, den.shape[1] + 2), np.uint8)
        inv = (den == 0).astype(np.uint8) * 255
        cv2.floodFill(inv, bg, (0, 0), 128)
        outside = inv == 128
        reachable = bool(outside[inner > 0].any())

        holes = _holes_of(den, cfg)
        # how many hole contour points fall inside the GT?
        cnt = np.zeros(den.shape, np.uint8)
        cv2.fillPoly(cnt, [g], 1)
        inside_pts = 0
        total_pts = 0
        for h in holes:
            pts = h.reshape(-1, 2)
            total_pts += len(pts)
            inside_pts += int(cnt[pts[:, 1], pts[:, 0]].sum())
        print(f"{iid} gt#{gi}: interior_reachable_from_outside={reachable} "
              f"holes={len(holes)} hole_pts_inside_gt={inside_pts}/{total_pts}")

        if reachable:
            # find the gap: walk the GT perimeter and look for non-red spots
            for s in range(4):
                a, u, n, L = _side_geom(g.astype(float), s)
                ts = np.linspace(0.02, 0.98, 60)
                pts = a[None, :] + ts[:, None] * (u * L)[None, :]
                xs = np.clip(np.round(pts[:, 0]).astype(int), 0, den.shape[1] - 1)
                ys = np.clip(np.round(pts[:, 1]).astype(int), 0, den.shape[0] - 1)
                red = den[ys, xs] > 0
                if not red.all():
                    miss = np.where(~red)[0]
                    print(f"    side{s}: {len(miss)}/{len(red)} non-red, "
                          f"first at t={ts[miss[0]]:.2f} (pos {miss[0]})")


if __name__ == "__main__":
    main()