"""LOOP-68 viz: crop around a FN, draw the hole, the fitted quad and the GT.

Usage: python proto/scope68_viz.py <id> <gt_index> [more...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _hole_quad
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
OUT = os.path.join(ROOT, "tmp_c1_viz")


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(HOLE_DEFAULT)
    args = sys.argv[1:]
    pairs = [(args[i], int(args[i + 1])) for i in range(0, len(args), 2)]
    for iid, gi in pairs:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        gt = load_boxes_any(RAW, iid)
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        den = denoise(mask, P["denoise_min_area"])
        g = np.asarray(gt[gi], np.int32)
        vis = img.copy()
        for h in _holes_of(den, cfg):
            cv2.drawContours(vis, [h], -1, (0, 255, 255), 1)
            q = _hole_quad(h, den, cfg)
            if q is not None:
                cv2.drawContours(vis, [q.astype(np.int32)], -1, (0, 255, 0), 1)
        cv2.polylines(vis, [g], True, (255, 0, 255), 2)
        for c in g:
            cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
        x0 = max(int(g[:, 0].min()) - 40, 0)
        x1 = min(int(g[:, 0].max()) + 40, vis.shape[1])
        y0 = max(int(g[:, 1].min()) - 40, 0)
        y1 = min(int(g[:, 1].max()) + 40, vis.shape[0])
        crop = vis[y0:y1, x0:x1]
        scale = max(1, int(700 / max(crop.shape[0], 1)))
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        out = os.path.join(OUT, f"{iid}_{gi}.png")
        cv2.imwrite(out, crop)
        print("wrote", out, crop.shape)


if __name__ == "__main__":
    main()