"""LOOP-44 viz: render merged red components (2+ GT boxes in one component).

Shows the image crop, the red mask, the component contour, the minAreaRect of
the whole component and the individual GT boxes - so we can see HOW the two
boxes are joined (corner touch / shared edge / bridge through junk).

Usage:
  python proto/scope44_viz_merged.py --n 6
  python proto/scope44_viz_merged.py --n 6 --id 20260905_193507_874
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from scope43_fn_forensics import perim_points, PROD, RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
OUT = os.path.join(ROOT, "tmp_merged_viz")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--id", default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ids = get_ids()
    if args.id:
        ids = [args.id]
    p = PROD
    shown = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        if len(gt) < 2:
            continue
        mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
        den = denoise(mask, p["denoise_min_area"]) if p["denoise_min_area"] > 0 else mask
        H, W = den.shape
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(den, 8)
        groups = collections.defaultdict(list)
        for gi, g in enumerate(gt):
            pts = perim_points(g, step=2)
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            vals = lab[ys, xs]
            vals = vals[vals > 0]
            if not len(vals):
                continue
            comp, k = collections.Counter(vals.tolist()).most_common(1)[0]
            if k / len(pts) >= 0.5:
                groups[comp].append(gi)
        for comp, gis in groups.items():
            if len(gis) < 2:
                continue
            x, y, w, h, area = stats[comp]
            pad = 12
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
            crop = img[y0:y1, x0:x1].copy()
            cmask = (lab[y0:y1, x0:x1] == comp).astype(np.uint8) * 255
            mcrop = mask[y0:y1, x0:x1].copy()
            vis = crop.copy()
            vis[cmask > 0] = (vis[cmask > 0] * 0.45 + np.array([0, 0, 140]) * 0.55).astype(np.uint8)
            cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                cv2.drawContours(vis, [c], -1, (0, 255, 0), 1)
                rect = cv2.minAreaRect(c)
                cv2.drawContours(vis, [cv2.boxPoints(rect).astype(np.int32)], -1, (0, 200, 255), 1)
            for gi in gis:
                g = gt[gi].astype(np.int32) - np.array([x0, y0])
                cv2.polylines(vis, [g], True, (255, 0, 255), 1)
            tile = np.hstack([crop, vis, cv2.cvtColor(mcrop, cv2.COLOR_GRAY2BGR)])
            sc = min(1600.0 / max(tile.shape[1], 1), 1.0)
            if sc < 1.0:
                tile = cv2.resize(tile, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUT, f"{iid}_c{comp}_g{gis}.jpg"), tile)
            print(f"  {iid} comp={comp} gts={gis} area={area} "
                  f"bbox=({x},{y},{w},{h}) -> tmp_merged_viz")
            shown += 1
            if shown >= args.n:
                return
    print(f"wrote {shown} tiles to {OUT}")


if __name__ == "__main__":
    main()