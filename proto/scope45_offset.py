"""LOOP-45 diagnostic: where is the hole-derived box relative to the GT?

The hole-based detector reaches TP=643 at D=12 but only 53 at D=3, i.e. the
recovered box is systematically OFFSET from the GT by 5-12 px.  This script
measures the signed per-side offset (and the size delta) so we can find out
whether the GT is the stroke centre line, the inner edge or the outer edge.

Usage: python proto/scope45_offset.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask
from canny_quad import get_ids, denoise, _order_quad
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, PROD_DEFAULT)
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, _holes_of, _hole_quad, _side_geom,
                           _band_centre)
from scope43_fn_forensics import PROD, RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def side_offset(gt, quad):
    """Signed offsets of quad's sides relative to gt's sides (gt = reference).

    Both are convex quads; match the sides by direction, then report the
    distance from the gt side line to the quad side line along the outward
    normal of the gt side (positive = quad side is OUTSIDE the gt).
    """
    out = []
    for i in range(4):
        a, u, n, L = _side_geom(np.asarray(gt, float), i)
        # quad side with the closest parallel direction
        best, berr = None, 1e9
        for j in range(4):
            ga, gu, gn, gL = _side_geom(np.asarray(quad, float), j)
            err = abs(abs(float(np.dot(gu, u))) - 1.0)
            if err < berr:
                berr, best = err, (ga, gu)
        if best is None:
            continue
        ga, gu = best
        # signed distance from the gt side line, measured along gt outward n
        out.append(float(np.dot(ga - a, n)))
    return out


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    cfg = dict(HOLE_DEFAULT)
    rows = []
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        for hole in _holes_of(mask, cfg):
            q = _hole_quad(hole, mask, cfg, P)
            if q is None:
                continue
            # nearest GT
            best = min(gt, key=lambda g: corner_match_distance(q.astype(np.int32), g))
            d = corner_match_distance(q.astype(np.int32), best)
            offs = side_offset(best, q)
            rect = cv2.minAreaRect(hole)
            grow = [0.0, 0.0]
            # size comparison along the gt axes
            ga, gu, gn, gL = _side_geom(np.asarray(best, float), 0)
            ga2, gu2, gn2, gL2 = _side_geom(np.asarray(best, float), 2)
            rows.append({
                "id": iid, "d": round(d, 1), "offs": [round(o, 1) for o in offs],
                "hole_wh": [round(rect[1][0], 1), round(rect[1][1], 1)],
                "gt_sides": [round(gL, 1), round(gL2, 1)],
            })
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] rows={len(rows)}")

    print(f"\nrows={len(rows)}")
    d = np.array([r["d"] for r in rows])
    for thr in (5, 8, 12, 20):
        print(f"  corner dist <= {thr:3d}: {int((d <= thr).sum()):4d}")
    # only rows that are plausibly a box (dist < 25)
    good = [r for r in rows if r["d"] <= 25]
    print(f"\nplausible pairs: {len(good)}")
    off = np.array([r["offs"] for r in good], dtype=float)
    print("  per-side signed offset (quad side relative to gt side, + = outside):")
    for k in range(4):
        col = off[:, k]
        print(f"    side{k}: med={np.median(col):6.2f}  p10={np.percentile(col,10):6.2f} "
              f"p90={np.percentile(col,90):6.2f}  mean={col.mean():6.2f}")
    # size ratio: hole vs gt along both axes
    print("\n  hole size vs gt side lengths (rough):")
    for r in good[:12]:
        print(f"    {r['id']} d={r['d']:6.1f} offs={r['offs']} "
              f"hole_wh={r['hole_wh']} gt_sides={r['gt_sides']}")

    out = os.path.join(ROOT, "benchmark", "scope45_offset.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()