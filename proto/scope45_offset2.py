"""LOOP-45 diagnostic v2: correct side matching between the hole quad and the GT.

scope45_profile.py proved the GT side line runs through the CENTRE of the
hand-drawn stroke (peak at s = -1..0, band ~[-3,+3]).  So the hole (inner
edge of the stroke) must be expanded by ~half the stroke width to land on
the GT.  This script measures the expansion that each side actually needs,
so the constant can be calibrated instead of guessed.

Usage: python proto/scope45_offset2.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _hole_quad, _side_geom
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def match_sides(gt, quad):
    """For each GT side, the signed offset of the matching quad side.

    A GT side has anchor a, direction u and OUTWARD normal n.  The matching
    quad side is the parallel one (|dot| ~ 1) that is closest to the GT
    side line; its offset along n is returned (positive = outside the GT).
    """
    out = []
    for i in range(4):
        a, u, n, L = _side_geom(np.asarray(gt, float), i)
        best, berr = None, 1e18
        for j in range(4):
            qa, qu, qn, qL = _side_geom(np.asarray(quad, float), j)
            if abs(abs(float(np.dot(qu, u))) - 1.0) > 0.12:
                continue
            s = float(np.dot(qa - a, n))
            if abs(s) < berr:
                berr, best = abs(s), s
        out.append(best)
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
            hr = cv2.minAreaRect(hole)
            harea = cv2.contourArea(hole)
            if harea / max(hr[1][0] * hr[1][1], 1e-6) < cfg["hole_fill"]:
                continue
            q = _hole_quad(hole, mask, cfg, P)
            if q is None:
                continue
            best = min(gt, key=lambda g: corner_match_distance(q.astype(np.int32), g))
            d = corner_match_distance(q.astype(np.int32), best)
            offs = match_sides(best, q)
            g0 = _side_geom(np.asarray(best, float), 0)
            g1 = _side_geom(np.asarray(best, float), 1)
            rows.append({
                "id": iid, "d": round(d, 1),
                "offs": [None if o is None else round(o, 1) for o in offs],
                "hole_wh": [round(hr[1][0], 1), round(hr[1][1], 1)],
                "gt_wh": [round(g0[3], 1), round(g1[3], 1)],
            })
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] rows={len(rows)}")

    d = np.array([r["d"] for r in rows])
    print(f"\nrows={len(rows)}")
    for thr in (5, 8, 12, 20, 30):
        print(f"  corner dist <= {thr:3d}: {int((d <= thr).sum()):4d}")

    good = [r for r in rows if r["d"] <= 12]
    print(f"\nmatched (<=12 px): {len(good)}")
    off = np.array([[0.0 if o is None else o for o in r["offs"]] for r in good])
    for k in range(4):
        col = off[:, k]
        print(f"    side{k}: med={np.median(col):6.2f} p10={np.percentile(col,10):6.2f} "
              f"p90={np.percentile(col,90):6.2f}  |med|={np.median(np.abs(col)):5.2f}")
    # hole size vs GT size
    hw = np.array([r["hole_wh"] for r in good], float)
    gw = np.array([r["gt_wh"] for r in good], float)
    # order both so the long side comes first
    hsort = np.sort(hw, axis=1)[:, ::-1]
    gsort = np.sort(gw, axis=1)[:, ::-1]
    dl = gsort[:, 0] - hsort[:, 0]
    ds = gsort[:, 1] - hsort[:, 1]
    print(f"\n  long side: gt - hole  med={np.median(dl):.2f} "
          f"p10={np.percentile(dl,10):.2f} p90={np.percentile(dl,90):.2f}")
    print(f"  short side: gt - hole med={np.median(ds):.2f} "
          f"p10={np.percentile(ds,10):.2f} p90={np.percentile(ds,90):.2f}")

    out = os.path.join(ROOT, "benchmark", "scope45_offset2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()