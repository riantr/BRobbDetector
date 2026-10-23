"""LOOP-68: why are the `contained_1` FNs off by 5-12 px?

`contained_1` = exactly ONE hole lies inside the GT box and the hole covers
~83% of the GT area - i.e. the hole IS the box interior, shrunk by the stroke
half width.  The production fit expands it back to the stroke centre, yet
the resulting quad is still >5px away from the GT for 74 boxes (median 11.8).

This script decomposes the error for every contained_1 FN:
  d_rect  - corner distance of the hole's plain minAreaRect
  d_fit   - corner distance of the production quad (red band fit)
  ang     - angle between the fitted side and the GT side
  scale   - fitted side length / GT side length
  off     - signed distance from the GT side to the fitted line

Usage: python proto/scope68_c1.py
"""
import os, sys, json, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, _holes_of, _hole_quad, _order_quad,
                           _side_geom, _fit_side_line, _line_isect)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)


def side_lengths(q):
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    rel = json.load(open(os.path.join(ROOT, "benchmark", "scope60_rel.json"),
                         encoding="utf-8"))["rows"]
    want = collections.defaultdict(list)
    for r in rel:
        if r["rel"] == "contained_1":
            want[r["id"]].append(r)
    rows = []
    for iid, rs in want.items():
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        holes = _holes_of(mask, CFG)
        H, W = mask.shape
        for rec in rs:
            g = np.asarray(gt[rec["gi"]], np.int32)
            gm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(gm, [g], 1)
            # the hole that lies inside the GT
            best = None
            for h in holes:
                hm = np.zeros((H, W), np.uint8)
                cv2.fillPoly(hm, [h.reshape(-1, 2).astype(np.int32)], 1)
                f_h_in_g = float((hm & (gm > 0)).sum()) / max(int(hm.sum()), 1)
                if f_h_in_g <= 0.75:
                    continue
                q = _hole_quad(h, mask, CFG)
                d = corner_match_distance(q.astype(np.int32), g) if q is not None else 1e9
                if best is None or d < best[0]:
                    best = (d, q, h)
            if best is None:
                continue
            d, q, h = best
            if q is None:
                rows.append({"id": iid, "gi": rec["gi"], "d": None})
                continue
            rect = cv2.minAreaRect(h)
            qrect = _order_quad(cv2.boxPoints(rect).astype(float))
            d_rect = corner_match_distance(qrect.astype(np.int32), g)
            gs = side_lengths(g)
            qs = side_lengths(q)
            # match sides: find the permutation of the fitted quad that best
            # matches the GT (same order as _order_quad: 0,1,2,3 around)
            # compare the side DIRECTIONS
            angs = []
            offs = []
            for i in range(4):
                a, u, n, L = _side_geom(q, i)
                ga, gu, gn, gL = _side_geom(g.astype(float), i)
                angs.append(math.degrees(math.acos(
                    float(np.clip(np.dot(u, gu), -1, 1)))))
                offs.append(float(np.dot(n, ga - a)))
            rows.append({"id": iid, "gi": rec["gi"],
                         "d": round(d, 1), "d_rect": round(d_rect, 1),
                         "gt_side": [round(x, 1) for x in gs],
                         "fit_side": [round(x, 1) for x in qs],
                         "aspect": round(max(gs) / max(min(gs), 1e-6), 2),
                         "ang": [round(x, 2) for x in angs],
                         "off": [round(x, 2) for x in offs],
                         "area_ratio": round((qs[0] * qs[2]) / max(gs[0] * gs[2], 1e-6), 3)})
    ok = [r for r in rows if r.get("d") is not None]
    print(f"contained_1 analysed: {len(ok)}")
    dd = sorted(r["d"] for r in ok)
    dr = sorted(r["d_rect"] for r in ok)
    print(f"  d_fit  : med={dd[len(dd)//2]:.1f}  <=8:{sum(1 for x in dd if x<=8)}"
          f"  <=12:{sum(1 for x in dd if x<=12)}  <=20:{sum(1 for x in dd if x<=20)}")
    print(f"  d_rect : med={dr[len(dr)//2]:.1f}  <=8:{sum(1 for x in dr if x<=8)}"
          f"  <=12:{sum(1 for x in dr if x<=12)}  <=20:{sum(1 for x in dr if x<=20)}")
    print(f"  d_rect < d_fit for {sum(1 for r in ok if r['d_rect'] < r['d'])} cases")
    print(f"  aspect med={sorted(r['aspect'] for r in ok)[len(ok)//2]:.2f}")
    print(f"  max|ang| med={sorted(max(r['ang']) for r in ok)[len(ok)//2]:.2f} deg")
    print(f"  area_ratio med={sorted(r['area_ratio'] for r in ok)[len(ok)//2]:.3f}")
    print(f"  max|off| med={sorted(max(abs(x) for x in r['off']) for r in ok)[len(ok)//2]:.2f} px")
    # error decomposition: is the fit a pure translation / scale / rotation?
    cnt = collections.Counter()
    for r in ok:
        a = r["aspect"]
        if a >= 4:
            cnt["elongated(>=4)"] += 1
        elif a >= 2:
            cnt["medium(2-4)"] += 1
        else:
            cnt["compact(<2)"] += 1
    print("  aspect classes:", dict(cnt))
    for r in sorted(ok, key=lambda r: r["d"])[:20]:
        print(f"   {r['id']} gt#{r['gi']} d={r['d']:6.1f} drect={r['d_rect']:6.1f} "
              f"aspect={r['aspect']:5.2f} ang={r['ang']} off={r['off']} "
              f"ar={r['area_ratio']}")
    out = os.path.join(ROOT, "benchmark", "scope68_c1.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()