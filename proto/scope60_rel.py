"""LOOP-60: classify every remaining FN by the RELATION between the GT box
and the hole regions of the image.

  contained  - a hole lies INSIDE the GT box  (its interior is cut, or the
               hole is only a part of the box)          -> split / partial
  merged     - a hole COVERS the GT box     (its interior is connected to a
               neighbour's interior through a gap)       -> merged
  none       - no hole is related to the GT box at all
  accurate   - a hole matches the GT box but its quad is >5px off

For "contained" cases the number of holes inside the GT box says whether the
interior is cut into pieces (>=2) or is a single clean hole (1).

Usage: python proto/scope60_rel.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, _red_support)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)


def poly_area(poly):
    return abs(cv2.contourArea(poly.astype(np.float32)))


def frac_in(mask_small, poly):
    """Fraction of a filled polygon that is set in mask_small."""
    tot = int(mask_small.sum())
    if tot == 0:
        return 0.0
    return float((mask_small & (poly > 0)).sum()) / tot


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    fn = json.load(open(os.path.join(ROOT, "benchmark", "scope47_fn2.json"),
                        encoding="utf-8"))["fn_detail"]
    byid = collections.defaultdict(list)
    for r in fn:
        byid[r["id"]].append(r)

    stat = collections.Counter()
    rows = []
    for iid in ids:
        rs = byid.get(iid)
        if not rs:
            continue
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
            inside = []          # holes inside the GT box
            covering = []        # holes covering the GT box
            best = None
            for h in holes:
                hm = np.zeros((H, W), np.uint8)
                pts = h.reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(hm, [pts], 1)
                f_h_in_g = frac_in(hm, gm)      # hole area inside the GT
                f_g_in_h = frac_in(gm, hm)      # GT area inside the hole
                q = _hole_quad(h, mask, CFG)
                d = corner_match_distance(q.astype(np.int32), g) if q is not None else 1e9
                if best is None or d < best[0]:
                    best = (d, q, f_h_in_g, f_g_in_h)
                if f_h_in_g > 0.75:
                    inside.append((f_g_in_h, d))
                if f_g_in_h > 0.5:
                    covering.append((f_h_in_g, d))
            if best is None:
                stat["no_hole"] += 1
                rows.append({"id": iid, "gi": rec["gi"], "rel": "no_hole"})
                continue
            d, q, f_h_in_g, f_g_in_h = best
            if inside:
                n_in = len(inside)
                key = f"contained_{min(n_in,4)}"
            elif covering:
                key = "merged"
            else:
                key = "no_hole"
            stat[key] += 1
            stat[f"{key}__{rec['reason']}"] += 1
            rows.append({"id": iid, "gi": rec["gi"], "reason": rec["reason"],
                         "rel": key, "d": round(d, 1),
                         "f_h_in_g": round(f_h_in_g, 2),
                         "f_g_in_h": round(f_g_in_h, 2),
                         "n_inside": len(inside),
                         "support": round(_red_support(mask, q), 2) if q is not None else 0})

    print("== FN relation to hole regions ==")
    for k, v in sorted(stat.items()):
        print(f"  {k:28s} {v:5d}")
    out = os.path.join(ROOT, "benchmark", "scope60_rel.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": dict(stat), "rows": rows}, f, ensure_ascii=False, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()