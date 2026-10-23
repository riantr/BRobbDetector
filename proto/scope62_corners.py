"""LOOP-62: for the FNs that DO have a clean single hole, how far are the GT
corners from the hole contour and from the red stroke?

If the GT corners sit on the hole contour (or on the red stroke) the hole has
the right SHAPE and only the parameterisation (minAreaRect + straight side
fit) is losing the accuracy - then a corner based quad can recover them.
If the corners are far from the red stroke, the annotation itself is
different from the label and no fit can recover them.

Usage: python proto/scope62_corners.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, _holes_of, _hole_quad, _hole_side_lines)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    fn = json.load(open(os.path.join(ROOT, "benchmark", "scope47_fn2.json"),
                        encoding="utf-8"))["fn_detail"]
    rel = json.load(open(os.path.join(ROOT, "benchmark", "scope60_rel.json"),
                         encoding="utf-8"))["rows"]
    relmap = {(r["id"], r["gi"]): r for r in rel}
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
        for rec in rs:
            r0 = relmap.get((iid, rec["gi"]))
            if r0 is None or not r0.get("rel", "").startswith("contained_1"):
                continue
            g = np.asarray(gt[rec["gi"]], np.float64)
            # the hole that is inside the GT box
            best = None
            for h in holes:
                pts = h.reshape(-1, 2).astype(np.int32)
                m = np.zeros(mask.shape, np.uint8)
                cv2.fillPoly(m, [pts], 1)
                gm = np.zeros(mask.shape, np.uint8)
                cv2.fillPoly(gm, [g.astype(np.int32)], 1)
                f = float((m & gm).sum()) / max(float(m.sum()), 1.0)
                if f > 0.75:
                    if best is None or m.sum() > best[0]:
                        best = (m.sum(), h, pts)
            if best is None:
                continue
            h = best[1]
            cpts = h.reshape(-1, 2).astype(np.float64)
            dc = []
            for p in g:
                d = np.sqrt(((cpts - p[None, :]) ** 2).sum(1)).min()
                dc.append(round(float(d), 1))
            # distance to the nearest red pixel of the mask
            dr = []
            ys, xs = np.nonzero(mask)
            rp = np.stack([xs, ys], 1).astype(np.float64)
            for p in g:
                d = np.sqrt(((rp - p[None, :]) ** 2).sum(1)).min()
                dr.append(round(float(d), 1))
            q = _hole_quad(h, mask, CFG)
            dq = corner_match_distance(q.astype(np.int32), gt[rec["gi"]]) if q is not None else -1
            stat["n"] += 1
            stat["contour_ok"] += 1 if max(dc) <= 5 else 0
            stat["red_ok"] += 1 if max(dr) <= 3 else 0
            rows.append({"id": iid, "gi": rec["gi"], "reason": rec["reason"],
                         "d_quad": dq, "d_contour": dc, "d_red": dr})
    print("== contained_1 FNs ==")
    for k, v in stat.most_common():
        print(f"  {k:14s} {v:5d}")
    mx_c = collections.Counter(max(r["d_contour"]) for r in rows)
    mx_r = collections.Counter(max(r["d_red"]) for r in rows)
    print("  max corner distance to hole contour:",
          sorted((k, v) for k, v in mx_c.items())[:12])
    print("  max corner distance to red stroke   :",
          sorted((k, v) for k, v in mx_r.items())[:12])
    out = os.path.join(ROOT, "benchmark", "scope62_corners.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": dict(stat), "rows": rows}, f, ensure_ascii=False, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()