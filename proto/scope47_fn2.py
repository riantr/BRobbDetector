"""LOOP-47: forensics on the remaining false negatives of the HOLE pipeline.

pm_v7 misses 520/1121 GT boxes at D=5; the hole pipeline (scope45/46) brings
that down to ~475.  This script re-derives the mechanism for every remaining
FN so the next iteration targets the real bottleneck instead of guessing.

Categories
  no_red        the perimeter is not red in the input image at all
  weak_red      the perimeter is red only under loose thresholds
  no_hole       the box interior is NOT enclosed (broken outline, or red
                stuff crossing the interior)
  hole_split    the interior IS enclosed but split into several partial
                holes, so no single hole reproduces the box
  hole_partial  a hole overlaps the box but its quad is a sub-rectangle
  gate          a hole quad matched the GT but was rejected by the
                orthogonality / side-length gate
  matched       matched but lost in the dedup

Usage: python proto/scope47_fn2.py
"""
import os, sys, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _hole_quad,
                           _side_geom)
from scope43_fn_forensics import perim_points, PROD, RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
DIST = 5.0
LOOSE = {"sat_min": 60, "val_min": 80, "hue_max": 8}
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)


def coverage(pts, mask):
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return float(np.count_nonzero(mask[ys, xs])) / len(pts)


def hole_in_poly(hole_pts, poly, shape):
    """Fraction of the hole contour points that fall inside the polygon."""
    if len(hole_pts) == 0:
        return 0.0
    cnt = np.zeros(shape, np.uint8)
    cv2.fillPoly(cnt, [poly.astype(np.int32)], 1)
    pts = hole_pts.reshape(-1, 2).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, cnt.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, cnt.shape[0] - 1)
    return float(cnt[pts[:, 1], pts[:, 0]].mean())


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    reasons = collections.Counter()
    fn_rows = []
    tp = 0
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        loose = red_mask(img, LOOSE["sat_min"], LOOSE["val_min"], LOOSE["hue_max"])
        den = denoise(mask, P["denoise_min_area"])
        holes = _holes_of(den, CFG)
        hquads = []
        for h in holes:
            q = _hole_quad(h, den, CFG)
            hquads.append((h, q))
        dets = [q for _, q in hquads if q is not None]

        # which GT boxes are matched
        matched = set()
        cands = []
        for di, d in enumerate(dets):
            for gj, g in enumerate(gt):
                dd = corner_match_distance(d.astype(np.int32), g)
                if dd <= DIST:
                    cands.append((dd, di, gj))
        cands.sort()
        ud, ug = set(), set()
        for dd, di, gj in cands:
            if di in ud or gj in ug:
                continue
            ud.add(di); ug.add(gj); matched.add(gj)
        tp += len(matched)

        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            pts = perim_points(g)
            prod_cov = coverage(pts, mask)
            loose_cov = coverage(pts, loose)
            # holes overlapping the GT interior
            ov = []
            for h, q in hquads:
                frac = hole_in_poly(h, g, den.shape)
                if frac > 0.02:
                    d = None if q is None else round(
                        corner_match_distance(q.astype(np.int32), g), 1)
                    ov.append((round(frac, 2), d if d is not None else 1e9))
            ov.sort(key=lambda t: (-t[0], t[1]))
            ov = [(f, None if d >= 1e9 else d) for f, d in ov]
            best = ov[0] if ov else None
            row = {"id": iid, "gi": gi, "prod_cov": round(prod_cov, 3),
                   "loose_cov": round(loose_cov, 3), "n_holes_ov": len(ov),
                   "best_ov": best}
            if loose_cov < 0.25:
                row["reason"] = "no_red"
            elif prod_cov < 0.25:
                row["reason"] = "weak_red"
            elif not ov:
                row["reason"] = "no_hole"
            elif best[1] is not None and best[1] <= 20:
                row["reason"] = "hole_mismatch"
            elif len(ov) > 1:
                row["reason"] = "hole_split"
            else:
                row["reason"] = "hole_partial"
            reasons[row["reason"]] += 1
            fn_rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] tp={tp} fn={len(fn_rows)} ({time.time()-t0:.0f}s)")

    print(f"\nTP={tp}  FN={len(fn_rows)}")
    for k, v in reasons.most_common():
        print(f"  {k:16s} {v:5d} ({v*100.0/len(fn_rows):5.1f}%)")

    out = os.path.join(ROOT, "benchmark", "scope47_fn2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tp": tp, "fn": len(fn_rows),
                   "fn_reasons": dict(reasons), "fn_detail": fn_rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")

    # a few examples per reason
    buckets = collections.defaultdict(list)
    for r in fn_rows:
        buckets[r["reason"]].append(r)
    for reason, items in buckets.items():
        print(f"\n-- {reason} samples:")
        for r in items[:6]:
            print(f"   {r['id']} gt#{r['gi']} prod_cov={r['prod_cov']} "
                  f"n_ov={r['n_holes_ov']} best={r['best_ov']}")


if __name__ == "__main__":
    main()