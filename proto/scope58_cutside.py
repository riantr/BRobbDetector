"""LOOP-58: for every remaining FN, look at the hole that covers the GT box
and measure the four sides' "red continuation beyond the corner".

A TRUE box side stops at the corner.  The side that CUTS a box interior
(a crossing stroke of a neighbouring box) keeps going past the corner, so
the fragment's cut side is the one with a long red continuation.  If that
signal is reliable it gives us a targeted repair: replace the cut side by
the next parallel stroke further out.

Reports for every FN:
  * best hole quad vs GT: is the hole bigger or smaller than the GT?
  * per-side red continuation length (px) beyond each corner
  * how many holes overlap the GT

Usage: python proto/scope58_cutside.py [--limit N]
"""
import os, sys, json, collections, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, _side_geom, _red_support)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)
EXT = 20.0
HALF = 2.0


def side_continuation(mask, a, u, n, L, ext=EXT, half=HALF):
    """Red run length beyond the start and beyond the end of a side."""
    H, W = mask.shape
    ts = np.arange(-ext, L + ext + 0.5, 1.0)
    pts = a[None, :] + ts[:, None] * u[None, :]
    hit = np.zeros(len(ts), bool)
    for o in (-half, 0.0, half):
        pp = pts + o * n[None, :]
        xs = np.clip(np.round(pp[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pp[:, 1]).astype(int), 0, H - 1)
        hit |= mask[ys, xs] > 0
    # run before t=0
    pre = 0
    for k in range(len(ts) - 1, -1, -1):
        if ts[k] >= 0:
            break
        if hit[k]:
            pre += 1
        else:
            break
    post = 0
    for k in range(len(ts)):
        if ts[k] <= L:
            continue
        if hit[k]:
            post += 1
        else:
            break
    return pre, post


def hole_in_poly(hole_pts, poly, shape):
    if len(hole_pts) == 0:
        return 0.0
    cnt = np.zeros(shape, np.uint8)
    cv2.fillPoly(cnt, [poly.astype(np.int32)], 1)
    pts = hole_pts.reshape(-1, 2).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, cnt.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, cnt.shape[0] - 1)
    return float(cnt[pts[:, 1], pts[:, 0]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ids = get_ids()
    if args.limit:
        ids = ids[:args.limit]
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
        infos = []
        for h in holes:
            r = _hole_side_lines(h, mask, CFG)
            if r is not None:
                infos.append((h, r[0], r[1]))
        for rec in rs:
            g = np.asarray(gt[rec["gi"]], np.float64)
            best = None
            for h, corners, lines in infos:
                q = _hole_quad(h, mask, CFG)
                if q is None:
                    continue
                d = corner_match_distance(q.astype(np.int32), gt[rec["gi"]])
                if best is None or d < best[0]:
                    best = (d, q, corners, lines, h)
            if best is None:
                stat["no_hole_at_all"] += 1
                continue
            d, q, corners, lines, h = best
            # size relation: hole area vs GT area
            ga = abs(cv2.contourArea(g.astype(np.float32)))
            ha = abs(cv2.contourArea(q.astype(np.float32)))
            rel = "bigger" if ha > ga * 1.15 else ("smaller" if ha < ga * 0.85 else "same")
            # per-side continuation
            cont = []
            for i in range(4):
                gg = _side_geom(corners, i)
                if gg is None:
                    cont.append((-1, -1))
                    continue
                a, u, n, L = gg
                cont.append(side_continuation(mask, a, u, n, L))
            mx = max(max(c) for c in cont)
            stat[f"rel_{rel}"] += 1
            stat[f"cut_{'yes' if mx >= 8 else 'no'}"] += 1
            rows.append({"id": iid, "gi": rec["gi"], "reason": rec["reason"],
                         "d": round(d, 1), "rel": rel, "max_cont": mx,
                         "cont": cont, "n_holes": len(infos),
                         "area_ratio": round(ha / max(ga, 1e-6), 2)})

    print("== FN hole statistics ==")
    for k, v in sorted(stat.items()):
        print(f"  {k:20s} {v:5d}")
    print("\n== per-reason size relation ==")
    c2 = collections.Counter((r["reason"], r["rel"]) for r in rows)
    for k, v in sorted(c2.items()):
        print(f"  {k} {v}")
    print("\n== cut-signal vs size relation ==")
    c3 = collections.Counter((r["rel"], r["max_cont"] >= 8) for r in rows)
    for k, v in sorted(c3.items()):
        print(f"  {k} {v}")
    out = os.path.join(ROOT, "benchmark", "scope58_cutside.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": dict(stat), "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()