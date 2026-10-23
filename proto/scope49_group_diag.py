"""LOOP-49: diagnose hole GROUPING (the "hole_split" FN bucket).

A box whose interior is crossed by another red stroke is split into several
holes.  Recovering the box means merging the pieces of ONE box - but two
touching boxes must never be merged.  Before tuning the grouping rule we need
hard numbers:

  * how many multi-hole groups exist at all
  * for each group, is the merged quad correct (<=5px to a GT), close (5..20)
    or simply wrong (>20)?
  * for every FN at D=5, is its box covered by a multi-hole group (grouping
    could fix it) or by a single hole (grouping cannot help)?

Usage: python proto/scope49_group_diag.py
"""
import os, sys, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, _shared_lines, _group_quad,
                           _red_support)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
DIST = 5.0
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)


def hole_in_poly(hole_pts, poly, shape):
    if len(hole_pts) == 0:
        return 0.0
    cnt = np.zeros(shape, np.uint8)
    cv2.fillPoly(cnt, [poly.astype(np.int32)], 1)
    pts = hole_pts.reshape(-1, 2).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, cnt.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, cnt.shape[0] - 1)
    return float(cnt[pts[:, 1], pts[:, 0]].mean())


def groups_of(infos):
    """Union-find over holes that share >= 2 (nearly) collinear side lines."""
    n = len(infos)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _shared_lines(infos[i][2], infos[j][2]) >= 2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    g = collections.OrderedDict()
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def best_gt_dist(q, gt):
    """min over GT of the 4-corner match distance (Hungarian within the pair)."""
    best = None
    for gj, g in enumerate(gt):
        d = corner_match_distance(q.astype(np.int32), g)
        if best is None or d < best[0]:
            best = (d, gj)
    return best


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    gstats = collections.Counter()
    gsize = collections.Counter()
    fn_reason = collections.Counter()
    fn_rows = []
    group_rows = []
    t0 = time.time()
    n_fn_total = 0

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        if P["denoise_min_area"] > 0:
            mask = denoise(mask, P["denoise_min_area"])

        holes = _holes_of(mask, CFG)
        infos = []
        for h in holes:
            r = _hole_side_lines(h, mask, CFG)
            if r is not None:
                infos.append((h, r[0], r[1]))
        if not infos:
            continue

        groups = groups_of(infos)

        # ---- group level: is the merged quad right?
        group_gt = []          # GT boxes touched by a group of size > 1
        for gidx in groups:
            if len(gidx) < 2:
                continue
            gsize[len(gidx)] += 1
            gq = _group_quad([infos[k][1] for k in gidx],
                             [infos[k][2] for k in gidx],
                             [cv2.contourArea(infos[k][0]) for k in gidx],
                             mask, CFG)
            if gq is None:
                gstats["no_quad"] += 1
                continue
            bd = best_gt_dist(gq, gt)
            d = bd[0] if bd else 1e9
            if d <= DIST:
                gstats["quad_ok"] += 1
            elif d <= 20:
                gstats["quad_near"] += 1
            else:
                gstats["quad_wrong"] += 1
            # which GT boxes do the individual pieces of this group cover?
            piece_gts = set()
            for k in gidx:
                for gj, g in enumerate(gt):
                    if hole_in_poly(infos[k][0], g, mask.shape) > 0.02:
                        piece_gts.add(gj)
            group_gt.extend(piece_gts)
            group_rows.append({"id": iid, "n": len(gidx), "d": round(d, 1),
                                "n_gt": len(piece_gts),
                                "gt": sorted(piece_gts)})

        multi_gt = set(group_gt)

        # ---- per GT: matched by a single-hole quad?  by a group quad?  by neither?
        dets = []
        for gidx in groups:
            if len(gidx) == 1:
                q = _hole_quad(infos[gidx[0]][0], mask, CFG)
                if q is not None:
                    dets.append(q)
            else:
                gq = _group_quad([infos[k][1] for k in gidx],
                                 [infos[k][2] for k in gidx],
                                 [cv2.contourArea(infos[k][0]) for k in gidx],
                                 mask, CFG)
                if gq is not None:
                    dets.append(gq)

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

        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            n_fn_total += 1
            ov = []
            for h, c, l in infos:
                f = hole_in_poly(h, g, mask.shape)
                if f > 0.02:
                    ov.append(f)
            if gi in multi_gt:
                reason = "fn_in_multi_group"
            elif ov:
                reason = "fn_single_hole"
            else:
                reason = "fn_no_hole"
            fn_reason[reason] += 1
            fn_rows.append({"id": iid, "gi": gi, "reason": reason,
                            "n_ov_holes": len(ov)})

        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] fn={n_fn_total} ({time.time()-t0:.0f}s)")

    print(f"\nmulti-hole groups: {sum(gsize.values())}  size hist={dict(gsize)}")
    print("group merged-quad quality:")
    for k, v in gstats.most_common():
        print(f"  {k:12s} {v:5d}")
    print(f"\nFN at D=5 (single-hole pipeline + group quads): {n_fn_total}")
    for k, v in fn_reason.most_common():
        print(f"  {k:20s} {v:5d} ({v*100.0/max(n_fn_total,1):5.1f}%)")

    out = os.path.join(ROOT, "benchmark", "scope49_group_diag.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"group_stats": dict(gstats), "group_size_hist": {str(k): v for k, v in gsize.items()},
                   "fn_reasons": dict(fn_reason), "fn_total": n_fn_total,
                   "groups": group_rows, "fn": fn_rows}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()