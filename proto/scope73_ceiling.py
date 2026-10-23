"""LOOP-73: what is the CEILING of the detector?

LOOP-72 found that the GT boxes of the misses have a median red support of
only 0.24 along their own sides (and only 11/164 are fully red supported).
If the GT rectangle is not drawn in the image, no detector can find it.

This script cross-tabulates, for ALL 1121 GT boxes:
  * was it matched by the current pipeline (TP / FN)?
  * how much red is on its own sides (support)?

The result gives the practical ceiling: the number of GT boxes that are
actually drawn in red.

Usage: python proto/scope73_ceiling.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect)
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def side_support(mask, g):
    H, W = mask.shape
    sups = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        L = float(np.linalg.norm(b - a))
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None] + ts[:, None] * (b - a)[None]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        sups.append(float((mask[ys, xs] > 0).mean()))
    return sups


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    cfg = dict(HOLE_DEFAULT)
    pipe = dict(PIPE_DEFAULT)
    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gts = gt_map[iid]
        if not gts:
            continue
        H, W = img.shape[:2]
        # a rectangle that touches the image edge is not a valid rectangle
        # (2026-10-21 rule) -> it is not a valid target either
        gts = [g for g in gts if not quad_on_boundary(g, W, H, 0.0)]
        if not gts:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        dets = detect(img, P, cfg, pipe)
        pairs = []
        for di in range(len(dets)):
            for gj in range(len(gts)):
                d = corner_match_distance(dets[di], np.asarray(gts[gj], np.int32))
                if d <= 5.0:
                    pairs.append((d, di, gj))
        pairs.sort()
        ud, ug = set(), set()
        for d, di, gj in pairs:
            if di in ud or gj in ug:
                continue
            ud.add(di)
            ug.add(gj)
        for gi, g in enumerate(gts):
            sups = side_support(mask, g)
            rows.append({"id": iid, "gi": gi, "tp": gi in ug,
                         "sup_mean": round(float(np.mean(sups)), 3),
                         "sup_min": round(float(np.min(sups)), 3),
                         "sup": [round(s, 3) for s in sups]})
    tp = [r for r in rows if r["tp"]]
    fn = [r for r in rows if not r["tp"]]
    print(f"GT={len(rows)}  TP={len(tp)}  FN={len(fn)}")
    print("\n-- red support along the GT's own sides --")
    for name, rs in (("TP", tp), ("FN", fn)):
        for key in ("sup_mean", "sup_min"):
            v = sorted(r[key] for r in rs)
            q = lambda p: v[int(len(v) * p)]
            print(f"  {name} {key:9s}: p10={q(.1):.2f} p25={q(.25):.2f} med={q(.5):.2f} "
                  f"p75={q(.75):.2f} p90={q(.9):.2f}")
    print("\n-- cross tab: min side support bucket --")
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        t = sum(1 for r in tp if lo <= r["sup_min"] < hi)
        f = sum(1 for r in fn if lo <= r["sup_min"] < hi)
        print(f"  min_sup in [{lo:.1f},{hi:.1f}):  TP={t:4d}  FN={f:4d}  "
              f"miss={f/max(t+f,1):.2f}")
    print("\n-- cross tab: mean side support bucket --")
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        t = sum(1 for r in tp if lo <= r["sup_mean"] < hi)
        f = sum(1 for r in fn if lo <= r["sup_mean"] < hi)
        print(f"  mean_sup in [{lo:.1f},{hi:.1f}):  TP={t:4d}  FN={f:4d}  "
              f"miss={f/max(t+f,1):.2f}")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        ceil = sum(1 for r in rows if r["sup_min"] >= th)
        got = sum(1 for r in rows if r["tp"] and r["sup_min"] >= th)
        print(f"  ceiling(sup_min>={th}): {ceil:4d} drawable GT, detected {got:4d} "
              f"({got/max(ceil,1):.2%})")
    out = os.path.join(ROOT, "benchmark", "scope73_ceiling.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()