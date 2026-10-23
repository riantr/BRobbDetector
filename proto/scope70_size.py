"""LOOP-70: are the remaining FNs concentrated in a GEOMETRY class?

If the missed boxes are systematically flatter / smaller / more elongated than
the detected ones, the cause is a threshold (min_area, min_side, hole_min_side,
hole_fill) and can be fixed by re-tuning that threshold.  If the FN geometry
is the same as the TP geometry, the cause is structural (no enclosed
interior) and no threshold will help.

Usage: python proto/scope70_size.py
"""
import os, sys, json, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect,
                            evaluate, mask_cache)

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    cfg = dict(HOLE_DEFAULT)
    pipe = dict(PIPE_DEFAULT)
    from common import red_mask
    det_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        if iid not in mask_cache:
            mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                          np.ones((3, 3), np.uint8),
                                          iterations=3) > 0
        det_map[iid] = detect(img, P, cfg, pipe)
    # match
    tp = []
    fn = []
    for iid in ids:
        dets = det_map.get(iid, [])
        gts = gt_map[iid]
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
            gg = np.asarray(g, np.float64)
            sides = [float(np.linalg.norm(gg[(i + 1) % 4] - gg[i])) for i in range(4)]
            w = max(sides[0], sides[2])
            h = max(sides[1], sides[3])
            rec = {"id": iid, "gi": gi, "w": w, "h": h,
                   "short": min(w, h), "long": max(w, h),
                   "aspect": max(w, h) / max(min(w, h), 1e-6),
                   "area": abs(cv2.contourArea(gg.astype(np.float32)))}
            (tp if gi in ug else fn).append(rec)
    print(f"TP={len(tp)}  FN={len(fn)}")

    def desc(name, rows, key):
        v = sorted(r[key] for r in rows)
        q = lambda p: v[int(len(v) * p)]
        return (f"{name:5s} n={len(rows):4d} min={v[0]:8.1f} p10={q(.10):8.1f} "
                f"p25={q(.25):8.1f} med={q(.5):8.1f} p75={q(.75):8.1f} "
                f"p90={q(.90):8.1f} max={v[-1]:8.1f}")
    for key in ("short", "long", "aspect", "area"):
        print(desc("TP", tp, key))
        print(desc("FN", fn, key))
        print()
    # short side buckets
    print("-- by short side --")
    buckets = [(0, 12), (12, 16), (16, 20), (20, 30), (30, 50), (50, 100), (100, 1e9)]
    for lo, hi in buckets:
        t = sum(1 for r in tp if lo <= r["short"] < hi)
        f = sum(1 for r in fn if lo <= r["short"] < hi)
        print(f"  short in [{lo:4.0f},{hi:6.0f}):  TP={t:4d}  FN={f:4d}  "
              f"miss_rate={f/max(t+f,1):.2f}")
    print("-- by aspect --")
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 4), (4, 6), (6, 1e9)]:
        t = sum(1 for r in tp if lo <= r["aspect"] < hi)
        f = sum(1 for r in fn if lo <= r["aspect"] < hi)
        print(f"  aspect in [{lo:4.1f},{hi:6.1f}):  TP={t:4d}  FN={f:4d}  "
              f"miss_rate={f/max(t+f,1):.2f}")
    out = os.path.join(ROOT, "benchmark", "scope70_size.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tp": tp, "fn": fn}, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()