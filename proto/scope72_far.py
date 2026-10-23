"""LOOP-72: decompose the error of the 152 "far" FNs.

A "far" FN is a box whose hole WAS found, whose quad WAS built and passed
every gate - but the quad is more than 5px away from the GT.  The error can
be

  angle   the fitted side is rotated relative to the GT side
  offset  the fitted side is parallel but displaced
  scale   the fitted quad is bigger/smaller than the GT
  alien   the fitted quad is a completely different box (wrong hole)

This script measures, per FN, the angle and the perpendicular offset of the
fitted side vs the GT side, and whether the GT side itself is covered by red
pixels (i.e. whether the GT is drawable at all).

Usage: python proto/scope72_far.py
"""
import os, sys, json, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, _holes_of,
                            _quad_of, _gate, _enforce, _red_support, detect)
from scope43_fn_forensics import RAW as RAW2

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    cfg = dict(CFG)
    pipe = dict(PIPE)
    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gts = gt_map[iid]
        if not gts:
            continue
        H, W = img.shape[:2]
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        holes = _holes_of(mask, cfg)
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
        hmasks = []
        for h in holes:
            hm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(hm, [h.reshape(-1, 2).astype(np.int32)], 1)
            hmasks.append(hm)
        for gi, g in enumerate(gts):
            if gi in ug:
                continue
            gg = np.asarray(g, np.int32)
            gm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(gm, [gg], 1)
            best = None
            for hi, hm in enumerate(hmasks):
                inter = int((hm & (gm > 0)).sum())
                if inter == 0:
                    continue
                f = inter / max(int(gm.sum()), 1)
                if best is None or f > best[0]:
                    best = (f, hi)
            if best is None:
                continue
            f, hi = best
            q = _quad_of(holes[hi], mask, cfg)
            if q is None:
                continue
            q = _gate(np.asarray(q, float), W, H, P)
            if q is None:
                continue
            q = _enforce(q, pipe["ortho"])
            d = corner_match_distance(np.asarray(q, np.int32), gg)
            # per side comparison (fit side i vs GT side i)
            angs, offs = [], []
            for i in range(4):
                a = np.asarray(q[i], float)
                b = np.asarray(q[(i + 1) % 4], float)
                u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                n = np.array([-u[1], u[0]])
                ga = np.asarray(gg[i], float)
                gb = np.asarray(gg[(i + 1) % 4], float)
                gu = (gb - ga) / max(np.linalg.norm(gb - ga), 1e-9)
                angs.append(math.degrees(math.acos(float(np.clip(np.dot(u, gu), -1, 1)))))
                offs.append(float(np.dot(n, ga - a)))
            # red support along the GT sides themselves
            gt_sup = []
            for i in range(4):
                a = gg[i].astype(float)
                b = gg[(i + 1) % 4].astype(float)
                L = np.linalg.norm(b - a)
                ns = max(int(L), 2)
                ts = np.linspace(0, 1, ns)
                pts = a[None] + ts[:, None] * (b - a)[None]
                xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
                ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
                gt_sup.append(float((mask[ys, xs] > 0).mean()))
            rows.append({"id": iid, "gi": gi, "d": round(d, 1),
                         "cover": round(f, 2),
                         "ang": [round(x, 2) for x in angs],
                         "off": [round(x, 2) for x in offs],
                         "gt_sup": [round(x, 2) for x in gt_sup]})
    print(f"far FNs analysed: {len(rows)}")
    def med(key, f=None):
        if f is None:
            f = (lambda x: x) if not isinstance(rows[0][key], list) else max
        v = sorted(f(r[key]) for r in rows)
        return v[len(v) // 2]
    print(f"  d            median={med('d'):.1f}")
    print(f"  max|ang|     median={med('ang', max):.2f} deg")
    print(f"  max|off|     median={med('off', lambda x: max(abs(v) for v in x)):.2f} px")
    print(f"  min(gt_sup)  median={med('gt_sup', min):.2f}")
    print(f"  cover        median={med('cover'):.2f}")
    # how many FNs have a GT that is fully red-supported (drawable)?
    ok = [r for r in rows if min(r["gt_sup"]) >= 0.6]
    print(f"  GT fully red supported (min side sup>=0.6): {len(ok)}/{len(rows)}")
    for r in rows[:15]:
        print(f"   {r['id']} gt#{r['gi']} d={r['d']:6.1f} cov={r['cover']:.2f} "
              f"ang={r['ang']} off={r['off']} gtsup={r['gt_sup']}")
    out = os.path.join(ROOT, "benchmark", "scope72_far.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()