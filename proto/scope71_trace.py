"""LOOP-71: trace every FN through the pipeline stages.

For each GT box the detector never matches, find out WHERE it dies:

  no_hole    the interior of the box is not an enclosed hole region
  quad_none  a hole exists but _hole_quad() refused it
             (hole_fill / hole_min_side / side fit)
  gate       the quad was built but the orthogonality / size gate rejected it
  support    the quad was built and gated but the red support was too low
  dedup      the quad was fine but a competing quad with a higher IoU won
  far        the quad survived but is >5px away from the GT

Usage: python proto/scope71_trace.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from cv_hybrid import dedup_keep_order
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, _holes_of,
                            _quad_of, _gate, _enforce, _red_support, detect,
                            detect_edgefit)
from scope43_fn_forensics import RAW as RAW2
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    cfg = dict(CFG)
    pipe = dict(PIPE)
    stage = collections.Counter()
    by_size = collections.defaultdict(collections.Counter)
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
        holes = _holes_of(mask, cfg)
        dets = detect(img, P, cfg, pipe)
        # GT matched by the pipeline?
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
        # hole regions
        hmasks = []
        for h in holes:
            hm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(hm, [h.reshape(-1, 2).astype(np.int32)], 1)
            hmasks.append(hm)
        for gi, g in enumerate(gts):
            if gi in ug:
                stage["tp"] += 1
                continue
            gg = np.asarray(g, np.int32)
            gm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(gm, [gg], 1)
            # the hole that overlaps the GT the most
            best = None
            for hi, hm in enumerate(hmasks):
                inter = int((hm & (gm > 0)).sum())
                if inter == 0:
                    continue
                f = inter / max(int(gm.sum()), 1)
                if best is None or f > best[0]:
                    best = (f, hi)
            if best is None:
                stage["no_hole"] += 1
                by_size["no_hole"][_sz(gg)] += 1
                rows.append({"id": iid, "gi": gi, "stage": "no_hole"})
                continue
            f, hi = best
            h = holes[hi]
            q = _quad_of(h, mask, cfg)
            if q is None:
                stage["quad_none"] += 1
                by_size["quad_none"][_sz(gg)] += 1
                rows.append({"id": iid, "gi": gi, "stage": "quad_none", "cover": round(f, 2)})
                continue
            q2 = _gate(np.asarray(q, float), W, H, P)
            if q2 is None:
                stage["gate"] += 1
                by_size["gate"][_sz(gg)] += 1
                rows.append({"id": iid, "gi": gi, "stage": "gate", "cover": round(f, 2)})
                continue
            q3 = _enforce(q2, pipe["ortho"])
            sup = _red_support(mask, q3)
            if sup < cfg["min_red_support"]:
                stage["support"] += 1
                by_size["support"][_sz(gg)] += 1
                rows.append({"id": iid, "gi": gi, "stage": "support",
                             "sup": round(sup, 2), "cover": round(f, 2)})
                continue
            # was it deduped away?
            merged = dedup_keep_order([np.asarray(x, np.int32) for x in dets],
                                       pipe["dedup_iou"])
            d = corner_match_distance(np.asarray(q3, np.int32), gg)
            stage["far" if d > 5.0 else "tp_but_dedup"] += 1
            by_size["far" if d > 5.0 else "tp_but_dedup"][_sz(gg)] += 1
            rows.append({"id": iid, "gi": gi, "stage": "far" if d > 5 else "dedup",
                         "d": round(d, 1), "cover": round(f, 2), "sup": round(sup, 2)})
    print("== FN stage breakdown ==")
    for k, v in stage.most_common():
        print(f"  {k:16s} {v:5d}")
    print("\n== by GT size class ==")
    for cls in ("tiny", "small", "medium", "large"):
        tot = sum(by_size[k][cls] for k in by_size)
        if not tot:
            continue
        print(f"  {cls}:")
        for k, v in by_size.items():
            if by_size[k][cls]:
                print(f"     {k:16s} {by_size[k][cls]:5d}")
    out = os.path.join(ROOT, "benchmark", "scope71_trace.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stage": dict(stage), "rows": rows}, f, ensure_ascii=False, indent=1)
    print("Saved", out)


def _sz(g):
    gg = np.asarray(g, np.float64)
    sides = [float(np.linalg.norm(gg[(i + 1) % 4] - gg[i])) for i in range(4)]
    s = min(max(sides[0], sides[2]), max(sides[1], sides[3]))
    if s < 20:
        return "tiny"
    if s < 30:
        return "small"
    if s < 50:
        return "medium"
    return "large"


if __name__ == "__main__":
    main()