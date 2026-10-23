"""Why are the real false positives false?  -> hole rectangularity.

The red annotation is not only the box: it also contains the class label text
("液体 0.95") and a leader arrow that points from the text to the object.
Where the arrow touches a box corner the hole of the red mask is no longer a
rectangle - it is a rectangle with a triangular spike - and the pipeline still
emits a (forced) rectangle for it.  Those are the false positives.

This script measures, for every emitted detection, how well the source hole
fills the emitted quad:

    hole_fill = area(hole) / area(quad)

A real box is ~1 - 2*stroke/side, a spiky hole is far below that.

Usage: python proto/fp_rectangularity.py
"""
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, iou_obb, load_expected
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect,
                           _holes_of, _quad_of, _enforce, _gate, _on_edge)
from eval_pointmatch import corner_match_distance, is_real_unlabeled
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"
D = 5.0


def hole_quads(mask, cfg, pipe, p, W, H):
    """Quad + fill ratio for every hole that survives the pipeline gates."""
    out = []
    for h in _holes_of(mask, cfg):
        area = cv2.contourArea(h)
        if area < 20:
            continue
        q = _quad_of(h, mask, cfg)
        if q is None:
            continue
        q = _enforce(q, pipe["ortho"])
        if _on_edge(q, W, H, p):
            continue
        qa = float(cv2.contourArea(q.astype(np.float32)))
        out.append((np.asarray(q, np.int32), area, qa,
                    area / max(qa, 1e-6)))
    return out


def main():
    ids = get_ids()
    exp = load_expected()
    rows = []
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        if P["denoise_min_area"] > 0:
            mask = denoise(mask, P["denoise_min_area"])
        wide = cv2.dilate(red_mask(img, 80, 110, 10),
                          np.ones((3, 3), np.uint8), iterations=3) > 0
        dets = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask)]
        hq = hole_quads(mask, HOLE_DEFAULT, PIPE_DEFAULT, P, W, H)
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        for d in dets:
            best = None
            for q, area, qa, fill in hq:
                v = iou_obb(d, q)
                if best is None or v > best[0]:
                    best = (v, area, qa, fill)
            fill = best[3] if best and best[0] > 0.3 else None
            cd = min([corner_match_distance(d, g) for g in gts], default=1e9)
            if cd <= D:
                kind = "tp"
            elif is_real_unlabeled(d, exp.get(iid, 0), len(gts), wide):
                kind = "real_unl"
            else:
                kind = "true_fp"
            rows.append(dict(iid=iid, kind=kind, fill=fill, cd=round(cd, 2)))
        if (n + 1) % 100 == 0:
            print("  %d/%d" % (n + 1, len(ids)), flush=True)

    with open(os.path.join(ROOT, "benchmark", "real_fp_fill.json"),
              "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    for kind in ("tp", "true_fp", "real_unl"):
        v = np.array([r["fill"] for r in rows
                      if r["kind"] == kind and r["fill"] is not None], float)
        if len(v) == 0:
            continue
        print("\n== %s (n=%d)" % (kind, len(v)))
        print("   hole_fill p5 %.3f  p10 %.3f  p25 %.3f  p50 %.3f  p75 %.3f  p90 %.3f"
              % tuple(np.percentile(v, [5, 10, 25, 50, 75, 90])))
    # separability
    tp = np.array([r["fill"] for r in rows
                   if r["kind"] == "tp" and r["fill"] is not None], float)
    fp = np.array([r["fill"] for r in rows
                   if r["kind"] == "true_fp" and r["fill"] is not None], float)
    print("\nthreshold sweep (drop trueFP whose fill < thr, TP lost)")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        lost = int((tp < thr).sum())
        killed = int((fp < thr).sum())
        print("   thr %.2f  TP lost %3d   trueFP killed %3d   net %+d"
              % (thr, lost, killed, killed - lost))


if __name__ == "__main__":
    main()