"""Systematic geometry bias of the production pipeline (real data).

For every matched pair (detection, GT) at D=5 the optimal corner assignment is
used to measure

  * the signed per-corner offset (det - gt)
  * the signed offset of every side (positive = the detection lies OUTSIDE
    the GT side)
  * the area / side-length ratio (det vs gt)

A systematic expansion would mean the quad has to be shrunk, a systematic
shrink means it has to be grown.

Usage: python proto/geom_bias.py
"""
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"
PERMS = list(itertools.permutations(range(4)))


def best_perm(det, gt):
    best, bp = 1e18, None
    for perm in PERMS:
        d = np.linalg.norm(det - gt[list(perm)], axis=1).max()
        if d < best:
            best, bp = d, perm
    return best, bp


def side_offset(det, gt, perm):
    """Signed distance of every det side from the corresponding gt side."""
    out = []
    for i in range(4):
        a = gt[perm[i]].astype(float)
        b = gt[perm[(i + 1) % 4]].astype(float)
        u = b - a
        L = float(np.linalg.norm(u))
        if L < 1e-6:
            continue
        u = u / L
        n = np.array([-u[1], u[0]])
        c = det.astype(float).mean(axis=0)
        out.append(float(np.dot(c - a, n)))
    return out


def main():
    ids = get_ids()
    cor = []
    sides = []
    ratios = []
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        # greedy match at D=5

        from eval_pointmatch import corner_match_distance
        cand = [(corner_match_distance(dets[di], gts[gi]), di, gi)
                for di in range(len(dets)) for gi in range(len(gts))]
        cand = [c for c in cand if c[0] <= 5.0]
        cand.sort()
        ud, ug = set(), set()
        for dd, di, gi in cand:
            if di in ud or gi in ug:
                continue
            ud.add(di)
            ug.add(gi)
            d = dets[di].astype(float)
            g = gts[gi].astype(float)
            _, perm = best_perm(d, g)
            cor.append(d - g[list(perm)])
            sides.append(side_offset(d, g, perm))
            da = cv2.contourArea(dets[di].astype(np.float32))
            ga = cv2.contourArea(gts[gi].astype(np.float32))
            ratios.append(da / max(ga, 1e-6))
        if (n + 1) % 100 == 0:
            print("  %d/%d" % (n + 1, len(ids)), flush=True)

    cor = np.array(cor)
    print("\nmatched pairs: %d" % len(cor))
    print("per-corner signed offset (det - gt), mean:")
    print("   dx %s" % np.round(cor[:, :, 0].mean(axis=0), 2))
    print("   dy %s" % np.round(cor[:, :, 1].mean(axis=0), 2))
    print("   |d| mean per corner %s"
          % np.round(np.linalg.norm(cor, axis=2).mean(axis=0), 2))
    print("   |dx| mean %.2f  |dy| mean %.2f"
          % (np.abs(cor[:, :, 0]).mean(), np.abs(cor[:, :, 1]).mean()))
    s = np.array(sides)
    print("\nside offset (positive = detection outside the GT):")
    print("   per side mean %s" % np.round(s.mean(axis=0), 2))
    print("   per side median %s" % np.round(np.median(s, axis=0), 2))
    print("   overall mean %.2f  median %.2f" % (s.mean(), np.median(s)))
    r = np.array(ratios)
    print("\narea ratio det/gt: mean %.3f median %.3f p10 %.3f p90 %.3f"
          % (r.mean(), np.median(r), np.percentile(r, 10), np.percentile(r, 90)))

    with open(os.path.join(ROOT, "benchmark", "geom_bias.json"), "w",
              encoding="utf-8") as f:
        json.dump({"corner": cor.round(2).tolist(),
                   "side": s.round(2).tolist(),
                   "area_ratio": r.round(3).tolist()}, f, ensure_ascii=False)
    print("saved benchmark/geom_bias.json")


if __name__ == "__main__":
    main()