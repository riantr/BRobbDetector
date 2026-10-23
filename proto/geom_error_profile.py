"""Geometric-error profile of the production pipeline on the real data.

The D=5 point-match metric only counts a TP when *all four* corners are
within D px.  This script measures how far the detector is really off:

  * for every rule-consistent GT: the corner distance to the nearest
    detection  (how many GTs would be recovered at D=8/12/20/30 ...)
  * for every unmatched detection: the corner distance to the nearest GT
    (how many "trueFP" are really mis-localised annotations)
  * the signed per-corner bias of the detected quad (systematic offset)

Usage: python proto/geom_error_profile.py
"""
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, load_expected
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect)
from eval_pointmatch import corner_match_distance, is_real_unlabeled
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"
BINS = [0, 2, 5, 8, 12, 20, 30, 50, 100, 1e9]
LBLS = ["0-2", "2-5", "5-8", "8-12", "12-20", "20-30", "30-50",
        "50-100", ">100"]


def hist(vals):
    c = Counter()
    for v in vals:
        for i in range(len(BINS) - 1):
            if BINS[i] <= v < BINS[i + 1]:
                c[LBLS[i]] += 1
                break
    return c


def main():
    ids = get_ids()
    exp = load_expected()
    gt_d = []          # GT -> nearest detection distance
    det_d = []         # detection -> nearest GT distance
    det_kind = []
    bias = []          # signed per corner offset (det - gt) for matched pairs
    ngt = 0
    ndet = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        wide = cv2.dilate(__import__("common").red_mask(img, 80, 110, 10),
                          np.ones((3, 3), np.uint8), iterations=3) > 0
        ngt += len(gts)
        ndet += len(dets)
        for g in gts:
            if not dets:
                gt_d.append(1e9)
                continue
            gt_d.append(min(corner_match_distance(d, g) for d in dets))
        for d in dets:
            if not gts:
                det_d.append(1e9)
                det_kind.append("fp_nogt")
                continue
            dd = min(corner_match_distance(d, g) for g in gts)
            det_d.append(dd)
            if dd <= 5.0:
                det_kind.append("tp")
                # signed bias: match the det corners to the gt corners
                gi = int(np.argmin([corner_match_distance(d, g) for g in gts]))
                g = gts[gi]
                bias.append((np.asarray(d, float) - np.asarray(g, float)))
            elif is_real_unlabeled(d, exp.get(iid, 0), len(gts), wide):
                det_kind.append("real_unl")
            else:
                det_kind.append("true_fp")

    print("GT %d  det %d" % (ngt, ndet))
    hg = hist(gt_d)
    hd = hist([v for v, k in zip(det_d, det_kind) if k == "true_fp"])
    print("\nGT -> nearest detection corner distance")
    for l in LBLS:
        print("   %-6s %5d  (%.1f%%)" % (l, hg.get(l, 0), hg.get(l, 0) / max(ngt, 1) * 100))
    print("\ntrueFP -> nearest GT corner distance")
    for l in LBLS:
        print("   %-6s %5d" % (l, hd.get(l, 0)))
    print("\ncumulative GT coverage")
    for thr in (2, 3, 5, 8, 12, 20, 30, 50):
        k = sum(1 for v in gt_d if v <= thr)
        print("   D<=%-3d %5d  (%.1f%%)" % (thr, k, k / max(ngt, 1) * 100))
    if bias:
        b = np.array(bias)          # (n, 4, 2)
        mag = np.linalg.norm(b, axis=2)
        print("\nmatched-pair corner bias (det - gt), n=%d" % len(b))
        print("   |dx| mean %6.2f   |dy| mean %6.2f   |d| mean %6.2f  p90 %6.2f"
              % (np.abs(b[:, :, 0]).mean(), np.abs(b[:, :, 1]).mean(),
                 mag.mean(), np.percentile(mag, 90)))
        print("   dx per corner: %s" % np.round(b[:, :, 0].mean(axis=0), 2))
        print("   dy per corner: %s" % np.round(b[:, :, 1].mean(axis=0), 2))
        print("   per-corner |d| mean: %s"
              % np.round(mag.mean(axis=0), 2))
        print("   per-corner |d| p90 : %s"
              % np.round(np.percentile(mag, 90, axis=0), 2))

    with open(os.path.join(ROOT, "benchmark", "geom_error_profile.json"),
              "w", encoding="utf-8") as f:
        json.dump({"gt_nearest_det": [round(v, 2) for v in gt_d],
                   "det_kind": det_kind,
                   "det_nearest_gt": [round(v, 2) for v in det_d]}, f,
                  ensure_ascii=False)
    print("\nsaved benchmark/geom_error_profile.json")


if __name__ == "__main__":
    main()