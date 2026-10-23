"""Analyse the real false positives of the production pipeline.

Runs the production pipeline (proto/scope45_holes.py) over the 500 real
benchmark images, tags every detection with the family that produced it
(hole / edgefit), matches against the rule-consistent GT at D=5 and dumps the
feature distribution of

  * TP       (matched detections)
  * true_fp  (unmatched detections that are not "real unlabeled")
  * real_unl (unmatched detections that look like a real unannotated object)

Features per detection: min/max side, aspect, area, red support on the
denoised red mask, red fill ratio inside the quad, axis alignment.

Usage:
    python proto/analyze_real_fp.py            # stats + json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, iou_obb, load_expected
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect,
                           detect_holes, detect_edgefit, _red_support)
from eval_pointmatch import (match_boxes_point, is_real_unlabeled,
                             corner_match_distance)
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"
D = 5.0


def sides_of(q):
    s = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    return min(s), max(s)


def axis_aligned(q):
    for i in range(4):
        d = q[(i + 1) % 4] - q[i]
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return False
        ang = abs(np.degrees(np.arctan2(d[1], d[0]))) % 90.0
        if min(ang, 90.0 - ang) > 2.0:
            return False
    return True


def fill_ratio(mask, quad):
    """Red pixels inside the quad / quad area."""
    q = np.asarray(quad, float)
    x0 = int(max(0, np.floor(q[:, 0].min())))
    x1 = int(min(mask.shape[1], np.ceil(q[:, 0].max()) + 1))
    y0 = int(max(0, np.floor(q[:, 1].min())))
    y1 = int(min(mask.shape[0], np.ceil(q[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = mask[y0:y1, x0:x1]
    m = np.zeros_like(sub)
    cv2.fillPoly(m, [(q - np.array([x0, y0])).astype(np.int32)], 1)
    area = float((q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min()))
    if area < 1:
        return 0.0
    return float(((sub > 0) & (m > 0)).sum()) / area


def tag_family(dets, holes, edges):
    tags = []
    for d in dets:
        h = max([iou_obb(d, x) for x in holes], default=0.0)
        e = max([iou_obb(d, x) for x in edges], default=0.0)
        if h > 0.5 and e > 0.5:
            tags.append("both")
        elif h >= e:
            tags.append("hole")
        else:
            tags.append("edgefit")
    return tags


def matched_index(dets, gts, D):
    cand = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            dd = corner_match_distance(d, g)
            if dd <= D:
                cand.append((dd, di, gi))
    cand.sort()
    used_d, used_g = set(), set()
    for _dd, di, gi in cand:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
    return used_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark",
                                                  "real_fp_profile.json"))
    args = ap.parse_args()

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
        holes = [np.asarray(q, np.int32) for q in
                 detect_holes(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask)]
        edges = [np.asarray(q, np.int32) for q in
                 detect_edgefit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask)]
        tags = tag_family(dets, holes, edges)

        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        used = matched_index(dets, gts, D)

        tp_dets = [dets[i] for i in sorted(used)]
        for di, d in enumerate(dets):
            q = np.asarray(d, float)
            mn, mx = sides_of(q)
            area = float(cv2.contourArea(d.astype(np.int32)))
            rec = dict(iid=iid, family=tags[di], min_side=mn, max_side=mx,
                       aspect=mx / max(mn, 1e-6), area=area,
                       axis=axis_aligned(q),
                       support=_red_support(mask, d),
                       fill=fill_ratio(mask, d),
                       max_iou_tp=max([iou_obb(d, t) for t in tp_dets],
                                     default=0.0),
                       max_iou_gt=max([iou_obb(d, g) for g in gts],
                                      default=0.0))
            if di in used:
                rec["kind"] = "tp"
            elif is_real_unlabeled(d, exp.get(iid, 0), len(gts), wide):
                rec["kind"] = "real_unl"
            else:
                rec["kind"] = "true_fp"
            rows.append(rec)
        if (n + 1) % 100 == 0:
            print("  %d/%d" % (n + 1, len(ids)), flush=True)

    for r in rows:
        for k in ("min_side", "max_side", "aspect", "area", "support", "fill"):
            r[k] = round(float(r[k]), 4)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d detections)" % (args.out, len(rows)))

    for kind in ("tp", "true_fp", "real_unl"):
        s = [r for r in rows if r["kind"] == kind]
        if not s:
            continue
        fam = {}
        for r in s:
            fam[r["family"]] = fam.get(r["family"], 0) + 1
        print("\n== %s: %d" % (kind, len(s)))
        print("   family: %s" % fam)
        for k in ("min_side", "max_side", "aspect", "area", "support", "fill"):
            v = np.array([r[k] for r in s], float)
            print("   %-9s p10 %8.3f  p50 %8.3f  p90 %8.3f  mean %8.3f"
                  % (k, np.percentile(v, 10), np.percentile(v, 50),
                     np.percentile(v, 90), v.mean()))
        ax = sum(1 for r in s if r["axis"])
        print("   axis aligned: %d/%d" % (ax, len(s)))
        if kind != "tp":
            for thr in (0.05, 0.1, 0.2, 0.4):
                a = sum(1 for r in s if r["max_iou_tp"] >= thr)
                b = sum(1 for r in s if r["max_iou_gt"] >= thr)
                print("   overlaps a TP (iou>=%.2f): %3d   a GT: %3d"
                      % (thr, a, b))


if __name__ == "__main__":
    main()