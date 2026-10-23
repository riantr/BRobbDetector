"""Why do the 149 real true false-positives happen?

135 of the 149 trueFP overlap an annotated GT box (IoU >= 0.05) and 84 of
them reach IoU >= 0.4 with it.  So the pipeline emits a *second* rectangle
for an object that was annotated once.  This script pairs every trueFP with
the GT it overlaps and with the TP detection that owns that GT, and prints the
feature differences, so that a suppression rule can be designed.

Usage: python proto/why_fp.py [--viz 16]
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
                           _red_support)
from eval_pointmatch import corner_match_distance, is_real_unlabeled
from quad_ortho import quad_on_boundary
from analyze_real_fp import (sides_of, axis_aligned, fill_ratio,
                             matched_index, tag_family)

ROOT = r"D:\src\CodeArts\BRobbDetector"
D = 5.0
TOL = 25.0     # "near miss" radius used for pairing


def near(a, b, tol):
    return corner_match_distance(a, b) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark",
                                                  "real_fp_pairs.json"))
    args = ap.parse_args()

    ids = get_ids()
    exp = load_expected()
    pairs = []
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
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        used = matched_index(dets, gts, D)
        for di, d in enumerate(dets):
            if di in used:
                continue
            if is_real_unlabeled(d, exp.get(iid, 0), len(gts), wide):
                continue
            # nearest GT by corner distance
            best = None
            for gi, g in enumerate(gts):
                cd = corner_match_distance(d, g)
                if best is None or cd < best[0]:
                    best = (cd, gi, g)
            if best is None or best[0] > TOL:
                pairs.append(dict(iid=iid, di=di, kind="lonely"))
                continue
            cd, gi, g = best
            # the TP that owns that GT
            owner = None
            for dj, dj_ in enumerate(dets):
                if dj == di:
                    continue
                if corner_match_distance(dj_, g) <= D:
                    owner = dj
                    break
            rec = dict(iid=iid, di=di, gt=gi, cd=round(cd, 2),
                       iou_gt=round(iou_obb(d, g), 3))
            q = np.asarray(d, float)
            mn, mx = sides_of(q)
            rec.update(min_side=round(mn, 1), max_side=round(mx, 1),
                       area=round(float(cv2.contourArea(d.astype(np.int32))), 1),
                       support=round(_red_support(mask, d), 3),
                       fill=round(fill_ratio(mask, d), 4),
                       axis=axis_aligned(q))
            if owner is not None:
                t = dets[owner]
                tq = np.asarray(t, float)
                tmn, tmx = sides_of(tq)
                rec["owner"] = dict(
                    min_side=round(tmn, 1), max_side=round(tmx, 1),
                    area=round(float(cv2.contourArea(t.astype(np.int32))), 1),
                    support=round(_red_support(mask, t), 3),
                    fill=round(fill_ratio(mask, t), 4),
                    axis=axis_aligned(tq),
                    iou=round(iou_obb(d, t), 3),
                    cd=round(corner_match_distance(d, t), 2))
            pairs.append(rec)
        if (n + 1) % 100 == 0:
            print("  %d/%d" % (n + 1, len(ids)), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)

    lonely = [p for p in pairs if p.get("kind") == "lonely"]
    withg = [p for p in pairs if p.get("kind") != "lonely"]
    withown = [p for p in withg if "owner" in p]
    print("\ntrueFP total %d | lonely (no GT within %dpx) %d | near a GT %d "
          "| of those, a TP owns that GT: %d"
          % (len(pairs), TOL, len(lonely), len(withg), len(withown)))

    if withown:
        for f_ in ("min_side", "max_side", "area", "support", "fill"):
            a = np.array([p[f_] for p in withown], float)
            b = np.array([p["owner"][f_] for p in withown], float)
            print("  %-9s FP  p50 %8.2f   TP(owner) p50 %8.2f   "
                  "FP<owner %d/%d"
                  % (f_, np.percentile(a, 50), np.percentile(b, 50),
                     int((a < b).sum()), len(a)))
        print("  cd(FP,owner) p50 %.1f p90 %.1f"
              % (np.percentile([p["owner"]["cd"] for p in withown], 50),
                 np.percentile([p["owner"]["cd"] for p in withown], 90)))
        print("  iou(FP,owner) p50 %.2f p90 %.2f"
              % (np.percentile([p["owner"]["iou"] for p in withown], 50),
                 np.percentile([p["owner"]["iou"] for p in withown], 90)))

    # how many trueFP are duplicates of a TP that is "the same object"?
    # -> a candidate rule: when two detections overlap heavily, keep the one
    #    with the higher red support.
    if withown:
        for name, key in (("support", "support"), ("fill", "fill"),
                          ("min_side", "min_side"), ("area", "area")):
            better = sum(1 for p in withown
                         if (p[key] > p["owner"][key]) ==
                         (name in ("support", "fill")))
            print("  rule keep-higher-%s picks the TP in %d/%d cases"
                  % (name, better, len(withown)))


if __name__ == "__main__":
    main()