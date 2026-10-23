"""Statistics of the red annotation furniture: label TEXT and leader ARROWS.

The burned-in annotation is not only the boxes: every object also carries a
red class label ("液体 0.95") and a leader arrow that points from the label
to the object.  Both are red and both create holes in the red mask, so they
are a major source of false positives.

This script measures, on the 500 real benchmark images:

  * the red components that are far away from every GT box  (the text)
    - count per image, area, bounding box size
  * the holes of the red mask and their rectangularity
    (area / minAreaRect area): a box interior is ~0.9, an arrow head or a
    character counter is far below that

Usage: python proto/measure_annotation_furniture.py
"""
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def bbox_dist(box, comp):
    """Distance between an axis aligned box and an axis aligned component."""
    ax0, ay0, ax1, ay1 = box
    bx0, by0, bx1, by1 = comp
    dx = max(0.0, max(ax0 - bx1, bx0 - ax1))
    dy = max(0.0, max(ay0 - by1, by0 - ay1))
    return float(np.hypot(dx, dy))


def main():
    ids = get_ids()
    n_text = []
    text_area = []
    text_wh = []
    comp_n = []
    hole_rect = []
    hole_n = []
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
               if not quad_on_boundary(g, W, H, 0.0)]
        gb = [(float(g[:, 0].min()), float(g[:, 1].min()),
               float(g[:, 0].max()), float(g[:, 1].max())) for g in gts]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp_n.append(len(cnts))
        nt = 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 20:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w > 40 or h > 40:
                continue
            if gb and min(bbox_dist(b, (x, y, x + w, y + h)) for b in gb) < 20:
                continue
            nt += 1
            text_area.append(a)
            text_wh.append((w, h))
        n_text.append(nt)

        # holes and their rectangularity
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is not None:
            hier = hier[0]
            nh = 0
            for i, c in enumerate(cnts):
                if hier[i][3] != -1:
                    continue
                child = hier[i][2]
                while child != -1:
                    a = cv2.contourArea(cnts[child])
                    if a >= 20:
                        (rw, rh) = cv2.minAreaRect(cnts[child])[1]
                        hole_rect.append(a / max(rw * rh, 1e-6))
                        nh += 1
                    child = hier[child][0]
            hole_n.append(nh)

    def stat(name, v):
        v = np.array(v, float)
        print("%-22s n=%6d  mean %7.2f  p10 %7.2f  p50 %7.2f  p90 %7.2f  max %7.2f"
              % (name, len(v), v.mean(), np.percentile(v, 10),
                 np.percentile(v, 50), np.percentile(v, 90), v.max()))

    stat("components / image", comp_n)
    stat("text components/img", n_text)
    stat("text area", text_area)
    wh = np.array(text_wh, float)
    if len(wh):
        stat("text width", wh[:, 0])
        stat("text height", wh[:, 1])
    stat("holes / image", hole_n)
    stat("hole rectangularity", hole_rect)
    hr = np.array(hole_rect, float)
    print("holes with rectangularity < 0.6: %d (%.1f%%)"
          % (int((hr < 0.6).sum()), (hr < 0.6).mean() * 100))
    print("holes with rectangularity < 0.75: %d (%.1f%%)"
          % (int((hr < 0.75).sum()), (hr < 0.75).mean() * 100))
    with open(os.path.join(ROOT, "benchmark", "real_furniture.json"), "w",
              encoding="utf-8") as f:
        json.dump({"comp_per_img": comp_n, "text_per_img": n_text,
                   "text_area": text_area,
                   "text_wh": text_wh,
                   "holes_per_img": hole_n,
                   "hole_rect": hole_rect}, f, ensure_ascii=False)
    print("saved benchmark/real_furniture.json")


if __name__ == "__main__":
    main()