"""LOOP-50: visual crops of the remaining false negatives.

Reads benchmark/scope47_fn2.json and writes a crop per sampled FN into
tmp_fn50/, showing: the input crop, the red mask, the hole contours, the GT
rectangle and the hole-pipeline detections.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _hole_quad
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
OUT = os.path.join(ROOT, "tmp_fn50")
CFG = dict(HOLE_DEFAULT)


def main():
    per_reason = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    only = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(os.path.join(ROOT, "benchmark", "scope47_fn2.json"),
                          encoding="utf-8"))["fn_detail"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["reason"]].append(r)

    gt_map = {}
    for reason, items in by.items():
        if only and reason != only:
            continue
        for k, r in enumerate(items[:per_reason]):
            iid, gi = r["id"], r["gi"]
            if iid not in gt_map:
                gt_map[iid] = load_boxes_any(RAW, iid)
            gt = gt_map[iid]
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
            mask = denoise(mask, P["denoise_min_area"])
            holes = _holes_of(mask, CFG)
            g = np.asarray(gt[gi], np.int32)
            x0, y0 = g.min(axis=0)
            x1, y1 = g.max(axis=0)
            pad = 60
            X0 = max(int(x0) - pad, 0); Y0 = max(int(y0) - pad, 0)
            X1 = min(int(x1) + pad, img.shape[1]); Y1 = min(int(y1) + pad, img.shape[0])
            crop = img[Y0:Y1, X0:X1].copy()
            mcrop = (mask[Y0:Y1, X0:X1] * 255)
            vis = crop.copy()
            # holes
            for h in holes:
                cv2.drawContours(vis, [h - np.array([X0, Y0])], -1, (255, 0, 255), 1)
            # detections
            for h in holes:
                q = _hole_quad(h, mask, CFG)
                if q is not None:
                    qq = q.astype(np.int32) - np.array([X0, Y0])
                    cv2.polylines(vis, [qq], True, (0, 255, 255), 1)
            # GT
            cv2.polylines(vis, [g - np.array([X0, Y0])], True, (0, 255, 0), 2)
            m3 = cv2.cvtColor(mcrop, cv2.COLOR_GRAY2BGR)
            both = np.vstack([vis, m3])
            best = r.get("best_ov")
            tag = f"{reason}_{k}"
            cv2.imwrite(os.path.join(OUT, f"{iid}_{gi}_{tag}.png"), both)
            print(f"{iid} gt#{gi} {reason} best={best} -> tmp_fn50/{iid}_{gi}_{tag}.png")


if __name__ == "__main__":
    main()