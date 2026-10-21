"""LOOP-39: parameter ablation for the edgefit-only + hue<=5 configuration.

The C# detector was re-based to the true GT (proto/gt_final.json):
hue<=5, edgefit-only family, min_area 900, boundary_band 6, dedup_iou 0.30.
This script ablates each parameter against Label@CARS_202609_raw so the final
C# constants are individually justified.

Usage: python proto/scope39_ablation.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

import common
import cv_hybrid
from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids
from cv_hybrid import FIT_DEFAULT, MASK_DEFAULT
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
DIST_VALUES = [3, 5, 8, 12]


def evaluate(det_map, ids, gt_dir, dist):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(gt_dir, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        md = cv2.dilate(red_mask(img), np.ones((3, 3), np.uint8), iterations=3)
        tp, unmatched = match_boxes_point(det, gt, dist)
        TP += tp; FN += len(gt) - tp; raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], exp.get(iid, 0), len(gt), md):
                real_unl += 1
            else:
                true_fp += 1
    tot = TP + FN
    rec = TP / tot * 100 if tot else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
            "real_unlabeled": real_unl, "recall": round(rec, 1),
            "precision": round(prec, 1), "corrected_f1": round(f1, 1)}


CONFIGS = [
    ("final(h5,ma900,bb6,di.30)", {}),
    ("min_area=1500", {"min_area": 1500.0}),
    ("min_area=500", {"min_area": 500.0}),
    ("boundary_band=20", {"boundary_band": 20.0}),
    ("boundary_band=0", {"boundary_band": 0.0}),
    ("dedup_iou=0.75", {"dedup_iou": 0.75}),
    ("dedup_iou=0.15", {"dedup_iou": 0.15}),
    ("ortho_tol=10", {"ortho_tol": 10.0}),
    ("ortho_tol=30", {"ortho_tol": 30.0}),
    ("denoise_min_area=0", {"denoise_min_area": 0}),
]


def main():
    ids = get_ids()
    results = {}
    for name, over in CONFIGS:
        p = dict(CQ_DEFAULT)
        p.update(FIT_DEFAULT)
        p.update(MASK_DEFAULT)
        p.update(over)
        t0 = time.time()
        det = {}
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            det[iid] = cv_hybrid.quads_edgefit(img, p) if img is not None else []
        n = sum(len(v) for v in det.values())
        row = {str(d): evaluate(det, ids, RAW, float(d)) for d in DIST_VALUES}
        results[name] = {"params": {k: p[k] for k in over}, "det": n, "per_D": row}
        print(f"  {name:28s} det={n:4d}  " + "  ".join(
            f"D{d}: {row[str(d)]['tp']:4d}/{row[str(d)]['corrected_f1']:5.1f}"
            for d in DIST_VALUES) + f"  ({time.time()-t0:.0f}s)")

    path = os.path.join(ROOT, "benchmark", "scope39_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "gt": RAW, "detector": "edgefit",
                   "hue": 5, "dist_values": DIST_VALUES, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()