"""LOOP-37: hue-tightened red mask.

The human annotation burned into Image@CARS_202609_annotated is drawn in pure
red (hue 0-4 for 85% of the annotation pixels), while the orange/red X-ray
object texture sits at hue 9-10 (73% of the off-annotation red pixels).  The
current red_mask() accepts hue 0-10, so the object texture floods the mask
(~75% of the mask pixels are object, only ~25% are annotation lines).

This script re-runs the best detector under the true GT (edgefit) with a
tightened hue range and measures the effect against Label@CARS_202609_raw.

Usage: python proto/scope37_hue.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

import common
import cv_hybrid
from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids
from cv_hybrid import FIT_DEFAULT

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
DIST_VALUES = [3, 5, 8, 12]


def make_mask(hue_max):
    def m(img, smin=80, vmin=110):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        m1 = cv2.inRange(hsv, (0, smin, vmin), (hue_max, 255, 255))
        m2 = cv2.inRange(hsv, (180 - hue_max if hue_max < 10 else 170, smin, vmin),
                         (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    return m


def evaluate(det_map, ids, gt_dir, dist):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = cv2.load_boxes = None
        from cv_scope34 import load_boxes_any
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


def main():
    ids = get_ids()
    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    print(f"LOOP-37 hue-tightened mask, edgefit detector, {len(ids)} images, "
          f"GT=raw, p.min_area={p['min_area']} ortho_tol={p['ortho_tol']}")

    variants = [("baseline(hue<=10)", common.red_mask),
                ("hue<=8", make_mask(8)),
                ("hue<=6", make_mask(6)),
                ("hue<=4", make_mask(4))]
    results = {}
    for name, fn in variants:
        t0 = time.time()
        cv_hybrid.red_mask = fn
        det_map = {}
        total = 0
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                det_map[iid] = []
                continue
            quads = cv_hybrid.quads_edgefit(img, p)
            det_map[iid] = quads
            total += len(quads)
        row = {str(d): evaluate(det_map, ids, RAW, float(d)) for d in DIST_VALUES}
        results[name] = {"det": total, "per_D": row}
        print(f"  {name:18s} det={total:4d}  " + "  ".join(
            f"D{d}: {row[str(d)]['tp']:4d}/{row[str(d)]['corrected_f1']:5.1f}"
            for d in DIST_VALUES) + f"   ({time.time()-t0:.0f}s)")

    path = os.path.join(ROOT, "benchmark", "scope37_hue.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "gt": RAW, "detector": "edgefit",
                   "params": {k: p[k] for k in sorted(p) if isinstance(p[k], (int, float, str, list))},
                   "dist_values": DIST_VALUES, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()