"""LOOP-38: hue sweep x detector family, against the TRUE GT.

LOOP-37 showed the red-mask hue range is the dominant error source:
hue<=6 lifts the edgefit detector from 336 TP / 39.5 corr_f1 to
589 TP / 65.7 corr_f1 at D=5 (GT = Label@CARS_202609_raw).

This script sweeps the hue cut-off across several detector families and also
re-checks the orthogonality-enforcement variants on top of the fixed mask.

Usage: python proto/scope38_hue_sweep.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

import common
import cv_hybrid
from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids
from cv_hybrid import FIT_DEFAULT, HYBRIDS
from cv_scope34 import (minarect, lsq_center, ortho_edge, ortho_blend,
                        load_boxes_any)

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
DIST_VALUES = [3, 5, 8, 12]
HUES = [10, 8, 7, 6, 5, 4]
DETECTORS = ["edgefit", "hybrid_cem", "mar", "hull", "hybrid_all"]


def make_mask(hue_max):
    def m(img, smin=80, vmin=110):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, smin, vmin), (hue_max, 255, 255))
        m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    return m


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


def run_detector(name, ids, p):
    if name in cv_hybrid.DETECTORS:
        det = {}
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            det[iid] = cv_hybrid.DETECTORS[name](img, p) if img is not None else []
        return det
    order = HYBRIDS[name]
    det = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det[iid] = []
            continue
        per = [cv_hybrid.DETECTORS[m](img, p) for m in order]
        det[iid] = cv_hybrid.dedup_keep_order(
            [q for lst in per for q in lst], p["dedup_iou"])
    return det


def deviation(q):
    q = np.asarray(q, float)
    worst = 0.0
    for i in range(4):
        p0, p1, p2 = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        worst = max(worst, abs(np.degrees(np.arccos(c)) - 90.0))
    return worst


def make_threshold(alpha_fn, thresh):
    def f(q):
        if deviation(q) <= thresh:
            return np.asarray(q, np.int32)
        o = ortho_edge(q).astype(float)
        a = alpha_fn
        return np.round((1 - a) * np.asarray(q, float) + a * o).astype(np.int32)
    return f


ENFORCE = {
    "none": lambda q: np.asarray(q, np.int32),
    "minarect": minarect,
    "ortho_edge": ortho_edge,
    "blend0.5": lambda q: ortho_blend(q, 0.5),
    "blend0.3": lambda q: ortho_blend(q, 0.3),
    "thr5_strict": make_threshold(1.0, 5),
    "thr10_blend": make_threshold(0.3, 10),
}


def main():
    ids = get_ids()
    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    print(f"LOOP-38 hue sweep x detector, {len(ids)} images, GT=raw\n")
    results = {}
    for hue in HUES:
        cv_hybrid.red_mask = make_mask(hue)
        for dname in DETECTORS:
            t0 = time.time()
            det = run_detector(dname, ids, p)
            n = sum(len(v) for v in det.values())
            row = {str(d): evaluate(det, ids, RAW, float(d)) for d in DIST_VALUES}
            results[f"hue{hue}_{dname}"] = {"det": n, "per_D": row}
            print(f"  hue<={hue:<2d} {dname:11s} det={n:4d}  " + "  ".join(
                f"D{d}: {row[str(d)]['tp']:4d}/{row[str(d)]['corrected_f1']:5.1f}"
                for d in DIST_VALUES) + f"  ({time.time()-t0:.0f}s)")

    # enforcement check on the best configuration
    best = max(results, key=lambda k: results[k]["per_D"]["5"]["corrected_f1"])
    hue = int(best.split("_")[0][3:])
    dname = best.split("_", 1)[1]
    print(f"\n  best config: {best} -> enforcement re-check")
    cv_hybrid.red_mask = make_mask(hue)
    base = run_detector(dname, ids, p)
    enf = {}
    for ename, fn in ENFORCE.items():
        dm = {iid: [fn(q) for q in base.get(iid, [])] for iid in ids}
        row = {str(d): evaluate(dm, ids, RAW, float(d)) for d in DIST_VALUES}
        enf[ename] = row
        print(f"    {ename:12s} " + "  ".join(
            f"D{d}: {row[str(d)]['tp']:4d}/{row[str(d)]['corrected_f1']:5.1f}"
            for d in DIST_VALUES))

    path = os.path.join(ROOT, "benchmark", "scope38_hue_sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "gt": RAW, "hues": HUES,
                   "detectors": DETECTORS, "dist_values": DIST_VALUES,
                   "results": results, "enforcement_on_best": enf},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()