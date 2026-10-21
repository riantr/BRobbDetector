"""Orthogonality enforcement, re-evaluated against the TRUE GT (raw).

The earlier conclusion "enforcing orthogonality costs ~20% TP" was measured
against Compare-V1 (a machine artifact).  This script re-runs the same
enforcement variants against Label@CARS_202609_raw, and on top of the
best-performing detector under the true GT (edgefit) as well as the frozen
pm_v5_2 production detector.

Usage: python proto/enforce_rawgt.py
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import get_ids
from cv_scope34 import (minarect, lsq_center, ortho_edge, ortho_blend,
                        load_boxes_any)

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
CMP1 = os.path.join(DATA, "Compare-V1@CARS_202609_annotated")
DET_DIRS = {
    "pm_v5_2": os.path.join(ROOT, "tmp_sc31_pm_v5_2"),
    "edgefit": os.path.join(ROOT, "tmp_cv_edgefit"),
}
DIST_VALUES = [3, 5, 8, 12]


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


METHODS = {
    "none": lambda q: np.asarray(q, np.int32),
    "minarect": minarect,
    "lsq_center": lsq_center,
    "ortho_edge": ortho_edge,
    "blend0.5": lambda q: ortho_blend(q, 0.5),
    "blend0.3": lambda q: ortho_blend(q, 0.3),
    "blend0.2": lambda q: ortho_blend(q, 0.2),
    "thr5_strict": make_threshold(1.0, 5),
    "thr10_strict": make_threshold(1.0, 10),
    "thr15_strict": make_threshold(1.0, 15),
    "thr10_blend": make_threshold(0.3, 10),
}


def evaluate(det_map, ids, gt_dir, dist):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = n_gt = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(gt_dir, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        n_gt += len(gt)
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
    return {"tp": TP, "fn": FN, "gt": n_gt, "raw_fp": raw_fp, "true_fp": true_fp,
            "real_unlabeled": real_unl, "recall": round(rec, 1),
            "corrected_f1": round(f1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="enforce_rawgt")
    args = ap.parse_args()

    ids = get_ids()
    out = {}
    for dname, ddir in DET_DIRS.items():
        if not os.path.isdir(ddir):
            continue
        base = {i: load_boxes_any(ddir, i) for i in ids}
        print(f"\n=== detector: {dname} ===")
        print(f"  {'method':13s} " + "  ".join(f"{'D'+str(d):>13s}" for d in DIST_VALUES)
              + "   (TP/corr_f1 vs TRUE GT)")
        rows = {}
        for mname, fn in METHODS.items():
            dm = {iid: [fn(q) for q in base.get(iid, [])] for iid in ids}
            row = {str(d): evaluate(dm, ids, RAW, float(d)) for d in DIST_VALUES}
            rows[mname] = row
            print(f"  {mname:13s} " + "  ".join(
                f"{row[str(d)]['tp']:5d}/{row[str(d)]['corrected_f1']:5.1f}" for d in DIST_VALUES))
        out[dname] = rows

    path = os.path.join(ROOT, "benchmark", args.save + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gt": RAW, "detectors": DET_DIRS, "dist_values": DIST_VALUES,
                   "results": out}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()