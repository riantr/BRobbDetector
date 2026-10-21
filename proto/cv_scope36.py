"""LOOP-36: orthogonality enforcement x candidate GT matrix (decisive).

LOOP-33/34/35 concluded that enforcing orthogonality costs ~20% TP. That was
measured against Compare-V1@CARS_202609_annotated, which this script shows is a
MACHINE-GENERATED "compare" artifact (10.5% of its boxes are grossly
non-orthogonal, only 73% have red support >= 0.7, and its deviation profile
mirrors our own detector output).

Label@CARS_202609_raw is the real hand-drawn GT:
  - 1121 boxes over 499/500 benchmark ids (2.25/img) vs Compare-V1's 746 (1.53)
  - orthogonality deviation median 0.33 deg, p90 0.96 deg, only 0.1% > 10 deg
  - 100% of boxes have red support >= 0.70 in the benchmark images
  - the raw images are the SAME pictures as the annotated ones (same dims,
    mean abs pixel diff 1-8, i.e. re-encoding only)

So the enforcement decision and the GT decision must be made together. This
script measures the full det x gt matrix, plus threshold-based enforcement
(orthogonalize only boxes whose deviation exceeds X deg).

Usage: python proto/cv_scope36.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import parse_points, match_boxes_point, is_real_unlabeled
from canny_quad import get_ids
from cv_scope34 import (minarect, lsq_center, ortho_edge, ortho_blend,
                        load_boxes_any, DET_DIR)

ROOT = r"D:\src\CodeArts\BRobbDetector"
D_EVAL = 5.0

GT_DIRS = {
    "Compare-V1": os.path.join(DATA, "Compare-V1@CARS_202609_annotated"),
    "Compare":    os.path.join(DATA, "Compare@CARS_202609_annotated"),
    "raw":        os.path.join(DATA, "Label@CARS_202609_raw"),
}


def deviation(q):
    """max |interior angle - 90| in degrees."""
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
    """Orthogonalize (toward ortho_edge) only when deviation > thresh."""
    def f(q):
        if deviation(q) <= thresh:
            return np.asarray(q, np.int32)
        o = ortho_edge(q).astype(float)
        a = alpha_fn
        return np.round((1 - a) * np.asarray(q, float) + a * o).astype(np.int32)
    return f


def eval_vs_gt(det_map, ids, gt_dir, dist=D_EVAL):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    n_gt = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(gt_dir, iid)
        n_gt += len(gt)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        tp, unmatched = match_boxes_point(det, gt, dist)
        TP += tp; FN += len(gt) - tp; raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], exp.get(iid, 0), len(gt), mask_d):
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
    ids = get_ids()
    exp = load_expected()
    csv_total = sum(exp.get(i, 0) for i in ids)
    print(f"LOOP-36: enforcement x GT matrix. {len(ids)} images, D={D_EVAL:.0f}")
    for name, d in GT_DIRS.items():
        n = sum(len(load_boxes_any(d, i)) for i in ids)
        print(f"  GT {name:12s} {n:5d} boxes")
    print(f"  CSV expected total (detail csv, these ids): {csv_total}\n")

    methods = {
        "none": lambda q: np.asarray(q, np.int32),
        "minarect": minarect,
        "lsq_center": lsq_center,
        "ortho_edge": ortho_edge,
        "blend0.5": lambda q: ortho_blend(q, 0.5),
        "blend0.3": lambda q: ortho_blend(q, 0.3),
        "blend0.2": lambda q: ortho_blend(q, 0.2),
        "thr5_strict":  make_threshold(1.0, 5),
        "thr10_strict": make_threshold(1.0, 10),
        "thr15_strict": make_threshold(1.0, 15),
        "thr10_blend":  make_threshold(0.3, 10),
    }

    results = {}
    print(f"  {'method':13s} " + "  ".join(f"{g:>22s}" for g in GT_DIRS))
    print(f"  {'':13s} " + "  ".join(f"{'TP/corr_f1/recall':>22s}" for _ in GT_DIRS))
    for name, fn in methods.items():
        dm = {iid: [fn(q) for q in load_boxes_any(DET_DIR, iid)] for iid in ids}
        row = {}
        cells = []
        for gname, gdir in GT_DIRS.items():
            m = eval_vs_gt(dm, ids, gdir)
            row[gname] = m
            cells.append(f"{m['tp']:4d}/{m['corrected_f1']:5.1f}/{m['recall']:5.1f}%")
        results[name] = row
        print(f"  {name:13s} " + "  ".join(f"{c:>22s}" for c in cells))

    out = os.path.join(ROOT, "benchmark", "cv_scope36.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"gt_dirs": GT_DIRS, "d_eval": D_EVAL,
                   "csv_expected_total": csv_total,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()