"""LOOP-35: Does enforcing orthogonality help when the GT is also orthogonal?

Hypothesis: the benchmark GT (Compare-V1) is non-orthogonal (median 2.08deg),
so enforcing orthogonality on the output diverges from it. If the GT is
orthogonalized (the true targets are rectangles), enforcement should help.

Matrix: {det: none, ortho_edge} x {gt: raw (as-is), ortho (orthogonalized)}.

Usage: python proto/cv_scope35.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, red_mask, load_expected
from canny_quad import get_ids
from cv_hybrid import ROOT
from eval_pointmatch import parse_points, match_boxes_point, is_real_unlabeled
from cv_scope34 import ortho_edge, minarect, load_boxes_any

D_EVAL = 5.0
DET_DIR = os.path.join(ROOT, "tmp_sc31_pm_v5_2")
GT_DIR = r"E:\Xray\CARS\data\Compare-V1@CARS_202609_annotated"


def eval_vs_gt(det_map, gt_map, ids, dist=5.0):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map.get(iid, [])
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
    return TP, round(rec, 1), round(f1, 1)


def main():
    ids = get_ids()
    print(f"LOOP-35: orthogonality enforcement x GT orthogonalization")

    det_none = {iid: [np.asarray(q, np.int32) for q in load_boxes_any(DET_DIR, iid)]
                for iid in ids}
    det_ortho = {iid: [ortho_edge(q) for q in load_boxes_any(DET_DIR, iid)]
                 for iid in ids}
    gt_raw = {iid: load_boxes_any(GT_DIR, iid) for iid in ids}
    gt_ortho = {iid: [ortho_edge(q) for q in load_boxes_any(GT_DIR, iid)]
                for iid in ids}

    print(f"\n  {'det':8s} x {'gt':8s} {'TP':>5s} {'recall':>7s} {'corr_f1':>8s}")
    for dname, dm in [("none", det_none), ("ortho", det_ortho)]:
        for gname, gm in [("raw", gt_raw), ("ortho", gt_ortho)]:
            TP, rec, f1 = eval_vs_gt(dm, gm, ids, D_EVAL)
            print(f"  {dname:8s} x {gname:8s} {TP:5d} {rec:6.1f}% {f1:7.1f}%")

    out = os.path.join(ROOT, "benchmark", "cv_scope35.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"note": "det x gt orthogonalization matrix"}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()