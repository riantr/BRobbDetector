"""Evaluate a detection directory (any producer) against the TRUE GT.

Usage:
  python proto/eval_det_dir.py tmp_cs_hue10
  python proto/eval_det_dir.py tmp_cs_hue5 --method cvhybrid
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import get_ids
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
CMP1 = os.path.join(DATA, "Compare-V1@CARS_202609_annotated")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("det_dir")
    ap.add_argument("--gt", default="raw", choices=["raw", "Compare-V1"])
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    ids = get_ids()
    dm = {i: load_boxes_any(args.det_dir, i) for i in ids}
    n = sum(len(v) for v in dm.values())
    gt_dir = RAW if args.gt == "raw" else CMP1
    row = {str(d): evaluate(dm, ids, gt_dir, float(d)) for d in DIST_VALUES}
    print(f"{os.path.basename(args.det_dir):22s} gt={args.gt:11s} det={n:4d}  " +
          "  ".join(f"D{d}: {row[str(d)]['tp']:4d}/{row[str(d)]['corrected_f1']:5.1f}"
                    for d in DIST_VALUES))
    if args.save:
        path = os.path.join(ROOT, "benchmark", args.save + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"det_dir": args.det_dir, "gt": gt_dir, "det": n,
                       "per_D": row}, f, ensure_ascii=False, indent=2)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()