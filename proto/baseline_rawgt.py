"""Re-baseline the point-match benchmark against the TRUE ground truth.

GT = Label@CARS_202609_raw (the hand-drawn annotation burned into
Image@CARS_202609_annotated).  The previous baseline was measured against
Compare-V1@CARS_202609_annotated, a machine artifact -- see
benchmark/gt_final.json for the arbitration evidence.

Usage:
  python proto/baseline_rawgt.py --save pm_v6_rawgt
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import get_ids
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
GT_DIRS = {
    "raw": os.path.join(DATA, "Label@CARS_202609_raw"),
    "Compare-V1": os.path.join(DATA, "Compare-V1@CARS_202609_annotated"),
}
DET_DIRS = {
    "pm_v5_2": os.path.join(ROOT, "tmp_sc31_pm_v5_2"),
    "cvhybrid_cs": os.path.join(ROOT, "tmp_csharp_cvhybrid"),
    "o15_ba": os.path.join(ROOT, "tmp_sc6_hy_o15_ba"),
    "hy_chm": os.path.join(ROOT, "tmp_sc3_hy_chm"),
    "hybrid_cem": os.path.join(ROOT, "tmp_cv_hybrid_cem"),
    "hull": os.path.join(ROOT, "tmp_cv_hull"),
    "edgefit": os.path.join(ROOT, "tmp_cv_edgefit"),
    "canny": os.path.join(ROOT, "tmp_canny_out"),
    "yolo": os.path.join(ROOT, "tmp_yolo_out"),
}
DIST_VALUES = [3, 5, 8, 12]


def evaluate(det_map, ids, gt_dir, dist):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = n_gt = n_det = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(gt_dir, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        n_gt += len(gt); n_det += len(det)
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
    return {"tp": TP, "fn": FN, "gt": n_gt, "det": n_det, "raw_fp": raw_fp,
            "true_fp": true_fp, "real_unlabeled": real_unl,
            "recall": round(rec, 1), "precision": round(prec, 1),
            "corrected_f1": round(f1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    ids = get_ids()
    print(f"Re-baseline against TRUE GT: {len(ids)} images, D={DIST_VALUES}\n")
    results = {}
    for dname, ddir in DET_DIRS.items():
        if not os.path.isdir(ddir):
            continue
        dm = {i: load_boxes_any(ddir, i) for i in ids}
        n_det = sum(len(v) for v in dm.values())
        results[dname] = {}
        for gname, gdir in GT_DIRS.items():
            results[dname][gname] = {str(d): evaluate(dm, ids, gdir, float(d))
                                     for d in DIST_VALUES}
        r = results[dname]["raw"]
        c = results[dname]["Compare-V1"]
        print(f"  {dname:12s} det={n_det:4d}  vs raw: " +
              "  ".join(f"D{d}={r[str(d)]['tp']:4d}/{r[str(d)]['corrected_f1']:5.1f}"
                        for d in DIST_VALUES) +
              f"   | vs Compare-V1 D5={c['5']['tp']}/{c['5']['corrected_f1']}")

    out = {"n_images": len(ids), "dist_values": DIST_VALUES,
           "gt_dirs": GT_DIRS, "results": results}
    if args.save:
        path = os.path.join(ROOT, "benchmark", args.save + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()