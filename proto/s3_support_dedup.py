"""S3: support-priority dedup.

Motivation: `dedup_iou` was never in the pm_v10 2^4 panel, and LOOP-31 showed it
is a violent trade (dedup_iou 1.0 -> TP 615 but corr_f1 55.7).  S2 just showed
the mechanism directly: the specialist pass found 4 GT boxes the shipped pass
missed, and dedup_keep_order discarded all 4 because the shipped quads are
emitted first and overlap them (union gained 0 TP while adding 551 trueFP).

Today dedup keeps the FIRST-EMITTED quad of an overlapping pair.  This keeps the
quad with the higher red support instead, with dedup_iou fixed at the shipped
0.40 (per the goal: do not retune the threshold unless the interaction is itself
measured - it is not, so it stays fixed).

Usage: python proto/s3_support_dedup.py --workers 12
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from cv_hybrid import iou_obb
from quad_ortho import quad_on_boundary
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect, detect_holes,
                           detect_edgefit, _red_support)
from s1_inward_complete import score, tp_boxset

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def dedup_by_support(quads, mask, thr=0.40):
    """Greedy dedup ordered by descending red support (ties: emission order)."""
    if not quads:
        return []
    sup = [float(_red_support(mask, np.asarray(q, float))) for q in quads]
    order = sorted(range(len(quads)), key=lambda i: (-sup[i], i))
    keep = []
    for i in order:
        qi = np.asarray(quads[i], np.int32)      # iou_obb requires int32
        if all(iou_obb(qi, np.asarray(quads[k], np.int32)) < thr for k in keep):
            keep.append(i)
    return [quads[i] for i in keep]


def _work(task):
    iid, mode = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, mode, []
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    if mode == "base":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    else:
        raw = detect_holes(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
        raw += detect_edgefit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
        det = dedup_by_support(raw, m, PIPE_DEFAULT.get("dedup_iou", 0.40))
    return iid, mode, [np.asarray(q, np.int32) for q in det]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/s3_support_dedup.json")
    ap.add_argument("--modes", default="base,sup")
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [g for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()
    print("%d images, %d GT boxes" % (len(ids), sum(len(v) for v in gt_map.values())),
          flush=True)

    modes = args.modes.split(",")
    out, dets = {}, {}
    for mode in modes:
        det_map = {}
        with Pool(args.workers) as pool:
            for iid, _m, det in pool.imap_unordered(
                    _work, [(i, mode) for i in ids], chunksize=3):
                det_map[iid] = det
        dets[mode] = det_map
        r = score(det_map, ids, gt_map, exp)
        out[mode] = {"metrics": r}
        print("  %-6s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d corr_f1@5=%.1f"
              % (mode, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"],
                 r["5"]["true_fp"], r["8"]["tp"], r["8"]["true_fp"],
                 r["12"]["tp"], r["12"]["true_fp"], r["5"]["n_det"],
                 r["5"]["corr_f1"]), flush=True)

    base = out["base"]["metrics"]
    print("\n== delta vs base ==")
    for m in modes:
        if m == "base":
            continue
        d = out[m]["metrics"]
        print("  %-6s dTP@3 %+4d dTP@5 %+4d dTP@8 %+4d dTrueFP@5 %+4d dcorr_f1 %+5.1f"
              % (m, d["3"]["tp"] - base["3"]["tp"], d["5"]["tp"] - base["5"]["tp"],
                 d["8"]["tp"] - base["8"]["tp"],
                 d["5"]["true_fp"] - base["5"]["true_fp"],
                 d["5"]["corr_f1"] - base["5"]["corr_f1"]))

    sb = tp_boxset(dets["base"], gt_map, ids)
    print("\n== TP-set decomposition at D=5 ==")
    for m in modes:
        if m == "base":
            continue
        sm = tp_boxset(dets[m], gt_map, ids)
        g, l = len(sm - sb), len(sb - sm)
        out[m]["tp_set"] = {"gains": g, "losses": l, "base_tp": len(sb),
                            "cand_tp": len(sm)}
        print("  %-6s base=%d cand=%d  GAINS=%d  LOSSES=%d" % (m, len(sb), len(sm), g, l))

    json.dump({"results": out}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()