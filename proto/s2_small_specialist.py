"""S2: small/thin-box specialist second pass, fused by UNION.

Motivation: the measured hard strata are short side 20-30 (47% miss) and
aspect >=6 (38% miss), while 50-100 misses only 9%.  Stage-2 CATE on the panel
adds: `hole_fit_pass` earns its TP mainly on min_side <= 50 (+0.011 TP/img,
p=0.067) and `hole_refine` is slightly NEGATIVE on min_side <= 30 (-0.016/img),
so a small-box pass should keep fit_pass and drop refine.

The specialist varies axes that P1 did NOT (P1 only varied denoise_min_area and
min_red_support, whose union gained 0 TP): the admission gates and the fit bands
  min_area 300 -> 120, min_side 12 -> 6, max_aspect 12 -> 20,
  hole_min_side 14 -> 8, hole_fill 0.55 -> 0.50, hole_refine 1 -> 0
then the two passes are UNIONed and deduped at the shipped IoU.

It must win as a UNION over the shipped pass (a replacement is not acceptable).

Usage: python proto/s2_small_specialist.py --workers 12
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
from cv_hybrid import dedup_keep_order
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect
from s1_inward_complete import score, tp_boxset

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def params_of(mode):
    if mode == "base":
        return dict(P), dict(HOLE_DEFAULT)
    p = dict(P)
    p.update(min_area=120.0, min_side=6.0, max_aspect=20.0)
    cfg = dict(HOLE_DEFAULT)
    cfg.update(hole_min_side=8.0, hole_fill=0.50, hole_refine=0,
               hole_fit_pass=2, hole_fit_tol=1.5)
    return p, cfg


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
    elif mode == "spec":
        p2, c2 = params_of("spec")
        det = detect(img, p2, c2, PIPE_DEFAULT, mask=m)
    else:                                   # union_40 / union_60
        d1 = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
        p2, c2 = params_of("spec")
        d2 = detect(img, p2, c2, PIPE_DEFAULT, mask=m)
        thr = 0.40 if mode == "union_40" else 0.60
        det = dedup_keep_order(list(d1) + list(d2), thr)
    return iid, mode, [np.asarray(q, np.int32) for q in det]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/s2_small_specialist.json")
    ap.add_argument("--modes", default="base,spec,union_40")
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
        print("  %-9s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
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
        print("  %-9s dTP@3 %+4d dTP@5 %+4d dTP@8 %+4d dTrueFP@5 %+4d dcorr_f1 %+5.1f"
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
        print("  %-9s base=%d cand=%d  GAINS=%d  LOSSES=%d" % (m, len(sb), len(sm), g, l))

    json.dump({"results": out}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()