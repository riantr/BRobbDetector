"""Parameter sweep on the REAL 500-image CARS benchmark.

The emulation benchmark (proto/sweep_emulation.py) is used to EXPLORE the
parameter space cheaply, but every candidate must be re-validated on the
real data before it is adopted.  This script runs the production pipeline
(proto/scope45_holes.py) over the 500 real images for a list of configuration
variants and scores each of them with the project point-match metric
(D = 3/5/8/12) against the rule-consistent GT.

Scoring mirrors proto/scope78_parity.py exactly:
  * TP  = all 4 corners within D px after an optimal 4-to-4 assignment
  * a detection that matches no GT box is a TRUE FP unless the
    `is_real_unlabeled` heuristic classifies it as a real (but unlabelled)
    annotation
  * the GT is the RAW GT with the 2026-10-21 edge rule applied (the boxes
    whose rectangle touches the image edge are not valid targets)

Usage:
    python proto/sweep_real.py --variants benchmark/sweep_real_r1.json
    python proto/sweep_real.py --variants ... --workers 14
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]

_CFG = {}
_PIPE = {}


def _prod_mask(img):
    """The production red mask (hue<=4, sat>=100, val>=140) + denoise."""
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    return m


def _loose_mask(img):
    """The wide mask used by the `is_real_unlabeled` heuristic."""
    return cv2.dilate(red_mask(img, 80, 110, 10),
                      np.ones((3, 3), np.uint8), iterations=3) > 0


def _init(cfg, pipe):
    global _CFG, _PIPE
    _CFG = cfg
    _PIPE = pipe


def _work(iid):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, []
    det = detect(img, P, _CFG, _PIPE, mask=_prod_mask(img))
    return iid, [np.asarray(q, np.int32) for q in det]


def apply_overrides(base, text):
    cfg = dict(base)
    if not text:
        return cfg
    for kv in filter(None, text.split(",")):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            cfg[k] = float(v)
        except ValueError:
            cfg[k] = v
    return cfg


def sample_ids(seed=0, n=500):
    """The benchmark image sample, with a configurable seed.

    canny_quad.get_ids() draws 500 images from the annotated pool with seed 0 -
    that is the TUNING set.  A different seed gives a held-out sample from the
    same pool (56832 images), which is how a candidate is checked for
    generalisation instead of being accepted on the set it was tuned on.
    """
    if seed == 0:
        ids = get_ids()
        return ids if n >= len(ids) else ids[:n]
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and
           os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return ids[:n]


def build_data(seed=0, n=500):
    ids = sample_ids(seed, n)
    gt_raw = {iid: load_boxes_any(RAW, iid) for iid in ids}
    exp = load_expected()
    gt = {}
    loose = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            gt[iid] = []
            continue
        H, W = img.shape[:2]
        gt[iid] = [g for g in gt_raw[iid] if not quad_on_boundary(g, W, H, 0.0)]
        loose[iid] = _loose_mask(img)
    return ids, gt, exp, loose


def score(ids, det_map, gt_map, exp, loose_map):
    tot = {D: {"tp": 0, "raw_fp": 0, "true_fp": 0, "real_unl": 0}
           for D in DIST_VALUES}
    ndet = 0
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map[iid]
        ndet += len(det)
        mask_d = loose_map.get(iid)
        csv_n = exp.get(iid, 0)
        for D in DIST_VALUES:
            tp, unmatched = match_boxes_point(det, gt, float(D))
            tot[D]["tp"] += tp
            tot[D]["raw_fp"] += len(unmatched)
            for di in unmatched:
                if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                    tot[D]["real_unl"] += 1
                else:
                    tot[D]["true_fp"] += 1
    total_gt = sum(len(gt_map[i]) for i in ids)
    out = {}
    for D in DIST_VALUES:
        r = tot[D]
        rec = r["tp"] / max(total_gt, 1) * 100
        prec = r["tp"] / max(r["tp"] + r["true_fp"], 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[str(D)] = dict(r, recall=round(rec, 1), precision=round(prec, 1),
                           corrected_f1=round(f1, 1))
    return ndet, total_gt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark",
                                                  "sweep_real_r1.json"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = the tuning set (canny_quad.get_ids); "
                         ">0 = a held-out sample of the same size")
    args = ap.parse_args()

    with open(args.variants, encoding="utf-8") as f:
        variants = json.load(f)

    ids, gt_map, exp, loose = build_data(args.seed, 500)
    if args.n:
        ids = ids[:args.n]
    total_gt = sum(len(gt_map[i]) for i in ids)
    print("real: %d images, %d GT boxes (rule-consistent, seed=%d)" %
          (len(ids), total_gt, args.seed), flush=True)

    rows = []
    for v in variants:
        cfg = apply_overrides(HOLE_DEFAULT, v.get("cfg", ""))
        pipe = apply_overrides(PIPE_DEFAULT, v.get("pipe", ""))
        t0 = time.time()
        det_map = {}
        with Pool(args.workers, initializer=_init, initargs=(cfg, pipe)) as pool:
            for iid, det in pool.imap_unordered(_work, ids, chunksize=4):
                det_map[iid] = det
        ndet, _, per_D = score(ids, det_map, gt_map, exp, loose)
        dt = time.time() - t0
        rec = {"name": v.get("name", "?"), "cfg": v.get("cfg", ""),
               "pipe": v.get("pipe", ""), "sec": round(dt, 1),
               "ndet": ndet, "ngt": total_gt, "per_D": per_D}
        rows.append(rec)
        print("%-26s %5.1fs ndet=%4d  " % (rec["name"], dt, ndet) +
              "  ".join("D%d %4d tp/%4d fp r%5.1f p%5.1f f%5.1f" %
                        (D, per_D[str(D)]["tp"], per_D[str(D)]["true_fp"],
                         per_D[str(D)]["recall"], per_D[str(D)]["precision"],
                         per_D[str(D)]["corrected_f1"])
                        for D in DIST_VALUES), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("saved %s" % args.out)


if __name__ == "__main__":
    main()