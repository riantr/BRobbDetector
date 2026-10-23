"""LOOP-80: factorial panel for the DoubleML analysis of the pm_v9 -> pm_v10 delta.

The pm_v10 change (hole_min_side 8->14, hole_fit_pass 0->2, hole_refine 0->1)
was adopted because it won the aggregate benchmark.  An aggregate delta does
not say WHICH knob did the work, and it cannot say whether a knob helps on
every image or only on some of them.

This script builds the panel that a DoubleML analysis needs:  every image is
run through EVERY cell of a 2^4 factorial design over the four knobs that
separated pm_v9 from pm_v10 (+ the `min_red_support=0.6` candidate that the
held-out check rejected).  One row per (image, cell):

    outcome   tp3/fp3/tp5/fp5/tp8/fp8/tp12/fp12/ndet  (per-image counts)
    treatment t_minside, t_fitpass, t_refine, t_sup60 (0/1)
    controls  image-level covariates (pre-treatment, config free)

The design is a COMPLETE BLOCK design (block = image), so the effect of a
knob is identified from the WITHIN-image contrast and the image-level
variance is differenced out.  The corner cells reproduce pm_v9 and pm_v10
exactly, which is the panel's own correctness check.

Usage:
    python proto/dml_panel.py --dataset real --out benchmark/dml_panel_real.json
    python proto/dml_panel.py --dataset emul --n 2500 --out benchmark/dml_panel_emul.json
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
from eval_emulation import side_support
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _holes_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
EMUL_IMG = os.path.join(ROOT, "data", "Image@emul4")
EMUL_LAB = os.path.join(ROOT, "data", "Label@emul4")
DIST_VALUES = [3, 5, 8, 12]

# (treatment column, config key, level 0, level 1)
FACTORS = [
    ("t_minside", "hole_min_side", 8.0, 14.0),
    ("t_fitpass", "hole_fit_pass", 0.0, 2.0),
    ("t_refine", "hole_refine", 0.0, 1.0),
    ("t_sup60", "min_red_support", 0.5, 0.6),
]

TAGS = {
    "t_minside": ("ms8", "ms14"),
    "t_fitpass": ("fp0", "fp2"),
    "t_refine": ("rf0", "rf1"),
    "t_sup60": ("sup50", "sup60"),
}

_CELLS = []
_EMUL = False


def build_cells():
    """The 2^4 design cells, each with the hole config it runs."""
    cells = []
    for bits in range(1 << len(FACTORS)):
        cfg = dict(HOLE_DEFAULT)
        levels = {}
        tags = []
        for k, (tcol, key, v0, v1) in enumerate(FACTORS):
            bit = (bits >> k) & 1
            cfg[key] = v1 if bit else v0
            levels[tcol] = bit
            tags.append(TAGS[tcol][bit])
        cells.append({"name": "-".join(tags), "bits": bits, "levels": levels,
                      "cfg": cfg})
    return cells


def _init(cells, emul):
    global _CELLS, _EMUL
    _CELLS = cells
    _EMUL = emul


# ----------------------------------------------------------------- helpers

def _prod_mask(img):
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    return m


def _loose_mask(img):
    return cv2.dilate(red_mask(img, 80, 110, 10),
                      np.ones((3, 3), np.uint8), iterations=3) > 0


def _quad_stats(q):
    q = np.asarray(q, float)
    sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    area = abs(float(sum(q[i][0] * q[(i + 1) % 4][1] -
                         q[(i + 1) % 4][0] * q[i][1] for i in range(4)))) / 2.0
    return min(sides), area


def _covariates(img, mask, gt):
    """Image level covariates - computed from the mask and the GT only.

    They do NOT depend on the treatment, so they are legitimate controls.
    """
    H, W = mask.shape
    ms, ar, sup_min, sup_mean = [], [], [], []
    for g in gt:
        mn, area = _quad_stats(g)
        ms.append(mn)
        ar.append(area)
        sups = side_support(mask, g)
        sup_min.append(float(min(sups)))
        sup_mean.append(float(np.mean(sups)))
    holes = _holes_of(mask, {"hole_min_comp_area": 60.0})
    hms, hfill = [], []
    for h in holes:
        area = cv2.contourArea(h)
        (_, _), (rw, rh), _ = cv2.minAreaRect(h)
        hms.append(float(min(rw, rh)))
        hfill.append(float(area / max(rw * rh, 1e-6)))
    n_red_comp = cv2.connectedComponents(mask, connectivity=8)[0] - 1
    n = len(gt)
    cov = {
        "n_gt": float(n),
        "gt_min_side_mean": float(np.mean(ms)) if ms else 0.0,
        "gt_min_side_min": float(np.min(ms)) if ms else 0.0,
        "gt_area_mean": float(np.mean(ar)) if ar else 0.0,
        "gt_area_max": float(np.max(ar)) if ar else 0.0,
        "gt_sup_mean": float(np.mean(sup_mean)) if sup_mean else 0.0,
        "gt_supmin_mean": float(np.mean(sup_min)) if sup_min else 0.0,
        "gt_supmin_min": float(np.min(sup_min)) if sup_min else 0.0,
        "gt_frac_sup60": float(np.mean([1.0 if s >= 0.6 else 0.0
                                        for s in sup_min])) if sup_min else 0.0,
        "img_w": float(W),
        "img_h": float(H),
        "img_ratio": float(W) / float(max(H, 1)),
        "red_ratio": float(np.count_nonzero(mask)) / float(max(H * W, 1)),
        "n_red_comp": float(n_red_comp),
        "n_holes": float(len(holes)),
        "hole_ms_mean": float(np.mean(hms)) if hms else 0.0,
        "hole_fill_mean": float(np.mean(hfill)) if hfill else 0.0,
    }
    return cov


def _score(det, gt, loose, csv_n):
    """Per-image counts of the point-match metric (real data: GT corrected)."""
    r = {"ndet": len(det)}
    for D in DIST_VALUES:
        tp, unmatched = match_boxes_point(det, gt, float(D))
        tfp = 0
        unl = 0
        for di in unmatched:
            if loose is not None and is_real_unlabeled(det[di], csv_n, len(gt),
                                                       loose):
                unl += 1
            else:
                tfp += 1
        r["tp%d" % D] = tp
        r["fp%d" % D] = tfp
        if D == 5:
            r["unl5"] = unl
    return r


# ----------------------------------------------------------------- workers

def _work_real(task):
    iid, gt, csv_n = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return None
    mask = _prod_mask(img)
    loose = _loose_mask(img)
    cov = _covariates(img, mask, gt)
    cells = {}
    for c in _CELLS:
        det = [np.asarray(q, np.int32)
               for q in detect(img, P, c["cfg"], PIPE_DEFAULT, mask=mask)]
        cells[c["name"]] = _score(det, gt, loose, csv_n)
    return {"iid": iid, "cov": cov, "cells": cells}


def _work_emul(task):
    from eval_emulation import color_mask
    stem, gt = task
    img = cv2.imread(os.path.join(EMUL_IMG, stem + ".jpg"))
    if img is None:
        return None
    mask = color_mask(img, "red", P, None)
    if P["denoise_min_area"] > 0:
        mask = denoise(mask, P["denoise_min_area"])
    cov = _covariates(img, mask, gt)
    cells = {}
    for c in _CELLS:
        det = [np.asarray(q, np.int32)
               for q in detect(img, P, c["cfg"], PIPE_DEFAULT, mask=mask)]
        # the emulation GT is COMPLETE -> every unmatched detection is a FP
        cells[c["name"]] = _score(det, gt, None, 0)
    return {"iid": stem, "cov": cov, "cells": cells}


# ----------------------------------------------------------------- build

def real_tasks(seed=0, n=500):
    from sweep_real import sample_ids
    ids = sample_ids(seed, n)
    exp = load_expected()
    tasks = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = [g for g in load_boxes_any(RAW, iid)
              if not quad_on_boundary(g, W, H, 0.0)]
        tasks.append((iid, gt, exp.get(iid, 0)))
    return tasks


def emul_tasks(n=0):
    from eval_emulation import load_gt
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in os.listdir(EMUL_LAB) if p.endswith(".json"))
    if n:
        stems = stems[:n]
    tasks = []
    for stem in stems:
        gt = [g for g, t in load_gt(stem, EMUL_LAB) if t == "红"]
        tasks.append((stem, gt))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["real", "emul"], default="real")
    ap.add_argument("--n", type=int, default=0, help="0 = all / 500 for real")
    ap.add_argument("--seed", type=int, default=0, help="real: 0 = tuning set")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cells = build_cells()
    if args.dataset == "real":
        tasks = real_tasks(args.seed, args.n or 500)
        work = _work_real
    else:
        tasks = emul_tasks(args.n)
        work = _work_emul
    n_gt = sum(len(t[1]) for t in tasks)
    print("%s: %d images, %d GT boxes, %d cells" %
          (args.dataset, len(tasks), n_gt, len(cells)), flush=True)

    t0 = time.time()
    per_image = []
    with Pool(args.workers, initializer=_init, initargs=(cells, args.dataset == "emul")) as pool:
        for k, res in enumerate(pool.imap_unordered(work, tasks, chunksize=4)):
            if res is not None:
                per_image.append(res)
            if (k + 1) % 100 == 0:
                print("  %d/%d  %.1fs" % (k + 1, len(tasks), time.time() - t0),
                      flush=True)
    print("  detect done in %.1fs" % (time.time() - t0), flush=True)

    rows = []
    for r in per_image:
        for c in cells:
            s = r["cells"][c["name"]]
            row = {"iid": r["iid"], "cell": c["name"]}
            row.update(c["levels"])
            row.update(r["cov"])
            row.update(s)
            rows.append(row)

    # the aggregate of every cell - the correctness check of the panel
    print("\n  %-26s %6s %6s %6s %6s %6s %6s %6s" %
          ("cell", "ndet", "tp3", "fp3", "tp5", "fp5", "tp8", "fp8"))
    summary = {}
    for c in cells:
        sub = [r for r in rows if r["cell"] == c["name"]]
        agg = {k: int(sum(r[k] for r in sub)) for k in
               ("ndet", "tp3", "fp3", "tp5", "fp5", "tp8", "fp8", "tp12", "fp12")}
        summary[c["name"]] = agg
        print("  %-26s %6d %6d %6d %6d %6d %6d %6d" %
              (c["name"], agg["ndet"], agg["tp3"], agg["fp3"], agg["tp5"],
               agg["fp5"], agg["tp8"], agg["fp8"]))

    out = args.out or os.path.join(ROOT, "benchmark",
                                   "dml_panel_%s.json" % args.dataset)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                   "n_images": len(tasks), "n_gt": n_gt,
                   "factors": [{"col": c[0], "key": c[1], "v0": c[2], "v1": c[3]}
                               for c in FACTORS],
                   "cells": [{"name": c["name"], "levels": c["levels"]}
                             for c in cells],
                   "cell_summary": summary,
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\nsaved %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()