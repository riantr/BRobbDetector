"""LOOP-82: box level DoubleML analysis - what makes a GT box MISSED?

The image level analysis (proto/dml_analyze.py) says which CONFIG KNOB does
the work.  This script asks the complementary question about the DATA: for
every GT box of the 500 real benchmark images, which property of the box
causes the detector to miss it?

Panel: one row per GT box (1076 rows).

    outcome   det5   1 = the box was detected at D=5 under the pm_v10 config
    treatment box level features (continuous):
                 box_sup_min   the MINIMUM per-side red coverage of the box
                               perimeter - "was the box drawn in red at all"
                 box_sup_mean  the mean per-side red coverage
                 box_min_side  the shorter side of the box
                 box_area      the area of the box
                 box_edge_dist the distance of the box to the image border
                 hole_ms       the shorter side of the hole that covers the box
                 hole_fill     the fill ratio of that hole
    controls  image level covariates (n_gt, red ratio, ...) - the box level
              features are correlated with each other (a small box has a small
              hole), so the DML nuisance functions have to take that out.

The panel is observational, so the estimates are conditional associations
under the DML assumption (no unmeasured confounding of the box features given
the image covariates), not interventions.  The annotation quality features
(box_sup_*) are measured on the INPUT image and are therefore pre-treatment
with respect to the detector.

Usage:
    python proto/dml_boxlevel.py --out benchmark/dml_boxlevel_real.json
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from eval_emulation import side_support
from eval_pointmatch import match_boxes_point
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _holes_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
BOX_FEATURES = ["box_sup_min", "box_sup_mean", "box_min_side", "box_area",
                "box_edge_dist", "hole_ms", "hole_fill"]


def _prod_mask(img):
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    return m


def _box_geom(q):
    q = np.asarray(q, float)
    sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    area = abs(float(sum(q[i][0] * q[(i + 1) % 4][1] -
                         q[(i + 1) % 4][0] * q[i][1] for i in range(4)))) / 2.0
    return min(sides), area


def _hole_of(holes, box):
    """The hole that covers the box (largest overlap of the minAreaRect)."""
    best = None
    best_ov = 0.0
    bq = np.asarray(box, float)
    bx0, by0 = bq.min(axis=0)
    bx1, by1 = bq.max(axis=0)
    barea = max((bx1 - bx0) * (by1 - by0), 1e-6)
    for h in holes:
        rect = cv2.minAreaRect(h)
        (cx, cy), (rw, rh), _ = rect
        hx0, hy0 = cx - rw / 2.0, cy - rh / 2.0
        hx1, hy1 = cx + rw / 2.0, cy + rh / 2.0
        ov = max(0.0, min(bx1, hx1) - max(bx0, hx0)) * \
            max(0.0, min(by1, hy1) - max(by0, hy0))
        if ov / barea > best_ov:
            best_ov = ov / barea
            best = (h, rect, ov / barea)
    return best


def _work(task):
    iid, gt = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return []
    H, W = img.shape[:2]
    mask = _prod_mask(img)
    holes = _holes_of(mask, {"hole_min_comp_area": 60.0})
    det = [np.asarray(q, np.int32)
           for q in detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]
    tp, unmatched = match_boxes_point(det, gt, 5.0)
    matched = set(range(len(det))) - set(unmatched)
    # which GT boxes were matched (greedy, same order as match_boxes_point)
    hit = _matched_gt(det, gt, 5.0)
    rows = []
    for gi, g in enumerate(gt):
        q = np.asarray(g, float)
        ms, area = _box_geom(g)
        sups = side_support(mask, g)
        ex = min(q[:, 0].min(), q[:, 1].min(), W - 1 - q[:, 0].max(),
                 H - 1 - q[:, 1].max())
        h = _hole_of(holes, g)
        if h is not None:
            (_, rect, cov) = h
            (_, _), (rw, rh), _ = rect
            hole_ms = float(min(rw, rh))
            hole_fill = float(cv2.contourArea(h[0]) / max(rw * rh, 1e-6))
        else:
            hole_ms = 0.0
            hole_fill = 0.0
        rows.append({
            "iid": iid, "gt_idx": gi,
            "det5": int(gi in hit),
            "box_sup_min": float(min(sups)),
            "box_sup_mean": float(np.mean(sups)),
            "box_min_side": float(ms),
            "box_area": float(area),
            "box_edge_dist": float(max(ex, 0.0)),
            "hole_ms": hole_ms,
            "hole_fill": hole_fill,
            "n_gt": float(len(gt)),
            "img_w": float(W), "img_h": float(H),
            "red_ratio": float(np.count_nonzero(mask)) / float(max(H * W, 1)),
            "n_red_comp": float(cv2.connectedComponents(mask, connectivity=8)[0] - 1),
            "n_holes": float(len(holes)),
        })
    return rows


def _matched_gt(det, gt, D):
    """The set of matched GT indices (same greedy order as match_boxes_point)."""
    from eval_pointmatch import corner_match_distance
    cand = []
    for di in range(len(det)):
        for gj in range(len(gt)):
            d = corner_match_distance(det[di], gt[gj])
            if d <= D:
                cand.append((d, di, gj))
    cand.sort()
    used_d, used_g, hit = set(), set(), set()
    for d, di, gj in cand:
        if di in used_d or gj in used_g:
            continue
        used_d.add(di)
        used_g.add(gj)
        hit.add(gj)
    return hit


# ------------------------------------------------------------------ dml

def _rf(kind="reg"):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if kind == "reg":
        return RandomForestRegressor(n_estimators=100, max_depth=8,
                                     min_samples_leaf=10, random_state=42,
                                     n_jobs=-1)
    return RandomForestClassifier(n_estimators=100, max_depth=8,
                                  min_samples_leaf=10, random_state=42,
                                  n_jobs=-1)


def dml_binary(df, y, dcols, xcols, n_rep=2, n_folds=5):
    from doubleml import DoubleMLData, DoubleMLPLR
    cols = [y] + list(dcols) + list(xcols)
    sub = df[cols].copy()
    dml_data = DoubleMLData(sub, y_col=y, d_cols=list(dcols),
                             x_cols=list(xcols))
    model = DoubleMLPLR(dml_data, _rf("clf"), _rf("reg"), n_folds=n_folds,
                         n_rep=n_rep, score="partialling out")
    model.fit()
    coef, se, p, ci = model.coef, model.se, model.pval, model.confint()
    out = {}
    for i, c in enumerate(dcols):
        out[c] = {"theta": float(np.ravel(coef)[i]),
                  "se": float(np.ravel(se)[i]),
                  "ci_low": float(np.asarray(ci)[i][0]),
                  "ci_high": float(np.asarray(ci)[i][1]),
                  "p": float(np.ravel(p)[i])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark",
                                                  "dml_boxlevel_real.json"))
    ap.add_argument("--panel-out", default=os.path.join(ROOT, "benchmark",
                                                       "dml_boxlevel_panel.json"))
    args = ap.parse_args()

    from canny_quad import get_ids
    ids = get_ids()
    if args.n:
        ids = ids[:args.n]
    tasks = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = [g for g in load_boxes_any(RAW, iid)
              if not quad_on_boundary(g, W, H, 0.0)]
        tasks.append((iid, gt))
    print("box level panel: %d images, %d GT boxes" %
          (len(tasks), sum(len(t[1]) for t in tasks)), flush=True)

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=4)):
            rows.extend(r)
            if (k + 1) % 100 == 0:
                print("  %d/%d  %.1fs" % (k + 1, len(tasks), time.time() - t0),
                      flush=True)
    df = pd.DataFrame(rows)
    print("  panel built in %.1fs  (%d rows, detected %d)" %
          (time.time() - t0, len(df), int(df["det5"].sum())), flush=True)
    df.to_json(args.panel_out, orient="records", force_ascii=False)

    # descriptive: the miss rate by annotation quality
    print("\n  miss rate by the min side red support of the GT box:")
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        s = df[(df.box_sup_min >= lo) & (df.box_sup_min < hi)]
        if not len(s):
            continue
        print("    support [%.1f,%.1f): %5d boxes  detected %5d  miss %5.1f%%" %
              (lo, hi, len(s), int(s.det5.sum()),
               100.0 * (1 - s.det5.mean())))
    ceiling = df[df.box_sup_min >= 0.6]
    print("    ceiling (support>=0.6): %d boxes, detected %d (%.1f%%)" %
          (len(ceiling), int(ceiling.det5.sum()), 100.0 * ceiling.det5.mean()))

    ctrl = [c for c in df.columns
            if c.startswith(("n_", "img_", "red_")) and df[c].std() > 0]
    print("\n  controls: %s" % ", ".join(ctrl), flush=True)

    results = {"n_boxes": int(len(df)), "n_images": len(tasks),
               "n_detected": int(df.det5.sum()), "controls": ctrl,
               "features": {}}
    for f in BOX_FEATURES:
        if df[f].std() == 0:
            continue
        x = [c for c in BOX_FEATURES if c != f] + ctrl
        try:
            r = dml_binary(df, "det5", [f], x)
        except Exception as exc:
            print("  %-14s FAILED: %s" % (f, str(exc)[:160]), flush=True)
            continue
        sd = float(df[f].std())
        e = r[f]
        results["features"][f] = {
            "dml": e, "std": sd,
            "effect_per_sd": e["theta"] * sd,
            "effect_per_sd_ci": [e["ci_low"] * sd, e["ci_high"] * sd]}
        print("  %-14s theta=%+0.5f [%+0.5f,%+0.5f] p=%.4f | per 1 SD (%0.1f): "
              "%+0.4f [%+0.4f,%+0.4f]" %
              (f, e["theta"], e["ci_low"], e["ci_high"], e["p"], sd,
               e["theta"] * sd, e["ci_low"] * sd, e["ci_high"] * sd), flush=True)

    # multi treatment version (all box features at once)
    try:
        r = dml_binary(df, "det5", BOX_FEATURES, ctrl)
        results["multi"] = r
        print("\n  multi treatment:")
        for c, e in r.items():
            sd = float(df[c].std())
            print("    %-14s theta=%+0.5f [%+0.5f,%+0.5f] p=%.4f | per 1 SD: "
                  "%+0.4f" % (c, e["theta"], e["ci_low"], e["ci_high"], e["p"],
                              e["theta"] * sd), flush=True)
    except Exception as exc:
        print("  multi FAILED: %s" % str(exc)[:160], flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()