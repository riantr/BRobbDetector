"""Hole level feature profile: what makes a red-mask hole produce a FP?

The real data's false positives are (almost) all produced by the `hole` family
and they overlap a real annotation - the leader arrow of the annotation lands
on a box corner and turns the hole of the red mask into a spiky polygon.  The
forced rectangle then has a corner pulled tens of pixels away.

This script builds, for every hole of every real image, the quad the pipeline
would emit and the shape features of the hole itself, and labels the hole
TP / FP by matching its quad against the GT.  The features are then compared
between the two populations, so that a discriminating gate can be designed.

Features per hole:
  fill      - contourArea / minAreaRect area (the existing `hole_fill` gate)
  edges     - number of vertices of approxPolyDP(eps = frac * perimeter)
  hull      - contourArea / convexHull area (1.0 = convex, < 1 = spiky)
  perim     - perimeter / perimeter of the minAreaRect (1.0 = clean)
  side_rms  - RMS deviation of the hole contour from its minAreaRect edge
  min_side  - the shorter side of the minAreaRect

Usage:
    python proto/hole_feature_profile.py --out benchmark/real_hole_features.json
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
from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import match_boxes_point
from quad_ortho import quad_on_boundary
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _quad_of,
                           _side_ok, _gate, _enforce, _on_edge)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"

_CFG = {}


def _init(cfg):
    global _CFG
    _CFG = cfg


def _features(hole):
    """Shape features of one hole contour."""
    area = cv2.contourArea(hole)
    rect = cv2.minAreaRect(hole)
    (cx, cy), (rw, rh), _ = rect
    perim = float(cv2.arcLength(hole, True))
    out = {"area": float(area),
           "min_side": float(min(rw, rh)),
           "max_side": float(max(rw, rh)),
           "fill": float(area / max(rw * rh, 1e-6)),
           "perim_ratio": float(perim / max(2.0 * (rw + rh), 1e-6))}
    hull = cv2.convexHull(hole)
    ha = cv2.contourArea(hull)
    out["hull"] = float(area / max(ha, 1e-6))
    for eps in (0.01, 0.02, 0.04, 0.08):
        ap = cv2.approxPolyDP(hole, eps * perim, True)
        out["edges_%g" % eps] = int(len(ap))
    # RMS deviation of the contour from the minAreaRect edge
    pts = hole.reshape(-1, 2).astype(np.float64)
    box = cv2.boxPoints(rect).astype(np.float64)
    c = pts.mean(axis=0)
    d = 0.0
    for i in range(4):
        a = box[i]
        b = box[(i + 1) % 4]
        e = b - a
        L = float(np.linalg.norm(e))
        if L < 1e-6:
            continue
        n = np.array([-e[1], e[0]]) / L
        d += float(np.sum(np.abs((pts - a) @ n))) * L
    out["side_rms"] = float(d / max(perim, 1e-6))
    return out


def _work(iid):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, []
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    m = denoise(m, P["denoise_min_area"])
    H, W = m.shape[:2]
    holes = _holes_of(m, _CFG)
    out = []
    for h in holes:
        f = _features(h)
        q = _quad_of(h, m, _CFG)
        if q is None:
            f["quad"] = None
            out.append(f)
            continue
        q = _gate(q, W, H, P)
        if q is None:
            f["quad"] = None
            out.append(f)
            continue
        q = _enforce(q, PIPE_DEFAULT["ortho"])
        if _on_edge(q, W, H, P):
            f["quad"] = None
            out.append(f)
            continue
        f["quad"] = [int(round(v)) for pt in q for v in pt]
        out.append(f)
    return iid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark",
                                                  "real_hole_features.json"))
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    cfg = dict(HOLE_DEFAULT)
    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            gt_map[iid] = []
            continue
        H, W = img.shape[:2]
        gt_map[iid] = [g for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]

    recs = []
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for iid, holes in pool.imap_unordered(_work, ids, chunksize=4):
            gt = gt_map[iid]
            quads = []
            idxs = []
            for k, f in enumerate(holes):
                if f.get("quad") is None:
                    continue
                q = np.array(f["quad"], np.int32).reshape(4, 2)
                quads.append(q)
                idxs.append(k)
            tp_set = set()
            if quads:
                # which quads are TPs (a quad may match a GT that another quad
                # already claimed -> greedy, same order as the pipeline)
                tp, unmatched = match_boxes_point(quads, gt, 5.0)
                uset = set(unmatched)
                for k, i in enumerate(idxs):
                    f = holes[i]
                    f["kind"] = "fp" if k in uset else "tp"
                    if k in uset:
                        tp_set.add(i)
            for k, f in enumerate(holes):
                f["iid"] = iid
                f.setdefault("kind", "nquad")
                recs.append(f)

    def stat(key, rows):
        v = np.asarray([r[key] for r in rows], float)
        if len(v) == 0:
            return {}
        return {"n": len(v),
                "p5": round(float(np.percentile(v, 5)), 3),
                "p25": round(float(np.percentile(v, 25)), 3),
                "p50": round(float(np.percentile(v, 50)), 3),
                "p75": round(float(np.percentile(v, 75)), 3),
                "p95": round(float(np.percentile(v, 95)), 3),
                "mean": round(float(v.mean()), 3)}

    groups = {"tp": [r for r in recs if r["kind"] == "tp"],
              "fp": [r for r in recs if r["kind"] == "fp"],
              "nquad": [r for r in recs if r["kind"] == "nquad"]}
    keys = ["fill", "hull", "perim_ratio", "side_rms", "min_side", "max_side",
            "area", "edges_0.01", "edges_0.02", "edges_0.04", "edges_0.08"]
    out = {"counts": {k: len(v) for k, v in groups.items()},
           "stats": {g: {k: stat(k, rows) for k in keys}
                     for g, rows in groups.items()}}
    print("holes: %s" % out["counts"])
    for g in ("tp", "fp", "nquad"):
        print("\n %s:" % g)
        for k in keys:
            s = out["stats"][g].get(k)
            if s:
                print("   %-12s n=%5d p5=%8.3f p25=%8.3f p50=%8.3f p75=%8.3f p95=%8.3f"
                      % (k, s["n"], s["p5"], s["p25"], s["p50"], s["p75"], s["p95"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()