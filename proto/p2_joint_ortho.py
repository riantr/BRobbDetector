"""P2: jointly-orthogonal rectangle estimation.

The shipped pipeline fits the 4 sides INDEPENDENTLY (TLS on the red pixels in a
band), intersects adjacent side lines into 4 corners, and then PROJECTS the
result onto an orthogonal quad (`_enforce(q, "ortho_edge")`).  That
fit-then-project order is not the least-squares solution of the constrained
problem.

P2 replaces it with a JOINT estimate: minimise, over the individual red-pixel
clouds of the 4 sides, the squared perpendicular residual subject to the quad
being an EXACT rectangle.  A rectangle has 5 parameters; for a fixed
orientation theta the optimal offsets are closed-form (the weighted mean of
each side's pixels in the rotated frame), so the sweep is 1-D over theta only.
The result is exactly orthogonal by construction, so the projection step
becomes a no-op.

  joint_A   refine -> JOINT fit (fallback: enforce) -> edge rule -> support gate
  joint_B   the shipped path, then a JOINt polish on the accepted quad, with
            the edge rule and support gate re-applied

Scored on the real 500 images with the GT-corrected metric.  P3 showed the GT
is precise (best-orthogonal-rect minimax residual p50 0.50 / max 1.32 px) and
100% of GT is reachable at D=2, so any precision gain here is real headroom:
the detector currently leaves 44 boxes in the (3,5] px band, 25 in (5,8] and
12 in (8,12].

Usage: python proto/p2_joint_ortho.py --workers 12 --out benchmark/p2_joint.json
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
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from quad_ortho import quad_on_boundary
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect_holes,
                           detect_edgefit, _holes_of, _quad_of, _refine_sides,
                           _gate, _enforce, _on_edge, _red_support, _hole_side_lines)

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]


def zcross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def joint_ortho_rect(mask, q, band=3.0, step_deg=0.25, span_deg=12.0,
                     min_pts=6, tol=2.5):
    """Exact-rectangle least-squares fit to the red pixels around `q`.

    For a fixed theta the two u-sides are parallel with offsets = the weighted
    mean of their pixels' u-coordinate, and likewise the two v-sides; the
    perpendicularity is exact.  theta is swept, the total per-side mean squared
    residual decides.  Returns None when a side has too few pixels, the fit is
    degenerate, or the residual is too large (the same guard the shipped
    refine step uses).
    """
    H, W = mask.shape
    q = np.asarray(q, float)
    offs = np.arange(-band, band + 0.5, 1.0)
    clouds = []
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 6:
            return None
        u = d / L
        n = np.array([-u[1], u[0]])
        ts = np.linspace(0.0, 1.0, max(int(L), 8))
        base = a[None, :] + ts[:, None] * d[None, :]
        pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
        xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
        sel = mask[ys, xs] > 0
        Px = pts[sel]
        if len(Px) < min_pts:
            return None
        clouds.append(Px)

    d0 = q[2] - q[1]
    th0 = float(np.arctan2(d0[1], d0[0]))
    steps = int(2 * span_deg / step_deg)
    best = None
    best_res = np.inf
    for k in range(steps + 1):
        th = th0 + np.radians(-span_deg + k * step_deg)
        u = np.array([np.cos(th), np.sin(th)])
        v = np.array([-np.sin(th), np.cos(th)])
        pu0 = clouds[0] @ u
        pu2 = clouds[2] @ u
        pv1 = clouds[1] @ v
        pv3 = clouds[3] @ v
        u0 = float(pu0.mean())
        u2 = float(pu2.mean())
        v1 = float(pv1.mean())
        v3 = float(pv3.mean())
        if abs(u0 - u2) < 3.0 or abs(v1 - v3) < 3.0:
            continue
        res = (((pu0 - u0) ** 2).mean() + ((pu2 - u2) ** 2).mean() +
               ((pv1 - v1) ** 2).mean() + ((pv3 - v3) ** 2).mean())
        if res < best_res:
            best_res = res
            best = (u, v, u0, u2, v1, v3)
    if best is None:
        return None
    u, v, u0, u2, v1, v3 = best
    # residual guard on the worst side
    for pu, off in ((clouds[0] @ u, u0), (clouds[2] @ u, u2),
                    (clouds[1] @ v, v1), (clouds[3] @ v, v3)):
        if float(np.percentile(np.abs(pu - off), 90)) > tol:
            return None
    lo_u, hi_u = (u0, u2) if u0 < u2 else (u2, u0)
    lo_v, hi_v = (v1, v3) if v1 < v3 else (v3, v1)
    corners = np.array([[lo_u, lo_v], [hi_u, lo_v], [hi_u, hi_v], [lo_u, hi_v]])
    rect = np.outer(corners[:, 0], u) + np.outer(corners[:, 1], v)
    # keep the input winding (the ortho enforcement depends on edge order)
    if zcross(q[1] - q[0], q[2] - q[1]) * zcross(rect[1] - rect[0], rect[2] - rect[1]) < 0:
        rect = rect[::-1]
    return rect


def detect_holes_joint(img, p, cfg, pipe, mask, mode):
    """detect_holes() with the joint orthogonal fit inserted."""
    H, W = img.shape[:2]
    holes = _holes_of(mask, cfg)
    out = []
    for h in holes:
        q = _quad_of(h, mask, cfg)
        if q is None:
            continue
        if cfg.get("hole_refine", 0):
            r = _refine_sides(mask, q, band=cfg.get("hole_refine_band", 4.0),
                              min_pts=int(cfg.get("hole_refine_pts", 8)),
                              tol=cfg.get("hole_refine_tol", 2.0))
            if r is not None:
                q = r
        if mode == "A":
            j = joint_ortho_rect(mask, q, band=cfg.get("hole_refine_band", 3.0))
            if j is not None:
                q = j
        q = _gate(q, W, H, p)
        if q is None:
            continue
        q = _enforce(q, pipe["ortho"])
        if mode == "B":
            j = joint_ortho_rect(mask, q, band=cfg.get("hole_refine_band", 3.0))
            if j is not None:
                q = j
        if _on_edge(q, W, H, p):
            continue
        if cfg.get("min_red_support", 0.0) > 0:
            if _red_support(mask, q) < cfg["min_red_support"]:
                continue
        out.append(np.asarray(q, np.int32))
    return out


def _work(task):
    iid, mode = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, mode, []
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    raw = []
    if mode == "base":
        raw += detect_holes(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
    else:
        raw += detect_holes_joint(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m, mode)
    raw += detect_edgefit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
    det = dedup_keep_order(raw, PIPE_DEFAULT.get("dedup_iou", P["dedup_iou"]))
    return iid, mode, [np.asarray(q, np.int32) for q in det]


def score(det_map, ids, gt_map, exp):
    masks = {}
    for iid in ids:
        if not det_map.get(iid):
            continue
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        masks[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                np.ones((3, 3), np.uint8), iterations=3) > 0
    res = {}
    for D in DIST_VALUES:
        tp = fp = unl = ndet = 0
        for iid in ids:
            det = det_map.get(iid, [])
            gt = gt_map.get(iid, [])
            ndet += len(det)
            t, unmatched = match_boxes_point(det, gt, float(D))
            tp += t
            md = masks.get(iid)
            cn = exp.get(iid, 0)
            for di in unmatched:
                if md is not None and is_real_unlabeled(det[di], cn, len(gt), md):
                    unl += 1
                else:
                    fp += 1
        ngt = sum(len(gt_map.get(i, [])) for i in ids)
        rec = tp / max(ngt, 1) * 100
        prec = tp / max(tp + fp, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        res[str(D)] = {"tp": tp, "true_fp": fp, "real_unl": unl, "n_det": ndet,
                       "recall": round(rec, 1), "precision": round(prec, 1),
                       "corr_f1": round(f1, 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/p2_joint.json")
    ap.add_argument("--modes", default="base,A,B")
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
    out = {}
    for mode in modes:
        det_map = {}
        with Pool(args.workers) as pool:
            for k, (iid, _m, det) in enumerate(pool.imap_unordered(
                    _work, [(i, mode) for i in ids], chunksize=3)):
                det_map[iid] = det
                if (k + 1) % 200 == 0:
                    print("  %s %d/%d" % (mode, k + 1, len(ids)), flush=True)
        r = score(det_map, ids, gt_map, exp)
        out[mode] = r
        print("  %-6s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d  corr_f1@5=%.1f"
              % (mode, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"],
                 r["5"]["true_fp"], r["8"]["tp"], r["8"]["true_fp"],
                 r["12"]["tp"], r["12"]["true_fp"], r["5"]["n_det"],
                 r["5"]["corr_f1"]), flush=True)

    if "base" in out:
        b = out["base"]
        print("\n== delta vs base ==")
        for m in modes:
            if m == "base":
                continue
            d = out[m]
            print("  %-6s dTP@3 %+4d  dTP@5 %+4d  dTP@8 %+4d  dTrueFP@5 %+4d  "
                  "dcorr_f1@5 %+5.1f"
                  % (m, d["3"]["tp"] - b["3"]["tp"], d["5"]["tp"] - b["5"]["tp"],
                     d["8"]["tp"] - b["8"]["tp"],
                     d["5"]["true_fp"] - b["5"]["true_fp"],
                     d["5"]["corr_f1"] - b["5"]["corr_f1"]))
    json.dump({"results": out, "modes": modes},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()