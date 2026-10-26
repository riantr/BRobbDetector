"""LOOP-44: do the two OptiNav algorithms transfer?

Source: https://www.optinav.info
  * Find_Ridges.java (Bob Dougherty, 2014-04-20, rev 2014-05-02)
        https://www.optinav.info/Find_Ridges.htm
        A Steger-style ridge detector: Gaussian pre-filter, 2D Hessian by
        finite differences, the SMALLER eigenvalue of the Hessian must be
        negative (a bright ridge), a non-maximum-suppression test along the
        ridge direction, and finally a thinning step that keeps at most
        n/minThin ridge points, the strongest ones first (criterion = -lm).
  * GCNR_.java (Bob Dougherty, 2026-06-05)
        https://www.optinav.info/Generalized_Contrast_Noise.htm
        The generalized Contrast-to-Noise Ratio (Rodriguez-Molares et al.,
        IEEE TUFFC 67(4):745-759, 2020): for two ROIs it builds the two
        histograms over a SHARED range, normalises them and returns
        gCNR = 1 - OVL, OVL = sum_i min(p0_i, p1_i).
        It is a MEASURE, not a segmentation: the transfer is to use it as a
        per-box plausibility score (a gate) instead of / next to the binary
        red support.

Both are mapped onto concrete variants of the CURRENT pipeline
(red mask -> holes -> side fit -> ortho gate -> red-support gate) and
measured on the 500-image real benchmark, which is the only gate.

  part G  Find Ridges: ridge mask, union mask, ridge support, ridge side fit
  part H  gCNR as a per-box score: AUC vs the binary red support + curves

Usage:
    python proto/optinav_transfer_eval.py --part G --workers 14
    python proto/optinav_transfer_eval.py --part H --workers 14
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
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from cv_hybrid import dedup_keep_order
from eval_pointmatch import match_boxes_point
from quad_ortho import quad_on_boundary
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support,
                           _holes_of, _quad_of, _refine_sides, _gate, _enforce,
                           _on_edge)

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]


# ================================================================= Find Ridges
def find_ridges(gray, sigma=2.0, min_thin=10.0):
    """OptiNav Find_Ridges.java, ported to numpy (literal port of the code).

    Returns (keep, strength):
      keep      bool array, True where the pixel is a retained ridge point
      strength  -lm (the smaller Hessian eigenvalue, negated) - the value the
                plugin uses to decide which ridge points survive the thinning

    The Java code:
      * blurs with radius=sigma, sigma=sigma (a (2*sigma+1) kernel), i.e.
        OpenCV's GaussianBlur with ksize=(2r+1, 2r+1), sigma=sigma
      * Lxx, Lyy by central differences, Lxy = (p++ + p-- - p+- - p-+)/4
        (the "corrected coefficient of mixed partial derivative" of 2014-05-03)
      * lm = the smaller eigenvalue of the Hessian; the ridge condition is
        lm < 0, i.e. a BRIGHT line on a dark background
      * the eigenvector of lm gives the ridge direction; the pixel is a ridge
        point when the directional derivative is negative in +v and positive
        in -v (a local maximum along the ridge direction)
      * thinning: at most n/minThin points survive, the ones with the
        strongest ridge (largest -lm) win
    """
    g = np.asarray(gray, np.float32)
    r = max(int(round(sigma)), 1)
    p = cv2.GaussianBlur(g, (2 * r + 1, 2 * r + 1), float(sigma))

    h, w = p.shape
    Lxx = np.zeros_like(p)
    Lyy = np.zeros_like(p)
    Lxy = np.zeros_like(p)
    Lxx[:, 1:-1] = p[:, :-2] - 2.0 * p[:, 1:-1] + p[:, 2:]
    Lyy[1:-1, :] = p[:-2, :] - 2.0 * p[1:-1, :] + p[2:, :]
    Lxy[1:-1, 1:-1] = (p[2:, 2:] + p[:-2, :-2] - p[2:, :-2] - p[:-2, 2:]) / 4.0

    tm = (Lxx - Lyy) / 2.0
    tp = (Lxx + Lyy) / 2.0
    root = np.sqrt(tm * tm + Lxy * Lxy)
    lm = tp - root                     # smaller eigenvalue
    strength = -lm

    # eigenvector of lm
    den1 = Lxy * Lxy + (lm - Lxx) ** 2
    den2 = (lm - Lyy) ** 2 + Lxy * Lxy
    use1 = den1 > den2
    sq1 = np.sqrt(np.maximum(den1, 1e-12))
    sq2 = np.sqrt(np.maximum(den2, 1e-12))
    u = np.where(use1, Lxy / sq1, (lm - Lyy) / sq2)
    v = np.where(use1, (lm - Lxx) / sq1, Lxy / sq2)

    # forward/backward differences along x and y (full size)
    fx = np.zeros_like(p)
    fx[:, :-1] = p[:, 1:] - p[:, :-1]
    bx = np.zeros_like(p)
    bx[:, 1:] = p[:, :-1] - p[:, 1:]
    fy = np.zeros_like(p)
    fy[:-1, :] = p[1:, :] - p[:-1, :]
    by = np.zeros_like(p)
    by[1:, :] = p[:-1, :] - p[1:, :]
    ui = u[1:-1, 1:-1]
    vi = v[1:-1, 1:-1]
    up_x = np.where(ui >= 0, fx[1:-1, 1:-1], -bx[1:-1, 1:-1])
    um_x = np.where(ui >= 0, -bx[1:-1, 1:-1], fx[1:-1, 1:-1])
    up_y = np.where(vi >= 0, fy[1:-1, 1:-1], -by[1:-1, 1:-1])
    um_y = np.where(vi >= 0, -by[1:-1, 1:-1], fy[1:-1, 1:-1])

    derP = ui * up_x + vi * up_y
    derM = ui * um_x + vi * um_y
    keep = np.zeros_like(p, dtype=bool)
    keep[1:-1, 1:-1] = (lm[1:-1, 1:-1] < 0) & (derP < 0) & (derM > 0)

    # thinning: keep at most n/minThin points, strongest first
    n = p.size
    max_points = max(int(n / max(min_thin, 1.0)), 1)
    cnt = int(keep.sum())
    if cnt > max_points:
        idx = np.flatnonzero(keep.ravel())
        crit = strength.ravel()[idx]
        order = np.argsort(crit, kind="stable")        # ascending
        drop = order[:cnt - max_points]
        flat = keep.ravel()
        flat[idx[drop]] = False
        keep = flat.reshape(p.shape)
    return keep, strength


def ridge_mask(img, channel="r", sigma=2.0, min_thin=10.0, dilate=1):
    """Binary ridge mask of an image channel ('r', 'g', 'b' or 'gray')."""
    if channel == "gray":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img[:, :, {"r": 2, "g": 1, "b": 0}[channel]]
    keep, _ = find_ridges(g, sigma=sigma, min_thin=min_thin)
    m = keep.astype(np.uint8) * 255
    if dilate > 0:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    return m


# ==================================================================== gCNR
def gcnr(a, b, n_bins=256):
    """Generalized Contrast-to-Noise Ratio of two pixel sets (OptiNav GCNR_).

    Port of GCNR_.java: the shared range is [min(min0,min1), max(max0,max1)],
    both histograms are built on it with n_bins bins, normalised to sum 1, and
        gCNR = 1 - sum_i min(p0_i, p1_i)      (1 - the overlap coefficient)
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi <= lo:
        return 0.0
    h0, _ = np.histogram(a, bins=n_bins, range=(lo, hi))
    h1, _ = np.histogram(b, bins=n_bins, range=(lo, hi))
    p0 = h0.astype(np.float64) / max(h0.sum(), 1)
    p1 = h1.astype(np.float64) / max(h1.sum(), 1)
    ovl = float(np.minimum(p0, p1).sum())
    return 1.0 - ovl


def ring_pixels(img, quad, half=2):
    """Pixels of the stroke band around the quad perimeter (a thin ring)."""
    H, W = img.shape[:2]
    q = np.asarray(quad, float)
    out = []
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        n = max(int(L * 2), 4)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        d = (b - a)[None, :]
        u = d / max(np.linalg.norm(d), 1e-9)
        nrm = np.array([-u[0, 1], u[0, 0]])
        for s in np.arange(-half, half + 0.5, 1.0):
            pts = a[None, :] + ts[:, None] * d + s * nrm[None, :]
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            out.append(img[ys, xs])
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


def interior_pixels(img, quad, shrink=0.0):
    """Pixels strictly inside the quad (optionally shrunk towards the centre)."""
    H, W = img.shape[:2]
    q = np.asarray(quad, float)
    c = q.mean(axis=0)
    if shrink > 0:
        q = c[None, :] + (q - c[None, :]) * (1.0 - shrink)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 255)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return img[m > 0]


def outside_pixels(img, quad, grow=12):
    """Pixels of a ring OUTSIDE the quad (the surrounding context)."""
    H, W = img.shape[:2]
    q = np.asarray(quad, float)
    inner = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inner, [np.round(q).astype(np.int32)], 255)
    k = np.ones((3, 3), np.uint8)
    outer = cv2.dilate(inner, k, iterations=max(int(grow), 1))
    ring = (outer > 0) & (inner == 0)
    return img[ring]


# =============================================================== GT helpers
def perimeter_pixels(quads, H, W, step=1):
    out = np.zeros((H, W), np.uint8)
    for q in quads:
        q = np.asarray(q, float)
        for i in range(4):
            a = q[i]
            b = q[(i + 1) % 4]
            L = float(np.linalg.norm(b - a))
            n = max(int(L / step), 2)
            ts = np.linspace(0.0, 1.0, n, endpoint=False)
            pts = a[None, :] + ts[:, None] * (b - a)[None, :]
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            out[ys, xs] = 1
    return out


def gt_band(quads, H, W, half=3):
    m = np.zeros((H, W), np.uint8)
    for q in quads:
        cv2.polylines(m, [np.asarray(q, np.int32)], True, 255, 1)
    k = np.ones((2 * half + 1, 2 * half + 1), np.uint8)
    return cv2.dilate(m, k) > 0


def prod_mask(img):
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    return m


def n_holes(mask):
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return int(np.sum(hier[0, :, 3] >= 0))


# ------------------------------------------------------------- part G: masks
_GT = {}


def _init_gt(gt_map):
    global _GT
    _GT = gt_map


def _work_gmask(iid):
    """Mask quality of the Find Ridges variants (no detector)."""
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, {}
    H, W = img.shape[:2]
    quads = _GT.get(iid, [])
    per = perimeter_pixels(quads, H, W) > 0
    band = gt_band(quads, H, W)
    npix = int(per.sum())
    masks = {
        "prod_hsv": prod_mask(img),
        "ridge_r": ridge_mask(img, "r", sigma=2.0, min_thin=10.0, dilate=1),
        "ridge_r_nodil": ridge_mask(img, "r", sigma=2.0, min_thin=10.0, dilate=0),
        "ridge_gray": ridge_mask(img, "gray", sigma=2.0, min_thin=10.0, dilate=1),
    }
    masks["union_r"] = cv2.bitwise_or(masks["prod_hsv"], masks["ridge_r"])
    out = {}
    for name, m in masks.items():
        mm = m > 0
        n = int(mm.sum())
        cover = float((mm & per).sum()) / max(npix, 1)
        prec = float((mm & band).sum()) / max(n, 1)
        out[name] = {"n_px": n, "gt_cover": round(cover, 4),
                     "in_gt_band": round(prec, 4), "n_holes": n_holes(m)}
    return iid, out


def part_g_mask(ids, gt_map, workers):
    res = {}
    with Pool(workers, initializer=_init_gt, initargs=(gt_map,)) as pool:
        for k, (iid, st) in enumerate(pool.imap_unordered(_work_gmask, ids, chunksize=4)):
            res[iid] = st
            if (k + 1) % 100 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    return res


# ------------------------------------------------- part G: end-to-end runs
def _detect_holes_with(img, p, cfg, pipe, mask, support_mask=None):
    """detect_holes() with a SEPARATE mask for the red-support gate.

    Copy of scope45_holes.detect_holes (the grouping is off in the shipped
    config, so it is reproduced verbatim) with the support gate reading
    `support_mask` instead of `mask`.
    """
    H, W = img.shape[:2]
    if support_mask is None:
        support_mask = mask
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
        if pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        else:
            if not _side_ok_local(q, p):
                continue
        q = _enforce(q, pipe["ortho"])
        if _on_edge(q, W, H, p):
            continue
        if not pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        if cfg.get("min_red_support", 0.0) > 0:
            if _red_support(support_mask, q) < cfg["min_red_support"]:
                continue
        out.append(np.asarray(q, np.int32))
    return out


def _side_ok_local(q, p):
    from canny_quad import _side_ok
    return _side_ok(q, p)


def _work_g(task):
    """End-to-end detection for one Find Ridges variant."""
    iid, variant = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, variant, []
    base = prod_mask(img)
    ridge_r = ridge_mask(img, "r", sigma=2.0, min_thin=10.0, dilate=1)
    ridge_g = ridge_mask(img, "gray", sigma=2.0, min_thin=10.0, dilate=1)
    union = cv2.bitwise_or(base, ridge_r)

    if variant == "base":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=base)
    elif variant == "ridge_r":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=ridge_r)
    elif variant == "ridge_gray":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=ridge_g)
    elif variant == "union_r":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=union)
    elif variant == "support_union_r":
        det = _detect_holes_with(img, P, HOLE_DEFAULT, PIPE_DEFAULT, base,
                                  support_mask=union)
    elif variant == "support_ridge_r":
        det = _detect_holes_with(img, P, HOLE_DEFAULT, PIPE_DEFAULT, base,
                                  support_mask=ridge_r)
    elif variant == "support_ridge_only_base_mask":
        # support from the ridge map, detection from the shipped mask
        det = _detect_holes_with(img, P, HOLE_DEFAULT, PIPE_DEFAULT, base,
                                  support_mask=ridge_r)
    else:
        raise ValueError(variant)
    return iid, variant, [np.asarray(q, np.int32) for q in det]


def score_variant(det_map, ids, gt_map):
    tot = {D: {"tp": 0, "fp": 0} for D in DIST_VALUES}
    ndet = 0
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        ndet += len(det)
        for D in DIST_VALUES:
            tp, unmatched = match_boxes_point(det, gt, float(D))
            tot[D]["tp"] += tp
            tot[D]["fp"] += len(unmatched)
    ngt = sum(len(gt_map.get(i, [])) for i in ids)
    res = {}
    for D, t in tot.items():
        rec = t["tp"] / max(ngt, 1) * 100
        prec = t["tp"] / max(t["tp"] + t["fp"], 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        res[str(D)] = {"tp": t["tp"], "fp": t["fp"], "recall": round(rec, 1),
                       "precision": round(prec, 1), "f1": round(f1, 1)}
    res["n_det"] = ndet
    res["ngt"] = ngt
    return res


def part_g(ids, gt_map, workers, variants):
    out = {}
    for v in variants:
        det_map = {}
        with Pool(workers) as pool:
            for k, (iid, _v, det) in enumerate(pool.imap_unordered(
                    _work_g, [(i, v) for i in ids], chunksize=4)):
                det_map[iid] = det
                if (k + 1) % 200 == 0:
                    print("  %s %d/%d" % (v, k + 1, len(ids)), flush=True)
        out[v] = score_variant(det_map, ids, gt_map)
        r = out[v]
        print("  %-28s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d"
              % (v, r["3"]["tp"], r["3"]["fp"], r["5"]["tp"], r["5"]["fp"],
                 r["8"]["tp"], r["8"]["fp"], r["12"]["tp"], r["12"]["fp"],
                 r["n_det"]), flush=True)
    return out


# --------------------------------------------------- part H: gCNR as a score
def _work_h(task):
    """Detections of the shipped pipeline + gCNR scores."""
    iid, = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, []
    m = prod_mask(img)

    out = []
    for q in detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m):
        q = np.asarray(q, float)
        ring = ring_pixels(img, q, half=2)
        inner = interior_pixels(img, q)
        outer = outside_pixels(img, q, grow=12)
        wBGR = np.array([0.114, 0.587, 0.299])
        out.append((
            np.asarray(q, np.int32),
            _red_support(m, q),
            gcnr(ring, inner),
            gcnr(ring, outer),
            gcnr(inner, outer),
            gcnr(ring @ wBGR, inner @ wBGR),
        ))
    return iid, out


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    a = np.asarray(pos + neg, float)
    r = np.argsort(np.argsort(a)) + 1.0
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def part_h(ids, gt_map, workers):
    det_map = {}
    with Pool(workers) as pool:
        for k, (iid, det) in enumerate(pool.imap_unordered(
                _work_h, [(i,) for i in ids], chunksize=4)):
            det_map[iid] = det
            if (k + 1) % 200 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)

    keys = ["sup", "gcnr_ring_in", "gcnr_ring_out", "gcnr_in_out", "gcnr_gray"]
    tp = {k: [] for k in keys}
    fp = {k: [] for k in keys}
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        _n, unmatched = match_boxes_point([d[0] for d in det], gt, 5.0)
        uset = set(unmatched)
        for i, d in enumerate(det):
            tgt = fp if i in uset else tp
            for j, k in enumerate(keys):
                tgt[k].append(d[j + 1])
    out = {"n_tp": len(tp["sup"]), "n_fp": len(fp["sup"])}
    for k in keys:
        out["auc_" + k] = round(auc(tp[k], fp[k]), 4)
        out["tp_p50_" + k] = round(float(np.percentile(tp[k], 50)), 4)
        out["fp_p50_" + k] = round(float(np.percentile(fp[k], 50)), 4)
    for k in keys:
        rows = []
        for t in np.arange(0.0, 1.001, 0.05):
            rows.append({"thr": round(float(t), 2),
                         "tp_kept": int(sum(1 for v in tp[k] if v >= t)),
                         "fp_kept": int(sum(1 for v in fp[k] if v >= t))})
        out["curve_" + k] = rows
    return out


# ------------------------------------------ part J: redness / ridge side refit
def redness_map(img):
    """The channel on which the annotation is the strongest bright ridge."""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    return r - np.maximum(g, b)


def _refine_weighted(score, q, band=3.0, mode="centroid", min_pts=8,
                     tol=2.0, min_w=1e-3):
    """Re-fit the 4 sides of a quad on a soft score map.

    mode="centroid": sub-pixel stroke centre = the score-weighted centroid of
                     a cross-section of the band (the score is clipped at 0).
    mode="argmax"  : sub-pixel stroke centre = the parabolic sub-pixel
                     maximum of the score across the band.

    The 4 refitted lines are re-intersected; the refit is rejected when a side
    has too few usable cross-sections or the points are not collinear.
    """
    H, W = score.shape
    offs = np.arange(-band, band + 0.5, 1.0)
    lines = []
    for i in range(4):
        a = np.asarray(q[i], float)
        b = np.asarray(q[(i + 1) % 4], float)
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
        prof = score[ys, xs]                       # (nt, noff)
        keep_t = []
        for r in range(prof.shape[0]):
            row = prof[r]
            if mode == "centroid":
                w = np.maximum(row, 0)
                sw = float(w.sum())
                if sw < min_w:
                    continue
                c = float((offs * w).sum() / sw)
            else:
                j = int(np.argmax(row))
                c = float(offs[j])
                if 0 < j < len(row) - 1:
                    d1 = row[j - 1] - row[j]
                    d2 = row[j + 1] - row[j]
                    den = d1 + d2
                    if abs(row[j]) > min_w and abs(den) > 1e-9:
                        c = float(offs[j] + np.clip(0.5 * (d1 - d2) / den, -1, 1) * 0.5)
            keep_t.append(c)
        if len(keep_t) < min_pts:
            return None
        P = base[:len(keep_t)] + np.asarray(keep_t)[:, None] * n[None, :]
        mu = P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(P - mu, full_matrices=False)
        dd = vt[0]
        if float(np.dot(dd, u)) < 0:
            dd = -dd
        nn = np.array([-dd[1], dd[0]])
        resid = np.abs((P - mu) @ nn)
        if float(np.percentile(resid, 90)) > tol:
            return None
        lines.append((mu, dd))
    out = []
    for i in range(4):
        x = _line_isect_pub(*lines[i - 1], *lines[i])
        if x is None:
            return None
        out.append(x)
    res = np.array(out, float)
    if not np.isfinite(res).all():
        return None
    return res


def _line_isect_pub(p1, d1, p2, d2):
    from scope45_holes import _line_isect
    return _line_isect(p1, d1, p2, d2)


def _detect_holes_refit(img, p, cfg, pipe, mask, mode="centroid"):
    """detect_holes() with an EXTRA side-refit on the redness ridge map."""
    from scope45_holes import _side_ok
    H, W = img.shape[:2]
    score = redness_map(img)
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
        rr = _refine_weighted(score, q, band=3.0, mode=mode, min_pts=8, tol=2.0)
        if rr is not None:
            q = rr
        if pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        else:
            if not _side_ok(q, p):
                continue
        q = _enforce(q, pipe["ortho"])
        if _on_edge(q, W, H, p):
            continue
        if not pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        if cfg.get("min_red_support", 0.0) > 0:
            if _red_support(mask, q) < cfg["min_red_support"]:
                continue
        out.append(np.asarray(q, np.int32))
    return out


def _work_j(task):
    iid, variant = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, variant, []
    m = prod_mask(img)
    if variant == "base":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    elif variant == "redness_centroid":
        det = _detect_holes_refit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m, "centroid")
    elif variant == "redness_argmax":
        det = _detect_holes_refit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m, "argmax")
    else:
        raise ValueError(variant)
    return iid, variant, [np.asarray(q, np.int32) for q in det]


def part_j(ids, gt_map, workers, variants):
    out = {}
    for v in variants:
        det_map = {}
        with Pool(workers) as pool:
            for k, (iid, _v, det) in enumerate(pool.imap_unordered(
                    _work_j, [(i, v) for i in ids], chunksize=4)):
                det_map[iid] = det
                if (k + 1) % 200 == 0:
                    print("  %s %d/%d" % (v, k + 1, len(ids)), flush=True)
        out[v] = score_variant(det_map, ids, gt_map)
        r = out[v]
        print("  %-18s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d"
              % (v, r["3"]["tp"], r["3"]["fp"], r["5"]["tp"], r["5"]["fp"],
                 r["8"]["tp"], r["8"]["fp"], r["12"]["tp"], r["12"]["fp"],
                 r["n_det"]), flush=True)
    return out


# ---------------------------------------------------- part I: combined gate
def _work_i(task):
    iid, = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, []
    m = prod_mask(img)
    wBGR = np.array([0.114, 0.587, 0.299])
    out = []
    for q in detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m):
        q = np.asarray(q, float)
        ring = ring_pixels(img, q, half=2)
        inner = interior_pixels(img, q)
        out.append((np.asarray(q, np.int32), _red_support(m, q),
                    gcnr(ring @ wBGR, inner @ wBGR)))
    return iid, out


def part_i(ids, gt_map, exp, workers):
    """Combined sup/gcnr gate, scored with the GT-CORRECTED metric."""
    from eval_pointmatch import is_real_unlabeled
    det_map = {}
    with Pool(workers) as pool:
        for k, (iid, det) in enumerate(pool.imap_unordered(
                _work_i, [(i,) for i in ids], chunksize=4)):
            det_map[iid] = det
            if (k + 1) % 200 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)

    masks = {}
    for iid in ids:
        if not det_map.get(iid):
            continue
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        masks[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                np.ones((3, 3), np.uint8), iterations=3) > 0

    def gate_of(kind, t):
        return lambda s, gr: (s >= t) if kind == "sup" else (
            (gr >= t) if kind == "gcnr" else (s >= 0.5 and gr >= t))

    configs = []
    for t in np.arange(0.5, 1.001, 0.05):
        configs.append(("sup", round(float(t), 2), 0.5))
    for t in np.arange(0.1, 0.701, 0.025):
        configs.append(("gcnr", 0.5, round(float(t), 3)))
        configs.append(("and", 0.5, round(float(t), 3)))
    configs.append(("none", 0.0, 0.0))

    rows = []
    for kind, ts, tg in configs:
        ntp = nfp = nunl = ndet = 0
        for iid in ids:
            det = det_map.get(iid, [])
            gt = gt_map.get(iid, [])
            mask_d = masks.get(iid)
            csv_n = exp.get(iid, 0)
            sel = []
            for q, s, gr in det:
                keep = (s >= ts) if kind == "sup" else (
                    (gr >= tg) if kind == "gcnr" else (s >= 0.5 and gr >= tg))
                if keep:
                    sel.append(q)
            ndet += len(sel)
            tp, unmatched = match_boxes_point(sel, gt, 5.0)
            ntp += tp
            for di in unmatched:
                if mask_d is not None and is_real_unlabeled(sel[di], csv_n,
                                                            len(gt), mask_d):
                    nunl += 1
                else:
                    nfp += 1
        ngt = sum(len(gt_map.get(i, [])) for i in ids)
        rec = ntp / max(ngt, 1) * 100
        prec = ntp / max(ntp + nfp, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"kind": kind, "sup_thr": ts, "gcnr_thr": tg,
                     "tp": ntp, "true_fp": nfp, "real_unl": nunl,
                     "n_det": ndet, "recall": round(rec, 1),
                     "precision": round(prec, 1), "corr_f1": round(f1, 1)})
    # print the pareto-relevant rows
    print("%-5s %7s %7s %6s %7s %7s %7s %6s %6s" %
          ("kind", "sup_t", "gcnr_t", "TP", "trueFP", "unl", "dets", "prec", "f1"))
    for r in rows:
        if r["tp"] >= 880:
            print("%-5s %7.2f %7.3f %6d %7d %7d %7d %6.1f %6.1f" %
                  (r["kind"], r["sup_thr"], r["gcnr_thr"], r["tp"], r["true_fp"],
                   r["real_unl"], r["n_det"], r["precision"], r["corr_f1"]))
    return {"rows": rows}


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="G")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

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
    print("%d images, %d GT boxes" % (len(ids), sum(len(v) for v in gt_map.values())),
          flush=True)

    part = args.part.upper()
    if part == "GM":
        res = part_g_mask(ids, gt_map, args.workers)
        names = list(next(iter(res.values())).keys())
        print("\n%-16s %10s %10s %10s %10s" %
              ("mask", "mask px", "gt_cover", "in_gt_band", "n_holes"))
        for n in names:
            agg = {k: float(np.mean([res[i][n][k] for i in res]))
                   for k in ("n_px", "gt_cover", "in_gt_band", "n_holes")}
            print("%-16s %10.0f %10.3f %10.3f %10.1f" %
                  (n, agg["n_px"], agg["gt_cover"], agg["in_gt_band"],
                   agg["n_holes"]))
    elif part == "G":
        variants = ["base", "ridge_r", "ridge_gray", "union_r",
                    "support_union_r", "support_ridge_r"]
        res = part_g(ids, gt_map, args.workers, variants)
    elif part == "GSUP":
        res = part_g(ids, gt_map, args.workers,
                     ["base", "support_union_r", "support_ridge_r"])
    elif part == "GMASK":
        res = part_g_mask(ids, gt_map, args.workers)
        names = list(next(iter(res.values())).keys())
        print("\n%-16s %10s %10s %10s %10s" %
              ("mask", "mask px", "gt_cover", "in_gt_band", "n_holes"))
        for n in names:
            agg = {k: float(np.mean([res[i][n][k] for i in res]))
                   for k in ("n_px", "gt_cover", "in_gt_band", "n_holes")}
            print("%-16s %10.0f %10.3f %10.3f %10.1f" %
                  (n, agg["n_px"], agg["gt_cover"], agg["in_gt_band"],
                   agg["n_holes"]))
    elif part == "J":
        variants = ["base", "redness_centroid", "redness_argmax"]
        res = part_j(ids, gt_map, args.workers, variants)
    elif part == "I":
        from common import load_expected
        res = part_i(ids, gt_map, load_expected(), args.workers)
    elif part == "H":
        res = part_h(ids, gt_map, args.workers)
        print(json.dumps({k: v for k, v in res.items()
                          if not k.startswith("curve")}, indent=1))
        for k in ("sup", "gcnr_ring_in", "gcnr_ring_out", "gcnr_in_out"):
            print("\ncurve %s" % k)
            for r in res["curve_" + k]:
                print("  %.2f  tp_kept=%4d  fp_kept=%4d" %
                      (r["thr"], r["tp_kept"], r["fp_kept"]))
    else:
        raise SystemExit("unknown part %s" % args.part)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("saved %s" % args.out)


if __name__ == "__main__":
    main()