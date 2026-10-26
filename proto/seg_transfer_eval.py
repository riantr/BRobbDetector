"""LOOP-43: does the ImageJ "Segmentation" algorithm collection transfer?

Source: D:\\src\\java\\algo\\Segmentation
  * Multi_OtsuThreshold.java / Multi_Thresholder.jar / Otsu_Thresholding.jar
        -> Otsu and multi-level Otsu (Liao 2001) thresholding
  * Mixture_Modeling.jar (GrayLevelClassMixtureModeling)
        -> 2-component Gaussian mixture thresholding
  * siox_.jar (SIOX_Segmentation / SioxSegmentator)
        -> flood fill from the image border with a colour tolerance
  * RATS_.jar (RATSQuadtree)
        -> region growing with hysteresis thresholds
  * Watershed_Algorithm.jar (WatershedFIFO)
        -> marker controlled watershed
  * Blob_Labeler.java
        -> 4/8-connected connected component labelling + vectorisation
  * Densitometry_1_Channel_.java / Densitometry_3_Channel_.java
        -> per-pixel colour distance to a reference colour (tolerance ball)
  * RetFM_J.java
        -> CLAHE + Otsu + EDM watershed + particle analysis

Each algorithm is mapped onto a concrete variant of the CURRENT pipeline
(red mask -> holes -> side fit) and measured on the 500-image real
benchmark, which is the only gate for a change.

  part A  mask quality of every mask alternative (no detector)
  part B  end-to-end: run the full detector with every mask alternative
  part C  densitometry as a *gate score* (graded redness instead of the
          binary red support): TP/FP separation, AUC + threshold curve
  part D  micro-tests: OpenCV hole connectivity (diagonal touch) and the
          effect of clearing the outer pixel ring (Blob_Labeler "edge")

Usage:
    python proto/seg_transfer_eval.py --part A
    python proto/seg_transfer_eval.py --part B --workers 14
    python proto/seg_transfer_eval.py --part C
    python proto/seg_transfer_eval.py --part D
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
from eval_pointmatch import match_boxes_point
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]

# tolerance of the Densitometry plugins (default intTol = 127)
DENS_TOL = 127.0
# flood fill tolerance of SIOX (40 leaks through the red stroke, 10 keeps it)
SIOX_TOL = 10


# --------------------------------------------------------------- GT helpers
def perimeter_pixels(quads, H, W, step=1):
    """Boolean image of the pixels that lie on the GT quad perimeter."""
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
    """Pixels within +-half of a GT line (a band, not the line itself)."""
    m = np.zeros((H, W), np.uint8)
    for q in quads:
        cv2.polylines(m, [np.asarray(q, np.int32)], True, 255, 1)
    k = np.ones((2 * half + 1, 2 * half + 1), np.uint8)
    return cv2.dilate(m, k) > 0


# ------------------------------------------------------------ mask builders
def ref_red(img, hsv_mask):
    """Median BGR colour of the reference (annotation) pixels."""
    bgr = img[hsv_mask > 0]
    if len(bgr) < 10:
        return np.array([0.0, 0.0, 255.0])
    return np.median(bgr.astype(np.float64), axis=0)


def tol_mask(img, ref, tol=DENS_TOL):
    """Densitometry 1-channel: |BGR - reference| <= tol."""
    d = np.linalg.norm(img.astype(np.float64) - ref[None, None, :], axis=2)
    return (d <= tol).astype(np.uint8) * 255


def otsu_mask(img):
    """Otsu_Thresholding.jar: single Otsu on the grey image."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def multi_otsu_mask(img, levels=3):
    """Multi_OtsuThreshold.java (Liao, Chen & Chung 2001).

    Maximises sum_c (S_c^2 / P_c) over levels-1 cut points.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    h /= max(h.sum(), 1.0)
    idx = np.arange(256, dtype=np.float64)
    P = np.cumsum(h)                       # P[i]  = sum h[0..i]
    S = np.cumsum(h * idx)                 # S[i]  = sum k*h[k], k<=i
    tot_p, tot_s = P[-1], S[-1]

    def cls(a, b):                        # class (a, b] -> (count, sum)
        cnt = P[b] - (P[a] if a >= 0 else 0.0)
        s = S[b] - (S[a] if a >= 0 else 0.0)
        return cnt, s

    best, best_t = -1.0, None
    if levels == 2:
        for t in range(0, 255):
            c1, s1 = cls(-1, t)
            c2, s2 = cls(t, 255)
            if c1 <= 0 or c2 <= 0:
                continue
            v = s1 * s1 / c1 + s2 * s2 / c2
            if v > best:
                best, best_t = v, (t,)
    else:
        for t1 in range(0, 254):
            c1, s1 = cls(-1, t1)
            if c1 <= 0:
                continue
            v1 = s1 * s1 / c1
            for t2 in range(t1 + 1, 255):
                c2, s2 = cls(t1, t2)
                c3, s3 = cls(t2, 255)
                if c2 <= 0 or c3 <= 0:
                    continue
                v = v1 + s2 * s2 / c2 + s3 * s3 / c3
                if v > best:
                    best, best_t = v, (t1, t2)
    m = np.zeros_like(g)
    if best_t:
        m[g > best_t[-1]] = 255
    return m


def mix_model_mask(img):
    """Mixture_Modeling.jar: 2-component 1-D GMM on the grey histogram."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    h /= max(h.sum(), 1.0)
    x = np.arange(256, dtype=np.float64)
    mu = np.array([60.0, 180.0])
    sd = np.array([40.0, 40.0])
    w = np.array([0.5, 0.5])
    for _ in range(60):
        p = w[:, None] * np.exp(-0.5 * ((x[None, :] - mu[:, None]) / sd[:, None]) ** 2) \
            / (sd[:, None] * np.sqrt(2 * np.pi))
        p = np.maximum(p, 1e-300)
        tot = p.sum(axis=0)
        p = p / tot[None, :]
        nk = p.sum(axis=1)
        w = nk / max(nk.sum(), 1e-12)
        mu = (p * x[None, :]).sum(axis=1) / np.maximum(nk, 1e-12)
        sd = np.sqrt(np.maximum((p * (x[None, :] - mu[:, None]) ** 2).sum(axis=1)
                                / np.maximum(nk, 1e-12), 1.0))
    # crossing point of the two components
    lo, hi = int(max(0, mu[0])), int(min(255, mu[1]))
    cross = hi
    for t in range(lo, hi + 1):
        d0 = w[0] * np.exp(-0.5 * ((t - mu[0]) / sd[0]) ** 2) / sd[0]
        d1 = w[1] * np.exp(-0.5 * ((t - mu[1]) / sd[1]) ** 2) / sd[1]
        if d0 >= d1:
            cross = t
            break
    m = np.zeros_like(g)
    m[g > cross] = 255
    return m


def siox_mask(img, ref, tol=SIOX_TOL):
    """siox_.jar: flood fill from the border with a colour tolerance.

    The unflooded region contains the annotation strokes AND the X-ray
    content they enclose; the strokes are then isolated with the
    Densitometry colour distance.

    NOTE: the flood tolerance is a knife edge on these images - at tol=40
    the fill leaks THROUGH the red stroke (the annotation disappears), at
    tol=10 the stroke survives but the whole box interior stays unflooded.
    """
    h, w = img.shape[:2]
    src = img.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    sentinel = (13, 251, 97)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(src, m, seed, sentinel, (tol,) * 3, (tol,) * 3, 4)
    unflooded = ~((src == np.array(sentinel)).all(axis=2))
    d = np.linalg.norm(img.astype(np.float64) - ref[None, None, :], axis=2)
    red = (d <= DENS_TOL).astype(np.uint8) * 255
    return cv2.bitwise_and(unflooded.astype(np.uint8) * 255, red)


def prod_mask(img):
    """The shipped mask: narrow HSV red rule + denoise."""
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    return m


def n_holes(mask):
    """Number of holes (child contours) = number of candidate boxes."""
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return int(np.sum(hier[0, :, 3] >= 0))


# ------------------------------------------------------------------ part A
_GT = {}


def _init_gt(gt_map):
    global _GT
    _GT = gt_map


def _work_a(iid):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, {}
    ref = ref_red(img, red_mask(img, P["sat_min"], P["val_min"], P["hue_max"]))
    masks = {
        "prod_hsv": prod_mask(img),
        "dens_tol": denoise(tol_mask(img, ref), P["denoise_min_area"]),
        "otsu": denoise(otsu_mask(img), P["denoise_min_area"]),
        "multi_otsu3": denoise(multi_otsu_mask(img, 3), P["denoise_min_area"]),
        "mix_model": denoise(mix_model_mask(img), P["denoise_min_area"]),
        "siox": denoise(siox_mask(img, ref), P["denoise_min_area"]),
    }
    return iid, _mask_stats(img, _GT.get(iid, []), masks)


def _mask_stats(img, quads, masks):
    H, W = img.shape[:2]
    per = perimeter_pixels(quads, H, W) > 0
    band = gt_band(quads, H, W)
    npix = int(per.sum())
    out = {}
    for name, m in masks.items():
        mm = m > 0
        n = int(mm.sum())
        cover = float((mm & per).sum()) / max(npix, 1)
        prec = float((mm & band).sum()) / max(n, 1)
        out[name] = {"n_px": n, "gt_cover": round(cover, 4),
                     "in_gt_band": round(prec, 4), "n_holes": n_holes(m)}
    return out


def part_a(ids, gt_map, workers):
    res = {}
    with Pool(workers, initializer=_init_gt, initargs=(gt_map,)) as pool:
        for k, (iid, st) in enumerate(pool.imap_unordered(_work_a, ids, chunksize=4)):
            res[iid] = st
            if (k + 1) % 100 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    return res


# ------------------------------------------------------------------ part B
def _detect_variant(task):
    iid, variant = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, variant, []
    ref = ref_red(img, red_mask(img, P["sat_min"], P["val_min"], P["hue_max"]))
    if variant == "prod_hsv":
        m = prod_mask(img)
    elif variant == "dens_tol":
        m = denoise(tol_mask(img, ref), P["denoise_min_area"])
    elif variant == "otsu":
        m = denoise(otsu_mask(img), P["denoise_min_area"])
    elif variant == "multi_otsu3":
        m = denoise(multi_otsu_mask(img, 3), P["denoise_min_area"])
    elif variant == "mix_model":
        m = denoise(mix_model_mask(img), P["denoise_min_area"])
    elif variant == "siox":
        m = denoise(siox_mask(img, ref), P["denoise_min_area"])
    else:
        raise ValueError(variant)
    det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    return iid, variant, [np.asarray(q, np.int32) for q in det]


def score_variant(det_map, ids, gt_map):
    tot = {D: {"tp": 0, "true_fp": 0, "unl": 0} for D in DIST_VALUES}
    ndet = 0
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        ndet += len(det)
        for D in DIST_VALUES:
            tp, unmatched = match_boxes_point(det, gt, float(D))
            tot[D]["tp"] += tp
            tot[D]["true_fp"] += len(unmatched)
    ngt = sum(len(gt_map.get(i, [])) for i in ids)
    res = {}
    for D, t in tot.items():
        rec = t["tp"] / max(ngt, 1) * 100
        prec = t["tp"] / max(t["tp"] + t["true_fp"], 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        res[str(D)] = {"tp": t["tp"], "true_fp": t["true_fp"],
                       "recall": round(rec, 1), "precision": round(prec, 1),
                       "f1": round(f1, 1)}
    res["n_det"] = ndet
    res["ngt"] = ngt
    return res


def part_b(ids, gt_map, workers, variants):
    out = {}
    for v in variants:
        det_map = {}
        with Pool(workers) as pool:
            for k, (iid, _v, det) in enumerate(pool.imap_unordered(
                    _detect_variant, [(i, v) for i in ids], chunksize=4)):
                det_map[iid] = det
                if (k + 1) % 200 == 0:
                    print("  %s %d/%d" % (v, k + 1, len(ids)), flush=True)
        out[v] = score_variant(det_map, ids, gt_map)
        r = out[v]
        print("  %-13s D=5 TP=%4d FP=%4d rec=%.1f prec=%.1f f1=%.1f  (dets=%d)"
              % (v, r["5"]["tp"], r["5"]["true_fp"], r["5"]["recall"],
                 r["5"]["precision"], r["5"]["f1"], r["n_det"]), flush=True)
    return out


# ------------------------------------------------------------------ part C
def _dens_scores(task):
    """Detection + (binary support, densitometry mean, densitometry min side)."""
    iid, = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, []
    hsv = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    ref = ref_red(img, hsv)
    m = prod_mask(img)
    d = np.linalg.norm(img.astype(np.float64) - ref[None, None, :], axis=2)
    red = d <= DENS_TOL
    H, W = img.shape[:2]
    out = []
    for q in detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m):
        q = np.asarray(q, float)
        sup = _red_support(m, q)
        per_vals = []
        side_means = []
        for i in range(4):
            a = q[i]
            b = q[(i + 1) % 4]
            L = float(np.linalg.norm(b - a))
            n = max(int(L / 2), 2)
            ts = np.linspace(0.0, 1.0, n, endpoint=False)
            pts = a[None, :] + ts[:, None] * (b - a)[None, :]
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            vals = red[ys, xs].astype(np.float64)
            per_vals.append(vals)
            side_means.append(float(vals.mean()))
        allv = np.concatenate(per_vals)
        out.append((np.asarray(q, np.int32), sup,
                    float(allv.mean()), float(min(side_means)),
                    float(np.percentile(allv, 25))))
    return iid, out


def auc(pos, neg):
    """AUC of pos (higher = TP) vs neg."""
    if not pos or not neg:
        return float("nan")
    a = np.asarray(pos + neg, float)
    r = np.argsort(np.argsort(a)) + 1.0
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def part_c(ids, gt_map, workers):
    det_map = {}
    with Pool(workers) as pool:
        for k, (iid, det) in enumerate(pool.imap_unordered(
                _dens_scores, [(i,) for i in ids], chunksize=4)):
            det_map[iid] = det
            if (k + 1) % 200 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)

    tp = {"sup": [], "dens": [], "dens_min": [], "dens_p25": []}
    fp = {k: [] for k in tp}
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        _n, unmatched = match_boxes_point([d[0] for d in det], gt, 5.0)
        uset = set(unmatched)
        for i, (q, sup, dm, dmin, dp25) in enumerate(det):
            tgt = tp if i not in uset else fp
            tgt["sup"].append(sup)
            tgt["dens"].append(dm)
            tgt["dens_min"].append(dmin)
            tgt["dens_p25"].append(dp25)
    out = {"n_tp": len(tp["sup"]), "n_fp": len(fp["sup"])}
    for k in tp:
        out["auc_" + k] = round(auc(tp[k], fp[k]), 4)
        out["tp_p50_" + k] = round(float(np.percentile(tp[k], 50)), 4)
        out["fp_p50_" + k] = round(float(np.percentile(fp[k], 50)), 4)
    # threshold curve for the densitometry score (higher = more red)
    for k in ("sup", "dens_min"):
        rows = []
        for t in np.arange(0.0, 1.001, 0.05):
            rows.append({"thr": round(float(t), 2),
                         "tp_kept": int(sum(1 for v in tp[k] if v >= t)),
                         "fp_kept": int(sum(1 for v in fp[k] if v >= t))})
        out["curve_" + k] = rows
    return out


# ------------------------------------------------------------------ part D
def part_d():
    out = {}
    # D1: does a diagonal touch create a hole? (Blob_Labeler 4c vs 8c)
    m = np.zeros((60, 60), np.uint8)
    cv2.rectangle(m, (10, 10), (49, 49), 255, 2)
    cv2.line(m, (0, 0), (10, 10), 255, 2)          # diagonal touch at the corner
    out["diagonal_touch_holes"] = n_holes(m)
    m2 = np.zeros((60, 60), np.uint8)
    cv2.rectangle(m2, (10, 10), (49, 49), 255, 2)
    out["clean_rect_holes"] = n_holes(m2)
    # D2: Blob_Labeler "edge" = clear the outer pixel ring
    m3 = np.zeros((60, 60), np.uint8)
    cv2.rectangle(m3, (0, 0), (59, 59), 255, 2)    # box clipped by the image edge
    out["edge_box_holes"] = n_holes(m3)
    m4 = m3.copy()
    m4[0, :] = 0
    m4[-1, :] = 0
    m4[:, 0] = 0
    m4[:, -1] = 0
    out["edge_box_holes_ring_cleared"] = n_holes(m4)
    return out


# ------------------------------------------------------------------ part E
def siox_unflooded(img, tol=SIOX_TOL):
    """siox_.jar: the region that the border flood fill cannot reach."""
    h, w = img.shape[:2]
    src = img.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    sentinel = np.array([13, 251, 97])
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(src, m, seed, tuple(int(v) for v in sentinel),
                      (tol,) * 3, (tol,) * 3, 4)
    return (~((src == sentinel).all(axis=2))).astype(np.uint8) * 255


def _work_e(task):
    """SIOX as a QUAD source: minAreaRect of the unflooded region."""
    iid, variant = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, variant, []
    from scope45_holes import _gate, _enforce, _on_edge
    H, W = img.shape[:2]
    m = prod_mask(img)
    u = siox_unflooded(img)
    cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in cs:
        if cv2.contourArea(c) < 300:
            continue
        q = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
        g = _gate(q, W, H, P)
        if g is None:
            continue
        g = _enforce(g, PIPE_DEFAULT["ortho"])
        if _on_edge(g, W, H, P):
            continue
        if _red_support(m, g) < HOLE_DEFAULT["min_red_support"]:
            continue
        quads.append(np.asarray(g, np.int32))
    if variant == "siox_quad":
        return iid, variant, quads
    # union with the shipped detections
    base = [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)]
    from cv_hybrid import dedup_keep_order
    return iid, variant, dedup_keep_order(base + quads, PIPE_DEFAULT["dedup_iou"])


def part_e(ids, gt_map, workers, variants=("siox_quad", "union")):
    out = {}
    for v in variants:
        det_map = {}
        with Pool(workers) as pool:
            for k, (iid, _v, det) in enumerate(pool.imap_unordered(
                    _work_e, [(i, v) for i in ids], chunksize=4)):
                det_map[iid] = det
                if (k + 1) % 200 == 0:
                    print("  %s %d/%d" % (v, k + 1, len(ids)), flush=True)
        out[v] = score_variant(det_map, ids, gt_map)
        r = out[v]
        print("  %-10s D=3 TP=%4d FP=%4d | D=5 TP=%4d FP=%4d | D=8 TP=%4d FP=%4d "
              "| D=12 TP=%4d FP=%4d (dets=%d)"
              % (v, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"], r["5"]["true_fp"],
                 r["8"]["tp"], r["8"]["true_fp"], r["12"]["tp"], r["12"]["true_fp"],
                 r["n_det"]), flush=True)
    return out


# ------------------------------------------------------------------ part F
def split_diagonals(mask):
    """Emulate 4-connected labelling on an 8-connected binary mask.

    Blob_Labeler.java offers a 4-connected mode; OpenCV's findContours is
    always 8-connected for the foreground.  A pixel pair that touches only
    diagonally (both orthogonal neighbours are background) is a 2x2
    checkerboard corner - the 8-connected labeling merges the two blobs, the
    4-connected one keeps them apart.  Breaking those links is the only way
    to emulate 4-connectivity with OpenCV.

    This matters for the "leader arrow pulls a box corner" failure mode: an
    arrow that only touches the box diagonally is merged into the box
    component by the 8-connected labeling and drags the corner fit.
    """
    m = mask > 0
    a = m[:-1, :-1]      # top-left
    b = m[1:, 1:]        # bottom-right
    c = m[1:, :-1]       # bottom-left
    d = m[:-1, 1:]       # top-right
    kill_b = a & b & ~c & ~d       # diagonal (a,b) only -> drop b
    kill_d = c & d & ~a & ~b       # anti-diagonal (c,d) only -> drop d
    out = m.copy()
    out[1:, 1:] &= ~kill_b
    out[:-1, 1:] &= ~kill_d
    return out.astype(np.uint8) * 255


def clear_edge_ring(mask):
    """Blob_Labeler 'edge correction': background border on x=0 and y=0."""
    out = mask.copy()
    out[0, :] = 0
    out[:, 0] = 0
    return out


def _work_f(task):
    iid, variant = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, variant, [], {}
    base = prod_mask(img)
    if variant == "base":
        m = base
    elif variant == "4conn":
        m = split_diagonals(base)
    elif variant == "edgecorr":
        m = clear_edge_ring(base)
    elif variant == "4conn_edgecorr":
        m = clear_edge_ring(split_diagonals(base))
    else:
        raise ValueError(variant)
    info = {
        "px_base": int((base > 0).sum()),
        "px_new": int((m > 0).sum()),
        "comp_base": int(cv2.connectedComponents((base > 0).astype(np.uint8),
                                                  connectivity=8)[0] - 1),
        "comp_new": int(cv2.connectedComponents((m > 0).astype(np.uint8),
                                                 connectivity=8)[0] - 1),
    }
    det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    return iid, variant, [np.asarray(q, np.int32) for q in det], info


def part_f(ids, gt_map, workers,
           variants=("base", "4conn", "edgecorr", "4conn_edgecorr")):
    out = {}
    for v in variants:
        det_map = {}
        infos = []
        with Pool(workers) as pool:
            for k, (iid, _v, det, info) in enumerate(pool.imap_unordered(
                    _work_f, [(i, v) for i in ids], chunksize=4)):
                det_map[iid] = det
                infos.append(info)
                if (k + 1) % 200 == 0:
                    print("  %s %d/%d" % (v, k + 1, len(ids)), flush=True)
        out[v] = score_variant(det_map, ids, gt_map)
        out[v]["mask_stats"] = {
            "px_removed_mean": round(float(np.mean(
                [i["px_base"] - i["px_new"] for i in infos])), 1),
            "components_before_mean": round(float(np.mean(
                [i["comp_base"] for i in infos])), 2),
            "components_after_mean": round(float(np.mean(
                [i["comp_new"] for i in infos])), 2),
        }
        r = out[v]
        print("  %-16s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d | px-removed=%5.1f comps %.1f->%.1f"
              % (v, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"], r["5"]["true_fp"],
                 r["8"]["tp"], r["8"]["true_fp"], r["12"]["tp"], r["12"]["true_fp"],
                 r["n_det"], r["mask_stats"]["px_removed_mean"],
                 r["mask_stats"]["components_before_mean"],
                 r["mask_stats"]["components_after_mean"]), flush=True)
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="A")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.part.upper() == "D":
        print(json.dumps(part_d(), indent=1))
        return

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
    if part == "A":
        res = part_a(ids, gt_map, args.workers)
        names = list(next(iter(res.values())).keys())
        print("\n%-13s %10s %10s %10s %10s" %
              ("mask", "mask px", "gt_cover", "in_gt_band", "n_holes"))
        for n in names:
            agg = {k: float(np.mean([res[i][n][k] for i in res]))
                   for k in ("n_px", "gt_cover", "in_gt_band", "n_holes")}
            print("%-13s %10.0f %10.3f %10.3f %10.1f" %
                  (n, agg["n_px"], agg["gt_cover"], agg["in_gt_band"],
                   agg["n_holes"]))
    elif part == "B":
        variants = ["prod_hsv", "dens_tol", "otsu", "multi_otsu3",
                    "mix_model", "siox"]
        res = part_b(ids, gt_map, args.workers, variants)
    elif part == "C":
        res = part_c(ids, gt_map, args.workers)
        print(json.dumps({k: v for k, v in res.items()
                          if not k.startswith("curve")}, indent=1))
        for k in ("sup", "dens_min"):
            print("\ncurve %s" % k)
            for r in res["curve_" + k]:
                print("  %.2f  tp_kept=%4d  fp_kept=%4d" %
                      (r["thr"], r["tp_kept"], r["fp_kept"]))
    elif part == "E":
        res = part_e(ids, gt_map, args.workers)
    elif part == "F":
        res = part_f(ids, gt_map, args.workers)
    else:
        raise SystemExit("unknown part %s" % args.part)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("saved %s" % args.out)


if __name__ == "__main__":
    main()