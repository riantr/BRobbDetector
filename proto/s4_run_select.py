"""S4: robust in-band pixel SELECTION (nearest-run core), not re-weighting.

P2's null result is a diagnosis, not a dead end: the fit-then-project is already
the least-squares optimum ON THE EVIDENCE IT IS GIVEN, so the residual corner
error lives in WHICH pixels the band contains (a neighbouring box's side, an
arrow spur).  The shipped `_refine_sides` fits ALL red pixels in a symmetric band
and then *rejects the whole side* when the p90 residual exceeds the tolerance -
so a contaminated band silently falls back to the unrefined (worse) side.

S4 instead, for every cross-section of the side band:
  1. find the red offset nearest the side line (offset 0),
  2. keep only a +-1 px core around it,
  3. fit the TLS line on those points only.
A second stroke 2+ px away is excluded instead of pulling the fit, and the
residual guard becomes meaningful instead of triggering a rejection.

Selection, not weighting -> distinct from LOOP-44 (redness-weighted centroid)
and from P2 (joint LS over all band pixels).

Usage: python proto/s4_run_select.py --workers 12
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
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect, detect_edgefit,
                           _holes_of, _quad_of, _gate, _enforce, _on_edge,
                           _red_support, _line_isect)
from s1_inward_complete import score, tp_boxset

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
VARIANTS = {"runs3": 3.0, "runs5": 5.0}


def refine_runs(mask, q, band, min_pts=8, tol=2.0, core=1):
    """Re-fit the 4 sides on the red-run CORE nearest each side line."""
    H, W = mask.shape
    offs = np.arange(-band, band + 0.5, 1.0)
    n_off = len(offs)
    zc = int(np.argmin(np.abs(offs)))
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
        prof = mask[ys, xs] > 0                    # (nt, n_off)
        z = np.where(prof, np.abs(offs)[None, :], np.inf)
        j0 = z.argmin(axis=1)
        has = np.isfinite(z.min(axis=1))
        if int(has.sum()) < min_pts:
            return None
        rows_all, off_all = [], []
        r_idx = np.arange(prof.shape[0])
        for dj in range(-core, core + 1):
            jj = np.clip(j0 + dj, 0, n_off - 1)
            ok = has & prof[r_idx, jj]
            if ok.any():
                rows_all.append(r_idx[ok])
                off_all.append(offs[jj[ok]])
        if not rows_all:
            return None
        rows = np.concatenate(rows_all)
        oo = np.concatenate(off_all)
        if len(rows) < min_pts:
            return None
        Pt = base[rows] + oo[:, None] * n[None, :]
        mu = Pt.mean(axis=0)
        _, _, vt = np.linalg.svd(Pt - mu, full_matrices=False)
        dd = vt[0]
        if float(np.dot(dd, u)) < 0:
            dd = -dd
        resid = np.abs((Pt - mu) @ n)
        if float(np.percentile(resid, 90)) > tol:
            return None
        lines.append((mu, dd))
    out = []
    for i in range(4):
        x = _line_isect(*lines[i - 1], *lines[i])
        if x is None or not np.isfinite(x).all():
            return None
        out.append(x)
    r = np.array(out, float)
    return r if np.isfinite(r).all() else None


def detect_holes_runs(img, p, cfg, pipe, mask, band):
    H, W = img.shape[:2]
    out = []
    for h in _holes_of(mask, cfg):
        q = _quad_of(h, mask, cfg)
        if q is None:
            continue
        if cfg.get("hole_refine", 0):
            # S4 replaces the shipped refine, keeping its band/tol/pts semantics
            r = refine_runs(mask, q, band,
                            min_pts=int(cfg.get("hole_refine_pts", 8)),
                            tol=cfg.get("hole_refine_tol", 2.0))
            if r is not None:
                q = r
        q = _gate(q, W, H, p)
        if q is None:
            continue
        q = _enforce(q, pipe["ortho"])
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
    if mode == "base":
        raw = detect_holes_shipped(img, m)
    else:
        raw = detect_holes_runs(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m, VARIANTS[mode])
    raw += detect_edgefit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
    det = dedup_keep_order(raw, PIPE_DEFAULT.get("dedup_iou", P["dedup_iou"]))
    return iid, mode, [np.asarray(q, np.int32) for q in det]


def detect_holes_shipped(img, mask):
    from scope45_holes import detect_holes
    return detect_holes(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/s4_run_select.json")
    ap.add_argument("--modes", default="base,runs3,runs5")
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
        print("  %-6s base=%d cand=%d  GAINS=%d  LOSSES=%d"
              % (m, len(sb), len(sm), len(sm - sb), len(sb - sm)))
        out[m]["tp_set"] = {"gains": len(sm - sb), "losses": len(sb - sm)}

    json.dump({"results": out}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()