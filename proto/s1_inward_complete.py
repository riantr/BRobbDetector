"""S1: INWARD cut-side completion (the mirror of the measured-dead outward move).

Motivation (scope58, benchmark/scope58_cutside.json): for a covering hole vs the
GT box, rel_bigger 114 / rel_same 53 / rel_smaller 30.  The outward completion
(P4) only addresses the "smaller" case (~30) and was measured HARMFUL
(D=5 886, gains 0 TP / losses 15).  The MAJORITY case is "bigger": the interior
merged through a gap, so the box's true side lies INSIDE the fragment.  This
slides a side flagged "cut" (red continues >= cut_len px past BOTH corners)
TOWARD the quad centre onto the next strong parallel stroke, then re-intersects
the 4 lines.

Mandatory per the goal: a TP-set gain/loss decomposition (a net-positive delta is
not sufficient).

Variants (each is one 500-image sweep; the goal caps an item at 3 sweeps):
  base    the shipped pm_v10 path (must reproduce the frozen table exactly)
  in060   inward, first shift with coverage >= 0.60
  in075   inward, stricter acceptance (coverage >= 0.75)

Usage: python proto/s1_inward_complete.py --workers 12
"""
import argparse
import itertools
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
                           _gate, _enforce, _on_edge, _red_support, _line_isect)

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]
PERMS = list(itertools.permutations(range(4)))

VARIANTS = {"in060": 0.60, "in075": 0.75}


def line_cov(mask, a, b, n, offset, band=1):
    """Red coverage of segment (a,b) translated by offset*n."""
    H, W = mask.shape
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 3:
        return 0.0
    ts = np.linspace(0.0, 1.0, max(int(L), 8))
    base = a[None, :] + ts[:, None] * d[None, :] + offset * n[None, :]
    hit = np.zeros(len(ts), bool)
    for s in np.arange(-band, band + 0.5, 1.0):
        pts = base + s * n[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        hit |= mask[ys, xs] > 0
    return float(hit.mean())


def cont_run(mask, a, u, max_len=25):
    """Consecutive red px continuing from `a` in direction `u`."""
    H, W = mask.shape
    cnt = 0
    for s in range(1, max_len + 1):
        p = a + s * u
        x0 = max(int(round(p[0])) - 1, 0)
        x1 = min(int(round(p[0])) + 1, W - 1)
        y0 = max(int(round(p[1])) - 1, 0)
        y1 = min(int(round(p[1])) + 1, H - 1)
        if mask[y0:y1 + 1, x0:x1 + 1].any():
            cnt += 1
        else:
            break
    return cnt


def inward_complete(mask, q, min_cov, max_shift=24, cut_len=5, stats=None):
    """Slide the CUT sides of q INWARD onto the next strong parallel stroke."""
    q = np.asarray(q, float)
    c = q.mean(axis=0)
    sides = []
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 6:
            return q, False
        u = d / L
        n = np.array([-u[1], u[0]])
        if np.dot(n, (a + b) / 2.0 - c) < 0:
            n = -n                                   # outward normal
        cov0 = line_cov(mask, a, b, n, 0.0)
        cont = min(cont_run(mask, a, -u), cont_run(mask, b, u))
        sides.append((a, b, u, n, cov0, cont))

    lines = []
    changed = False
    for (a, b, u, n, cov0, cont) in sides:
        newline = (a, u)
        if cont >= cut_len:
            if stats is not None:
                stats['cut_sides'] = stats.get('cut_sides', 0) + 1
            for s in range(2, max_shift + 1):
                cov = line_cov(mask, a, b, n, -float(s))     # INWARD
                if cov >= min_cov:
                    newline = (a - s * n, u)
                    changed = True
                    if stats is not None:
                        stats['replaced'] = stats.get('replaced', 0) + 1
                    break
        lines.append(newline)

    out = []
    for i in range(4):
        x = _line_isect(lines[i - 1][0], lines[i - 1][1],
                        lines[i][0], lines[i][1])
        if x is None or not np.isfinite(x).all():
            return q, False
        out.append(x)
    r = np.array(out, float)
    if not np.isfinite(r).all():
        return q, False
    if changed and stats is not None:
        stats['quads_changed'] = stats.get('quads_changed', 0) + 1
    return r, changed


def detect_holes_s1(img, p, cfg, pipe, mask, min_cov, stats):
    H, W = img.shape[:2]
    out = []
    for h in _holes_of(mask, cfg):
        stats['quads'] = stats.get('quads', 0) + 1
        q = _quad_of(h, mask, cfg)
        if q is None:
            continue
        if cfg.get("hole_refine", 0):
            r = _refine_sides(mask, q, band=cfg.get("hole_refine_band", 4.0),
                              min_pts=int(cfg.get("hole_refine_pts", 8)),
                              tol=cfg.get("hole_refine_tol", 2.0))
            if r is not None:
                q = r
        r, ch = inward_complete(mask, q, min_cov, stats=stats)
        if ch:
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
        return iid, mode, [], {}
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    stats = {}
    raw = []
    if mode == "base":
        raw += detect_holes(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
    else:
        raw += detect_holes_s1(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m,
                               VARIANTS[mode], stats)
    raw += detect_edgefit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
    det = dedup_keep_order(raw, PIPE_DEFAULT.get("dedup_iou", P["dedup_iou"]))
    return iid, mode, [np.asarray(q, np.int32) for q in det], stats


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
        res[str(D)] = {"tp": tp, "true_fp": fp, "real_unl": unl, "n_det": ndet,
                       "recall": round(rec, 1), "precision": round(prec, 1),
                       "corr_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 1)}
    return res


def tp_boxset(det_map, gt_map, ids, D=5.0):
    """The set of (image, GT index) pairs matched at D - for gain/loss."""
    out = set()
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        for gi, g in enumerate(gt):
            gq = np.asarray(g, float)
            best = None
            for i in range(len(det)):
                a = np.asarray(det[i], float)[:, None, :]
                dm = np.linalg.norm(a - gq[None, :, :], axis=2)
                worst = min(max(dm[p[0], 0], dm[p[1], 1], dm[p[2], 2], dm[p[3], 3])
                            for p in PERMS)
                if worst <= D and (best is None or worst < best):
                    best = worst
            if best is not None:
                out.add((iid, gi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/s1_inward.json")
    ap.add_argument("--modes", default="base,in060,in075")
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [g for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()
    ngt = sum(len(v) for v in gt_map.values())
    print("%d images, %d GT boxes" % (len(ids), ngt), flush=True)

    modes = args.modes.split(",")
    out = {}
    dets = {}
    for mode in modes:
        det_map = {}
        agg = {}
        with Pool(args.workers) as pool:
            for k, (iid, _m, det, st) in enumerate(pool.imap_unordered(
                    _work, [(i, mode) for i in ids], chunksize=3)):
                det_map[iid] = det
                for kk, vv in st.items():
                    agg[kk] = agg.get(kk, 0) + vv
        dets[mode] = det_map
        r = score(det_map, ids, gt_map, exp)
        out[mode] = {"metrics": r, "mechanism": agg}
        print("  %-6s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d corr_f1@5=%.1f  %s"
              % (mode, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"],
                 r["5"]["true_fp"], r["8"]["tp"], r["8"]["true_fp"],
                 r["12"]["tp"], r["12"]["true_fp"], r["5"]["n_det"],
                 r["5"]["corr_f1"], agg), flush=True)

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

    if "base" in dets:
        sb = tp_boxset(dets["base"], gt_map, ids)
        print("\n== TP-set decomposition at D=5 (mandatory for a repair move) ==")
        for m in modes:
            if m == "base":
                continue
            sm = tp_boxset(dets[m], gt_map, ids)
            gains = len(sm - sb)
            losses = len(sb - sm)
            out[m]["tp_set"] = {"gains": gains, "losses": losses,
                                "base_tp": len(sb), "cand_tp": len(sm)}
            print("  %-6s base=%d cand=%d  GAINS=%d  LOSSES=%d  -> %s"
                  % (m, len(sb), len(sm), gains, losses,
                     "gains>losses" if gains > losses else "REJECT (gains <= losses)"))

    json.dump({"results": out, "ngt": ngt},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()