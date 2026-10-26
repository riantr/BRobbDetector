"""P4: partial-hole outward side completion.

Diagnosis this attacks (benchmark/scope58_cutside.json, the one diagnostic in
the LOOP-47..59 series with NO follow-up variant): a box whose interior is
crossed by a neighbouring box's side fragments into small holes.  The covering
hole quad is then a SUB-rectangle of the box, i.e. the box's true sides lie
OUTSIDE the fragment.

The signal (scope58): a TRUE box side stops at its corner, while a CUT side
(the crossing stroke of a neighbour) keeps going PAST the corner.  Measured:
129 FN have a detectable cut side (cut_yes 129 / cut_no 68).

This builds the repair scope58 proposed but nobody implemented: flag the cut
sides by red continuation beyond the corners, then slide each cut side
outward (away from the quad centre) until it lands on the next strong parallel
red stroke, and re-intersect the 4 lines.

  p4_cut     replace only sides flagged as cut
  p4_auto    replace a side when the outward search finds a much better line

Scored on the real 500 images with the GT-corrected metric.

Usage: python proto/p4_outward_complete.py --workers 12 --out benchmark/p4_outward.json
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
                           _gate, _enforce, _on_edge, _red_support, _line_isect)

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]

STATS = {"quads": 0, "cut_sides": 0, "replaced": 0, "quads_changed": 0}


def line_at(mask, a, b, n, offset, band=1, nsamp=None):
    """Red coverage of the segment (a,b) translated by `offset*normal(n)`."""
    H, W = mask.shape
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 3:
        return 0.0
    if nsamp is None:
        nsamp = max(int(L), 8)
    ts = np.linspace(0.0, 1.0, nsamp)
    base = a[None, :] + ts[:, None] * d[None, :] + offset * n[None, :]
    hit = np.zeros(len(ts), bool)
    for s in np.arange(-band, band + 0.5, 1.0):
        pts = base + s * n[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        hit |= mask[ys, xs] > 0
    return float(hit.mean())


def continuation(mask, a, u, max_len=25, band=1):
    """How many consecutive red px continue from `a` in direction `u`."""
    H, W = mask.shape
    cnt = 0
    for s in range(1, max_len + 1):
        p = a + s * u
        xs = np.clip(np.round(np.array([p[0] - band, p[0], p[0] + band])).astype(int), 0, W - 1)
        ys = np.clip(np.round(np.array([p[1], p[1], p[1]])).astype(int), 0, H - 1)
        # a small 3x3 neighbourhood -> tolerant of stroke thickness
        x0, x1 = max(int(round(p[0])) - 1, 0), min(int(round(p[0])) + 1, W - 1)
        y0, y1 = max(int(round(p[1])) - 1, 0), min(int(round(p[1])) + 1, H - 1)
        if mask[y0:y1 + 1, x0:x1 + 1].any():
            cnt += 1
        else:
            break
    return cnt


def outward_complete(mask, q, max_shift=24, min_cov=0.60, cut_len=5, mode="cut"):
    """Slide the CUT sides of `q` outward onto the next parallel stroke."""
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
            n = -n                                     # outward
        cov0 = line_at(mask, a, b, n, 0.0, band=1)
        cont = min(continuation(mask, a, -u), continuation(mask, b, u))
        sides.append((a, b, u, n, L, cov0, cont))

    lines = []
    changed = False
    for (a, b, u, n, L, cov0, cont) in sides:
        newline = (a, u)
        if cont >= cut_len:
            STATS["cut_sides"] += 1
            best = None
            for s in range(2, max_shift + 1):
                cov = line_at(mask, a, b, n, float(s), band=1)
                if cov >= min_cov:
                    best = (s, cov)
                    break
            if best is not None:
                s, cov = best
                if mode == "force" or cov > cov0 + 0.15:
                    newline = (a + s * n, u)
                    changed = True
                    STATS["replaced"] += 1
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
    if changed:
        STATS["quads_changed"] += 1
    return r, changed


def detect_holes_p4(img, p, cfg, pipe, mask, mode):
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
        STATS["quads"] += 1
        r, changed = outward_complete(mask, q, mode=mode)
        if changed:
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
    raw = []
    if mode == "base":
        raw += detect_holes(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m)
    else:
        raw += detect_holes_p4(img, P, HOLE_DEFAULT, PIPE_DEFAULT, m, mode)
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
    ap.add_argument("--out", default="benchmark/p4_outward.json")
    ap.add_argument("--modes", default="base,cut")
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
        STATS.update({k: 0 for k in STATS})
        det_map = {}
        with Pool(args.workers) as pool:
            for k, (iid, _m, det) in enumerate(pool.imap_unordered(
                    _work, [(i, mode) for i in ids], chunksize=3)):
                det_map[iid] = det
        r = score(det_map, ids, gt_map, exp)
        out[mode] = r
        print("  %-6s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d  corr_f1@5=%.1f"
              % (mode, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"],
                 r["5"]["true_fp"], r["8"]["tp"], r["8"]["true_fp"],
                 r["12"]["tp"], r["12"]["true_fp"], r["5"]["n_det"],
                 r["5"]["corr_f1"]), flush=True)
        if mode != "base":
            print("       mechanism: quads=%d cut_sides=%d replaced=%d "
                  "quads_changed=%d"
                  % (STATS["quads"], STATS["cut_sides"], STATS["replaced"],
                     STATS["quads_changed"]), flush=True)

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