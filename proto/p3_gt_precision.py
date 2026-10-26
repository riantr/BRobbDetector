"""P3: how precise is the hand-drawn GT itself?

The detector's quad is forced to be strictly orthogonal (ortho_edge).  So the
achievable D=3 rate is bounded by the GT: if the hand-drawn GT rectangle is
itself skewed by more than 3 px, NO strictly-orthogonal detector output can put
all 4 corners within 3 px of it.

For every rule-consistent GT box this measures
  * ortho_dev      max |interior angle - 90| in degrees
  * best_rect_px   the minimax residual of the BEST strictly-orthogonal
                   rectangle fitted to the 4 GT corners (px) - i.e. the hard
                   floor for this GT box at any D
  * coord_off      offset of the GT side line from the actual red ink centre
                   (cross-section centroid on the redness channel), px

Reports the D-ceiling table: the fraction of GT boxes that are reachable at
D=1..12 by a perfect orthogonal detector, vs the detector's actual rates
(D=3 857/1076 = 79.6%, D=5 901/1076 = 83.7%).

Usage: python proto/p3_gt_precision.py [--out benchmark/p3_gt_precision.json]
"""
import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
PERMS = list(itertools.permutations(range(4)))
PERM_IDX = np.array(PERMS)                 # (24, 4)
DIST_VALUES = [1, 2, 3, 4, 5, 8, 12]


def order_cyclic(q):
    """Order the 4 corners cyclically around the centroid (convex quad)."""
    q = np.asarray(q, float)
    c = q.mean(axis=0)
    a = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    return q[np.argsort(a)]


def interior_angles(q):
    ang = []
    for i in range(4):
        p0 = q[(i - 1) % 4]
        p1 = q[i]
        p2 = q[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cosv = float(np.dot(v1, v2) / (n1 * n2))
        ang.append(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
    return np.array(ang)


def best_ortho_rect(q, step_deg=0.25, span_deg=20.0):
    """Minimax residual of the best strictly-orthogonal rectangle to q.

    For a fixed orientation theta the optimal orthogonal rectangle whose sides
    are parallel to theta is the axis-aligned bounding box of the corners in
    the rotated frame.  The residual is min over the 24 corner-correspondences
    of the max per-corner distance.  theta is swept.
    """
    q = np.asarray(q, float)
    d = q[2] - q[1]                                  # a side direction
    th0 = float(np.arctan2(d[1], d[0]))
    best = np.inf
    best_th = th0
    steps = int(2 * span_deg / step_deg)
    for k in range(steps + 1):
        th = th0 + np.radians(-span_deg + k * step_deg)
        u = np.array([np.cos(th), np.sin(th)])
        v = np.array([-np.sin(th), np.cos(th)])
        pu = q @ u
        pv = q @ v
        lo_u, hi_u = pu.min(), pu.max()
        lo_v, hi_v = pv.min(), pv.max()
        rc = np.array([[lo_u, lo_v], [hi_u, lo_v], [hi_u, hi_v], [lo_u, hi_v]])
        rect = np.outer(rc[:, 0], u) + np.outer(rc[:, 1], v)   # (4,2)
        # distance matrix GT corner i -> rect corner j
        dm = np.linalg.norm(q[:, None, :] - rect[None, :, :], axis=2)   # (4,4)
        # min over the 24 correspondences of the max distance
        vals = dm[PERM_IDX[:, 0], 0], dm[PERM_IDX[:, 1], 1], \
            dm[PERM_IDX[:, 2], 2], dm[PERM_IDX[:, 3], 3]
        mx = np.maximum(np.maximum(vals[0], vals[1]), np.maximum(vals[2], vals[3]))
        m = float(mx.min())
        if m < best:
            best = m
            best_th = th
    return best, np.degrees(best_th - th0)


def redness_map(img):
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    return r - np.maximum(g, b)


def gt_side_offset(score, q, band=4.0):
    """Signed offset of the GT side line from the red-ink centre, per side."""
    H, W = score.shape
    offs = np.arange(-band, band + 0.5, 1.0)
    out = []
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 6:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        ts = np.linspace(0.15, 0.85, max(int(L * 0.5), 6))
        base = a[None, :] + ts[:, None] * d[None, :]
        pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
        xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
        prof = score[ys, xs]
        w = np.maximum(prof, 0)
        sw = w.sum(axis=1)
        ok = sw > 1.0
        if not ok.any():
            continue
        c = (offs[None, :] * w).sum(axis=1)[ok] / sw[ok]
        out.extend(c.tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/p3_gt_precision.json")
    args = ap.parse_args()

    ids = get_ids()
    rows = []
    all_offs = []
    for n, iid in enumerate(ids, 1):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        quads = [g for g in load_boxes_any(RAW, iid)
                 if not quad_on_boundary(g, W, H, 0.0)]
        if not quads:
            continue
        score = redness_map(img)
        for gi, g in enumerate(quads):
            q = order_cyclic(g)
            ang = interior_angles(q)
            if ang is None:
                continue
            dev = float(np.max(np.abs(ang - 90.0)))
            rectres, dth = best_ortho_rect(q)
            sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
            off = gt_side_offset(score, q)
            all_offs.extend(off)
            rows.append({
                "id": iid, "gi": gi,
                "ortho_dev": round(dev, 3),
                "best_rect_px": round(rectres, 3),
                "best_rect_dth": round(dth, 2),
                "min_side": round(min(sides), 1),
                "max_side": round(max(sides), 1),
                "off_n": len(off),
                "off_bias": round(sum(off) / len(off), 3) if off else None,
                "off_abs_p50": (round(sorted(abs(x) for x in off)[len(off) // 2], 3)
                                if off else None),
            })
        if n % 100 == 0:
            print("  %d/%d" % (n, len(ids)), flush=True)

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    dev = [r["ortho_dev"] for r in rows]
    res = [r["best_rect_px"] for r in rows]
    offs = all_offs
    n = len(rows)
    print("\nimages=%d  GT boxes measured=%d" % (len(ids), n))
    print("\northogonality deviation (deg):")
    print("  p50=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f max=%.2f"
          % (pct(dev, .50), pct(dev, .75), pct(dev, .90), pct(dev, .95),
             pct(dev, .99), max(dev)))
    for t in (1, 2, 3, 5, 10, 15, 20):
        k = sum(1 for x in dev if x > t)
        print("  >%2d deg: %4d  (%5.1f%%)" % (t, k, 100.0 * k / n))

    print("\nBEST strictly-orthogonal rectangle fitted to the GT corners,"
          " minimax corner residual (px):")
    print("  p50=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f max=%.2f"
          % (pct(res, .50), pct(res, .75), pct(res, .90), pct(res, .95),
             pct(res, .99), max(res)))
    print("\n  == D-ceiling: fraction of GT reachable by a PERFECT orthogonal detector ==")
    actual = {3: 857 / 1076, 5: 901 / 1076}
    print("  %-6s %-14s %-14s %s" % ("D", "GT-reachable", "detector", "gap"))
    for D in DIST_VALUES:
        k = sum(1 for x in res if x <= D + 0.0)
        frac = 100.0 * k / n
        extra = ""
        if D in actual:
            extra = "%.1f%% -> gap %+.1f pts (%d boxes)" % (
                100.0 * actual[D], frac - 100.0 * actual[D],
                k - round(actual[D] * n))
        print("  %-6d %-14s %-14s %s" % (D, "%4d (%5.1f%%)" % (k, frac),
                                         ("%5.1f%%" % (100.0 * actual[D])) if D in actual else "-",
                                         extra))

    print("\nGT side line vs red-ink centre offset (px), %d samples:" % len(offs))
    o = np.asarray(offs)
    print("  bias=%+.3f  sigma=%.3f  |off| p50=%.3f p90=%.3f p95=%.3f"
          % (o.mean(), o.std(), np.percentile(np.abs(o), 50),
             np.percentile(np.abs(o), 90), np.percentile(np.abs(o), 95)))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "summary": {
                       "n_gt": n,
                       "ortho_dev_p50": pct(dev, .50), "ortho_dev_p90": pct(dev, .90),
                       "ortho_dev_p95": pct(dev, .95),
                       "best_rect_px_p50": pct(res, .50),
                       "best_rect_px_p90": pct(res, .90),
                       "best_rect_px_p95": pct(res, .95),
                       "reachable": {str(D): sum(1 for x in res if x <= D)
                                     for D in DIST_VALUES},
                       "n_gt_total": 1076,
                       "detector": {"3": 857, "5": 901},
                       "off_bias": float(o.mean()), "off_sigma": float(o.std()),
                   }}, f, ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()