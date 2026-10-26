"""Diagnostic: does the OptiNav Find Ridges ridge map land on the annotation
stroke CENTRE (which is where the GT side line runs)?

For every GT box the perpendicular distance of every red-channel ridge point
to the nearest GT side line is measured, plus the ridge coverage of the GT
perimeter.  If the ridge map localises the stroke centre the distance
histogram must peak at 0 and the coverage must be high.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from optinav_transfer_eval import find_ridges, prod_mask
from quad_ortho import quad_on_boundary

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def side_distance(pts, quads):
    """Min |distance| of every point to any side line of any quad."""
    out = []
    for q in quads:
        q = np.asarray(q, float)
        d = None
        for i in range(4):
            a = q[i]
            b = q[(i + 1) % 4]
            v = b - a
            L = float(np.linalg.norm(v))
            if L < 1e-6:
                continue
            u = v / L
            n = np.array([-u[1], u[0]])
            dist = np.abs((pts - a[None, :]) @ n)
            d = dist if d is None else np.minimum(d, dist)
        if d is not None:
            out.append(d)
    if not out:
        return np.zeros(0)
    return np.minimum.reduce([d for d in out])


def main():
    ids = get_ids()[:200]
    stats = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        quads = [g for g in load_boxes_any(RAW, iid)
                 if not quad_on_boundary(g, W, H, 0.0)]
        if not quads:
            continue
        r = img[:, :, 2]
        keep, strength = find_ridges(r, sigma=2.0, min_thin=10.0)
        ys, xs = np.nonzero(keep)
        rp = np.stack([xs, ys], axis=1).astype(float)
        # GT perimeter sample points
        pp = []
        for q in quads:
            q = np.asarray(q, float)
            for i in range(4):
                a = q[i]
                b = q[(i + 1) % 4]
                L = float(np.linalg.norm(b - a))
                n = max(int(L), 4)
                ts = np.linspace(0.0, 1.0, n, endpoint=False)
                pp.append(a[None, :] + ts[:, None] * (b - a)[None, :])
        pp = np.concatenate(pp)
        # distance of the GT perimeter to the nearest ridge point
        d_gt = side_distance(pp, quads)  # placeholder, replaced below
        # coverage: fraction of GT perimeter pixels that are ridge points
        pxi = np.clip(np.round(pp[:, 0]).astype(int), 0, W - 1)
        pyi = np.clip(np.round(pp[:, 1]).astype(int), 0, H - 1)
        cov = float(keep[pyi, pxi].mean())
        # signed offset of the ridge points near each side line
        near = []
        for q in quads:
            q = np.asarray(q, float)
            for i in range(4):
                a = q[i]
                b = q[(i + 1) % 4]
                v = b - a
                L = float(np.linalg.norm(v))
                if L < 1e-6:
                    continue
                u = v / L
                n = np.array([-u[1], u[0]])
                rel = rp - a[None, :]
                t = rel @ u
                d = rel @ n
                sel = (t > 0) & (t < L) & (np.abs(d) < 6.0)
                if sel.any():
                    near.append(d[sel])
        near = np.concatenate(near) if near else np.zeros(0)
        k = iid
        stats.setdefault("cov", []).append(cov)
        stats.setdefault("n_ridge", []).append(int(keep.sum()))
        stats.setdefault("near_n", []).append(int(near.size))
        if near.size:
            stats.setdefault("near_d", []).append(near)
    print("images=%d  ridge px/image=%.0f  GT-perimeter ridge coverage=%.3f"
          % (len(stats["cov"]), np.mean(stats["n_ridge"]), np.mean(stats["cov"])))
    allnear = np.concatenate(stats["near_d"]) if stats.get("near_d") else np.zeros(0)
    print("ridge points within 6px of a GT side line: %d" % allnear.size)
    if allnear.size:
        print("  signed distance to the GT line: p5=%.2f p25=%.2f p50=%.2f "
              "p75=%.2f p95=%.2f  mean=%.3f"
              % (np.percentile(allnear, 5), np.percentile(allnear, 25),
                 np.percentile(allnear, 50), np.percentile(allnear, 75),
                 np.percentile(allnear, 95), allnear.mean()))
        h, edges = np.histogram(allnear, bins=np.arange(-6, 6.1, 1.0))
        for i in range(len(h)):
            print("   %+5.1f..%+5.1f  %6d" % (edges[i], edges[i + 1], h[i]))


if __name__ == "__main__":
    main()