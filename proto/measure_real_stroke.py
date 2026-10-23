"""Characterise the hand-drawn annotation of the real CARS data.

The emulation dataset draws perfect rectangles; the real annotation is drawn by
hand.  To make the emulation a meaningful proxy we have to know how the real
stroke differs from the ideal rectangle:

  wobble   - max distance of the stroke pixels from the labelled side line
  width    - thickness of the stroke (per side)
  gaps     - fraction of the side without any stroke pixel
  overshoot- how far the stroke runs past the labelled corner
  fill     - contourArea / rectArea of the hole inside the box

Usage: python proto/measure_real_stroke.py [--n 120]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from scope45_holes import load_all


def side_pixels(mask, a, b, band=6.0):
    """Stroke pixels near the segment a-b and their distance to the line."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    H, W = mask.shape
    ts = np.arange(0, L, 0.5)
    off = np.arange(-band, band + 0.01, 0.5)
    T, O = np.meshgrid(ts, off, indexing="ij")
    pts = a[None, None, :] + T[:, :, None] * u[None, None, :] + O[:, :, None] * n[None, None, :]
    xs = np.clip(np.round(pts[:, :, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, :, 1]).astype(int), 0, H - 1)
    hit = mask[ys, xs] > 0
    return hit, ts, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--band", type=float, default=6.0)
    args = ap.parse_args()

    ids, gt_map, exp = load_all()
    ids = ids[:args.n]
    wobble = []
    widths = []
    gaps = []
    overshoot = []
    fills = []

    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        m = red_mask(img, 80, 110, 10)      # the loose red window of the eval
        for g in gt_map.get(iid, []):
            g = np.asarray(g, float)
            box_gap = 0.0
            box_wob = 0.0
            box_w = []
            for i in range(4):
                a, b = g[i], g[(i + 1) % 4]
                r = side_pixels(m, a, b, args.band)
                if r is None:
                    continue
                hit, ts, off = r
                if not hit.any():
                    box_gap += 1.0
                    continue
                # the stroke pixels of this side: those closest to the line
                dist = np.abs(off)
                per_t = []
                thick = []
                for j in range(hit.shape[0]):
                    col = hit[j]
                    if not col.any():
                        per_t.append(0.0)
                        continue
                    idx = np.nonzero(col)[0]
                    per_t.append(1.0)
                    thick.append(dist[idx].min() * 2)
                box_gap += 1.0 - float(np.mean(per_t))
                if thick:
                    box_w.append(float(np.median(thick)))
                # wobble: distance of the stroke centre line from the ideal line
                # fit the centre of the stroke per column
                cols = []
                for j in range(hit.shape[0]):
                    idx = np.nonzero(hit[j])[0]
                    if len(idx):
                        cols.append(off[idx].mean())
                if len(cols) > 4:
                    box_wob = max(box_wob, float(np.max(np.abs(cols))))
            wobble.append(box_wob)
            widths.append(float(np.median(box_w)) if box_w else 0.0)
            gaps.append(box_gap / 4.0)
            # overshoot: stroke pixels beyond the corner along the side
            for i in range(4):
                a, b = g[i], g[(i + 1) % 4]
                d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
                for end, sign in ((a, -1), (b, 1)):
                    p = end + d * (sign * 12)
                    r = side_pixels(m, p, p + d, 1.0)
                    if r is not None and r[0].any():
                        overshoot.append(sign)

    def rep(name, arr):
        a = np.array(arr, float) if arr else np.zeros(1)
        print("  %-10s n=%5d p10=%6.2f p25=%6.2f p50=%6.2f p75=%6.2f p90=%6.2f max=%6.2f"
              % (name, len(a), *np.percentile(a, [10, 25, 50, 75, 90]), a.max()))

    print("real annotation stroke, %d boxes" % len(wobble))
    rep("wobble", wobble)
    rep("width", widths)
    rep("gap", gaps)


if __name__ == "__main__":
    main()