"""Probe: can the OptiNav Find Ridges ridge map be tuned into a usable mask?

Sweeps the channel (red / grey), the Gaussian radius, the thinning factor and
a ridge-strength threshold, and reports the GT-perimeter coverage and the
in-GT-band precision of the resulting mask.  The production mask is the
reference: cover 0.87 / in-band 0.59.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from optinav_transfer_eval import prod_mask, perimeter_pixels, gt_band
from quad_ortho import quad_on_boundary

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def ridge(gray, sigma, min_thin):
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
    lm = tp - root
    strength = -lm
    den1 = Lxy * Lxy + (lm - Lxx) ** 2
    den2 = (lm - Lyy) ** 2 + Lxy * Lxy
    use1 = den1 > den2
    sq1 = np.sqrt(np.maximum(den1, 1e-12))
    sq2 = np.sqrt(np.maximum(den2, 1e-12))
    u = np.where(use1, Lxy / sq1, (lm - Lyy) / sq2)
    v = np.where(use1, (lm - Lxx) / sq1, Lxy / sq2)
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
    n = p.size
    max_points = max(int(n / max(min_thin, 1.0)), 1)
    cnt = int(keep.sum())
    if cnt > max_points:
        idx = np.flatnonzero(keep.ravel())
        crit = strength.ravel()[idx]
        order = np.argsort(crit, kind="stable")
        flat = keep.ravel()
        flat[idx[order[:cnt - max_points]]] = False
        keep = flat.reshape(p.shape)
    return keep, strength


def main():
    ids = get_ids()[:120]
    variants = []
    for ch in ("r", "gray"):
        for sigma in (1.0, 2.0, 3.0):
            for thin in (10.0, 100.0):
                for thr in (0.0, 0.5, 1.0, 2.0, 4.0):
                    variants.append((ch, sigma, thin, thr))
    acc = {v: {"cov": [], "prec": [], "px": []} for v in variants}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        quads = [g for g in load_boxes_any(RAW, iid)
                 if not quad_on_boundary(g, W, H, 0.0)]
        if not quads:
            continue
        per = perimeter_pixels(quads, H, W) > 0
        band = gt_band(quads, H, W)
        npix = int(per.sum())
        chan = {"r": img[:, :, 2], "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
        cache = {}
        for v in variants:
            ch, sigma, thin, thr = v
            key = (ch, sigma, thin)
            if key not in cache:
                cache[key] = ridge(chan[ch], sigma, thin)
            keep, strength = cache[key]
            m = keep & (strength >= thr)
            b = m
            n = int(b.sum())
            if n == 0:
                acc[v]["cov"].append(0.0)
                acc[v]["prec"].append(0.0)
                acc[v]["px"].append(0)
                continue
            acc[v]["cov"].append(float((b & per).sum()) / max(npix, 1))
            acc[v]["prec"].append(float((b & band).sum()) / n)
            acc[v]["px"].append(n)
    rows = []
    for v in variants:
        a = acc[v]
        rows.append((float(np.mean(a["prec"])), float(np.mean(a["cov"])),
                     float(np.mean(a["px"])), v))
    rows.sort(reverse=True)
    print("%-38s %8s %8s %8s" % ("variant", "prec", "cover", "px"))
    for prec, cov, px, v in rows[:25]:
        print("%-38s %8.3f %8.3f %8.0f" % ("ch=%s s=%.0f thin=%.0f thr=%.1f" % v,
                                           prec, cov, px))
    print("\n... worst 5")
    for prec, cov, px, v in rows[-5:]:
        print("%-38s %8.3f %8.3f %8.0f" % ("ch=%s s=%.0f thin=%.0f thr=%.1f" % v,
                                           prec, cov, px))


if __name__ == "__main__":
    main()