"""Decisive diagnostic: is the OptiNav ridge response a usable STROKE-CENTRE
locator, even though its global mask is texture-dominated?

For every GT box side, the cross-section profile of the ridge strength is
sampled in a +-4 px band around the GT line.  If the ridge detector localises
the annotation stroke centre, the argmax of that profile must sit at offset
~0 and with a small spread.  Several input channels are compared because the
annotation is a bright line in the red channel but not necessarily in the
grey one.

Output per channel:
  hit   fraction of cross-sections whose ridge argmax is within 1 px of 0
  bias  mean signed offset of the argmax
  sigma std of the argmax offset
  peak  mean ridge strength of the argmax (contrast of the locator)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from optinav_transfer_eval import find_ridges
from quad_ortho import quad_on_boundary

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def channels(img):
    b, g, r = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), \
        img[:, :, 2].astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    redness = r - np.maximum(g, b)
    return {
        "r": r,
        "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32),
        "redness": redness,
        "sat": hsv[:, :, 1].astype(np.float32),
    }


def main():
    ids = get_ids()[:150]
    OFF = np.arange(-4.0, 4.01, 0.5)
    res = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        quads = [g for g in load_boxes_any(RAW, iid)
                 if not quad_on_boundary(g, W, H, 0.0)]
        if not quads:
            continue
        chans = channels(img)
        st = {}
        for name, ch in chans.items():
            _keep, strength = find_ridges(ch, sigma=2.0, min_thin=1e9)
            st[name] = strength
        for q in quads:
            q = np.asarray(q, float)
            for i in range(4):
                a = q[i]
                b = q[(i + 1) % 4]
                v = b - a
                L = float(np.linalg.norm(v))
                if L < 8:
                    continue
                u = v / L
                n = np.array([-u[1], u[0]])
                ts = np.linspace(0.1, 0.9, 12)
                base = a[None, :] + ts[:, None] * v[None, :]
                pts = base[:, None, :] + OFF[None, :, None] * n[None, None, :]
                xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
                ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
                for name in chans:
                    prof = st[name][ys, xs]          # (12, len(OFF))
                    j = np.argmax(prof, axis=1)
                    off = OFF[j]
                    peak = prof[np.arange(len(j)), j]
                    res.setdefault(name, {}).setdefault("off", []).extend(off)
                    res.setdefault(name, {}).setdefault("peak", []).extend(peak)
    print("%-9s %8s %8s %8s %8s %8s" % ("channel", "hit(1px)", "bias", "sigma",
                                         "peak_med", "peak_mean"))
    for name in ("r", "gray", "redness", "sat"):
        off = np.asarray(res[name]["off"])
        peak = np.asarray(res[name]["peak"])
        hit = float(np.mean(np.abs(off) <= 1.0))
        print("%-9s %8.3f %8.3f %8.3f %8.2f %8.2f"
              % (name, hit, off.mean(), off.std(),
                 np.median(peak), peak.mean()))
    # histogram of the offset for the best-looking channel
    for name in ("redness", "r"):
        off = np.asarray(res[name]["off"])
        h, e = np.histogram(off, bins=np.arange(-4.25, 4.26, 0.5))
        print("\noffset histogram, channel %s (n=%d)" % (name, off.size))
        for i in range(len(h)):
            print("  %+5.1f  %6.1f%%  %s" % (e[i], 100.0 * h[i] / off.size,
                                             "#" * int(h[i] / max(off.size / 100, 1))))
        # how often is the GT line itself near a ridge STRONGER than the local max? 
        print("  frac |argmax|<=1px = %.3f" % np.mean(np.abs(off) <= 1.0))


if __name__ == "__main__":
    main()