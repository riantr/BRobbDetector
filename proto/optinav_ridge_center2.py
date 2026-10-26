"""Head-to-head: which locator puts the side line closest to the GT line?

For the SAME cross-sections (bands around every GT box side, +-4 px) compare
  1. mask centroid   - centroid of the binary production red mask in the band
                       (this is what the shipped side fit measures)
  2. redness centroid- centroid of the redness weight R-max(G,B) in the band
  3. ridge argmax    - argmax of the OptiNav ridge strength on the redness
                       channel (the +-0.5 px grid of the profile)
  4. ridge subpixel  - parabolic sub-pixel refinement around the argmax
and reports the signed offset from the GT line for each.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from optinav_transfer_eval import find_ridges
from quad_ortho import quad_on_boundary

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def main():
    ids = get_ids()[:150]
    OFF = np.arange(-4.0, 4.01, 0.5)
    out = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        quads = [g for g in load_boxes_any(RAW, iid)
                 if not quad_on_boundary(g, W, H, 0.0)]
        if not quads:
            continue
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        redness = r - np.maximum(g, b)
        m = red_mask(img, 100, 140, 4) > 0
        _keep, st = find_ridges(redness, sigma=2.0, min_thin=1e9)

        for q in quads:
            q = np.asarray(q, float)
            for i in range(4):
                a = q[i]
                vv = q[(i + 1) % 4] - a
                L = float(np.linalg.norm(vv))
                if L < 8:
                    continue
                u = vv / L
                n = np.array([-u[1], u[0]])
                ts = np.linspace(0.1, 0.9, 12)
                base = a[None, :] + ts[:, None] * vv[None, :]
                pts = base[:, None, :] + OFF[None, :, None] * n[None, None, :]
                xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
                ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
                rw = redness[ys, xs]
                mm = m[ys, xs]
                for row in range(len(ts)):
                    cv_ = rw[row]
                    cm_ = mm[row]
                    rec = {}
                    if cm_.any():
                        rec["mask_centroid"] = float((OFF[cm_]).mean())
                    if cv_.sum() > 1:
                        rec["redness_centroid"] = float((OFF * np.maximum(cv_, 0)).sum()
                                                        / max(np.maximum(cv_, 0).sum(), 1e-9))
                    s = st[ys[row], xs[row]]
                    j = int(np.argmax(s))
                    rec["ridge_argmax"] = float(OFF[j])
                    if 0 < j < len(s) - 1:
                        d1 = s[j - 1] - s[j]
                        d2 = s[j + 1] - s[j]
                        den = d1 + d2
                        if abs(den) > 1e-9:
                            dx = 0.5 * (d1 - d2) / den
                            rec["ridge_subpx"] = float(OFF[j] + np.clip(dx, -1, 1) * 0.5)
                    for k, v in rec.items():
                        out.setdefault(k, []).append(v)
    print("%-18s %7s %8s %8s %8s %8s %8s" %
          ("locator", "n", "hit1px", "hit0.5", "|off|med", "bias", "sigma"))
    order = ["mask_centroid", "redness_centroid", "ridge_argmax", "ridge_subpx"]
    for k in order:
        if k not in out:
            continue
        o = np.asarray(out[k])
        print("%-18s %7d %8.3f %8.3f %8.3f %8.3f %8.3f" %
              (k, o.size, float(np.mean(np.abs(o) <= 1.0)),
               float(np.mean(np.abs(o) <= 0.5)), float(np.median(np.abs(o))),
               float(o.mean()), float(o.std())))
    # joint: mask centroid limited to the same cross-sections as ridge_subpx
    print("\npaired (only cross-sections where BOTH exist):")
    n = min(len(out.get("mask_centroid", [])), 0)
    # rebuild pairing properly
    print("  (see per-locator table above; pairing done in optinav_ridge_pair.py)")


if __name__ == "__main__":
    main()