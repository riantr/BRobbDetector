"""Profile the hand-drawn stroke of the real annotation.

proto/measure_real_degrade.py showed that a real GT box with a low red
support is NOT missing a chunk of the outline: the red covers the whole side
(band_cover = 1.00) but wanders 1-4 px away from the ideal side line.  This
script measures that wander so that the emulation can reproduce it:

  * offset(t)  - the perpendicular distance of the red band centre from the
                  ideal side line, sampled along the side
  * thickness  - the local stroke thickness (red pixels across the band)
  * wobble     - the offset profile after removing the straight part, i.e. how
                  much the pen meanders around the ideal line

Usage: python proto/measure_real_wobble.py [--n 200]
"""
import argparse
import collections
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
from scope45_holes import P, RAW
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"
BAND = 8.0


def side_profile(mask, g, i, nbin=12):
    """offset(t) and thickness(t) of the red band along side i."""
    H, W = mask.shape
    a = np.asarray(g[i], float)
    b = np.asarray(g[(i + 1) % 4], float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    x0 = int(max(0, np.floor(min(a[0], b[0]) - BAND - 1)))
    x1 = int(min(W, np.ceil(max(a[0], b[0]) + BAND + 2)))
    y0 = int(max(0, np.floor(min(a[1], b[1]) - BAND - 1)))
    y1 = int(min(H, np.ceil(max(a[1], b[1]) + BAND + 2)))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = mask[y0:y1, x0:x1] > 0
    ys, xs = np.nonzero(roi)
    if len(xs) == 0:
        return None
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    rel = pts - a[None, :]
    t = rel @ u
    perp = rel @ n
    sel = (t >= 0) & (t <= L) & (np.abs(perp) <= BAND)
    t, perp = t[sel], perp[sel]
    if len(t) < 5:
        return None
    edges = np.linspace(0, L, nbin + 1)
    offs, ths = [], []
    for k in range(nbin):
        m = (t >= edges[k]) & (t < edges[k + 1] if k < nbin - 1 else t <= edges[-1])
        if m.sum() < 2:
            offs.append(None)
            ths.append(None)
            continue
        p = perp[m]
        offs.append(float(np.median(p)))
        ths.append(int(m.sum()))
    return {"L": L, "off": offs, "th": ths,
            "sup": float((np.abs(perp) < 0.5).mean())}


def work(iid):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return []
    H, W = img.shape[:2]
    gts = load_boxes_any(RAW, iid)
    gts = [g for g in gts if not quad_on_boundary(g, W, H, 0.0)]
    if not gts:
        return []
    mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    mask = denoise(mask, P["denoise_min_area"])
    out = []
    for g in gts:
        for i in range(4):
            pr = side_profile(mask, g, i)
            if pr:
                out.append({"id": iid, "side": i, **pr})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "real_wobble.json"))
    args = ap.parse_args()
    ids = get_ids()
    if args.n:
        ids = ids[:args.n]
    with Pool(args.workers) as pool:
        chunks = pool.map(work, ids, chunksize=4)
    rows = [r for c in chunks for r in c]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("side profiles: %d -> %s" % (len(rows), args.out))

    # offset distribution, per support bucket
    print("\n  |offset| distribution of the red band centre (px)")
    print("  bucket        n   p25   p50   p75   p90   p99   mean")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        vals = []
        for r in rows:
            if not (lo <= r["sup"] < hi):
                continue
            vals += [abs(o) for o in r["off"] if o is not None]
        if not vals:
            continue
        v = np.array(vals)
        print("    [%.1f,%.1f)  %5d  %4.1f  %4.1f  %4.1f  %4.1f  %4.1f  %4.1f"
              % (lo, hi, len(v), np.percentile(v, 25), np.percentile(v, 50),
                 np.percentile(v, 75), np.percentile(v, 90), np.percentile(v, 99),
                 v.mean()))

    # stroke thickness (red pixels across the band, per 1/12 of the side)
    print("\n  stroke thickness (red px across the band) per support bucket")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        vals = [t for r in rows if lo <= r["sup"] < hi for t in r["th"] if t]
        if not vals:
            continue
        v = np.array(vals)
        print("    [%.1f,%.1f)  n=%6d  p10 %4.1f  p50 %4.1f  p90 %4.1f"
              % (lo, hi, len(v), np.percentile(v, 10), np.percentile(v, 50),
                 np.percentile(v, 90)))

    # how smooth is the wander?  lag-1 autocorrelation of the offset profile
    print("\n  wander smoothness: corr(offset(t), offset(t+1 bin))")
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9)):
        cs = []
        for r in rows:
            if not (lo <= r["sup"] < hi):
                continue
            o = [x for x in r["off"] if x is not None]
            if len(o) >= 4:
                a = np.array(o[:-1])
                b = np.array(o[1:])
                if a.std() > 0.05 and b.std() > 0.05:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
        if cs:
            print("    [%.1f,%.1f)  n=%4d  corr %.2f" % (lo, hi, len(cs), np.mean(cs)))

    # a few concrete profiles
    print("\n  example offset profiles (px, 12 bins along the side)")
    shown = 0
    for r in rows:
        if r["sup"] < 0.4 and all(o is not None for o in r["off"]):
            print("    %s side %d  L=%5.1f  sup=%.2f  %s" % (
                r["id"], r["side"], r["L"], r["sup"],
                " ".join("%+4.1f" % o for o in r["off"])))
            shown += 1
            if shown >= 12:
                break


if __name__ == "__main__":
    sys.exit(main())