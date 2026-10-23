"""Quality check of the wobbly emulation: achieved vs targeted red coverage.

The generator solves the wobble amplitude so that the coverage of the ideal
side line matches the target of the recipe (the coverage measured on the real
data).  This script verifies that the rendered images really reach the target
with the PRODUCTION red mask (hue<=4, sat>=100, val>=140) and reports the
error, so that the emulation can be trusted as a proxy.

Usage: python proto/qc_wobbly.py --img-dir data/Image@emulcal --lab-dir data/Label@emulcal
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import red_mask
from scope45_holes import P
from gen_emulation import primary_mask

ROOT = r"D:\src\CodeArts\BRobbDetector"
COLOR_TYPE = {"红": "red", "绿": "green", "蓝": "blue"}


def side_band_cover(mask, g, band=8.0):
    """Fraction of each side that carries red within +-band px of the ideal
    line (1.0 = the side is red over its whole length).

    Same definition as the real measurement
    (proto/measure_real_degrade.py::side_band_stats) - a side with a
    band_cover < 1 is a *broken* side, and a broken side is what makes the
    detector lose the box.
    """
    H, W = mask.shape
    out = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            out.append(0.0)
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        x0 = int(max(0, np.floor(min(a[0], b[0]) - band - 1)))
        x1 = int(min(W, int(np.ceil(max(a[0], b[0]) + band + 2))))
        y0 = int(max(0, np.floor(min(a[1], b[1]) - band - 1)))
        y1 = int(min(H, int(np.ceil(max(a[1], b[1]) + band + 2))))
        if x1 <= x0 or y1 <= y0:
            out.append(0.0)
            continue
        roi = mask[y0:y1, x0:x1] > 0
        ys, xs = np.nonzero(roi)
        if len(xs) == 0:
            out.append(0.0)
            continue
        pts = np.stack([xs + x0, ys + y0], axis=1).astype(float)
        rel = pts - a[None, :]
        t = rel @ u
        perp = rel @ n
        sel = (t >= -band) & (t <= L + band) & (np.abs(perp) <= band)
        t = t[sel]
        if len(t) == 0:
            out.append(0.0)
            continue
        bins = np.zeros(int(L) + 1, bool)
        idx = np.clip(np.round(t).astype(int), 0, int(L))
        bins[idx] = True
        out.append(float(bins[:int(L) + 1].mean()))
    return out


def side_support(mask, g):
    H, W = mask.shape
    sups = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        L = float(np.linalg.norm(b - a))
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        sups.append(float((mask[ys, xs] > 0).mean()))
    return sups


def side_band_width(mask, g, band=6.0):
    """Perpendicular extent (px) of the red band along each side.

    Measured exactly like the real annotation (proto/measure_real_line.py):
    in 12 windows along the side, the distance between the first and the last
    red pixel across the band.  The real annotation measures p10 3.0 /
    p50 4.0 / p90 7.5 px with the production mask - the emulation must reach
    the same value, otherwise the coverage of the ideal line reacts to a
    different distance than in the real data.
    """
    H, W = mask.shape
    out = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            out.append(0.0)
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None, :] + ts[:, None] * d[None, :]
        widths = []
        step = max(4, ns // 12)
        for t0 in range(0, max(1, ns - 3), step):
            sel = []
            for t in range(t0, min(t0 + step, ns)):
                px = int(round(pts[t, 0]))
                py = int(round(pts[t, 1]))
                for o in range(-int(band), int(band) + 1):
                    x = int(round(px + n[0] * o))
                    y = int(round(py + n[1] * o))
                    if 0 <= x < W and 0 <= y < H and mask[y, x]:
                        sel.append(o)
            if sel:
                widths.append(max(sel) - min(sel) + 1)
        out.append(float(np.percentile(widths, 50)) if widths else 0.0)
    return out


def work(task):
    stem, lab_dir, img_dir = task
    with open(os.path.join(lab_dir, stem + ".json"), encoding="utf-8") as f:
        lab = json.load(f)
    img = cv2.imread(os.path.join(img_dir, stem + ".jpg"))
    if img is None:
        return []
    out = []
    masks = {}
    for bi, info in enumerate(lab.get("infos", [])):
        cname = info.get("type", "红")
        if cname not in masks:
            masks[cname] = primary_mask(img, cname)
        mask = masks[cname]
        pts = np.array([[int(x), int(y)] for x, y in
                        (p.split(",") for p in info["points"].split(";"))], float)
        sup = side_support(mask, pts)
        out.append({"stem": stem, "idx": bi, "type": cname, "sup": sup,
                    "min": min(sup),
                    "band": side_band_width(mask, pts),
                    "cover": side_band_cover(mask, pts)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=os.path.join(ROOT, "data", "Image@emulcal"))
    ap.add_argument("--lab-dir", default=os.path.join(ROOT, "data", "Label@emulcal"))
    ap.add_argument("--target", default=os.path.join(ROOT, "benchmark", "real_gt_recipe.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="", help="restrict to one colour")
    ap.add_argument("--out", default="", help="write the per box records to JSON")
    args = ap.parse_args()

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(args.lab_dir, "*.json")))
    with Pool(args.workers) as pool:
        chunks = pool.map(work, [(s, args.lab_dir, args.img_dir) for s in stems],
                          chunksize=4)
    rows = [r for c in chunks for r in c]
    print("boxes: %d" % len(rows))
    only = args.only
    if only:
        rows = [r for r in rows if r["type"] == only]

    tgt = json.load(open(args.target, encoding="utf-8"))
    tmin = np.array([min(s[0] for s in rec) for rec in tgt])
    amin = np.array([r["min"] for r in rows])

    hist = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    print("\n  min-side coverage      target      achieved")
    for lo, hi in hist:
        t = 100.0 * np.mean((tmin >= lo) & (tmin < hi))
        a = 100.0 * np.mean((amin >= lo) & (amin < hi))
        print("    [%.1f,%.1f)         %5.1f%%      %5.1f%%" % (lo, hi, t, a))
    print("\n  mean min coverage  target %.3f  achieved %.3f" % (tmin.mean(), amin.mean()))

    # per side error (achieved vs the recipe targets is not matched 1:1, so
    # compare the marginal distributions of the single side coverage)
    tsides = np.array([s[0] for rec in tgt for s in rec])
    asides = np.array([v for r in rows for v in r["sup"]])
    print("\n  single side coverage    target      achieved")
    for lo, hi in hist:
        t = 100.0 * np.mean((tsides >= lo) & (tsides < hi))
        a = 100.0 * np.mean((asides >= lo) & (asides < hi))
        print("    [%.1f,%.1f)         %5.1f%%      %5.1f%%" % (lo, hi, t, a))
    print("\n  mean side coverage   target %.3f  achieved %.3f" % (tsides.mean(), asides.mean()))

    # width of the red band across the side (the real annotation: p50 4.0 px)
    bands = np.array([v for r in rows for v in r.get("band", [])])
    bands = bands[bands > 0]
    if len(bands):
        print("\n  red band width (px)   p10 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f"
              % tuple(np.percentile(bands, [10, 25, 50, 75, 90])))
        print("  (real annotation: p10 3.0  p25 3.5  p50 4.0  p75 4.0  p90 7.5)")

    # broken sides: the red does not reach the ideal line over the whole side
    nmiss = []
    for r in rows:
        nmiss.append(sum(1 for c in r.get("cover", []) if c < 0.999))
    nmiss = np.array(nmiss)
    print("\n  broken sides per box   " +
          "  ".join("%d: %5.1f%%" % (k, 100.0 * (nmiss == k).mean())
                    for k in range(5)))
    print("  (real annotation: 0: 78.1%  1: 15.6%  2: 3.8%  3: 1.3%  4: 0.9%)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())