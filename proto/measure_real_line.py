"""Measure the production-mask stroke of the real annotation.

  width  - the thickness of the red band (perpendicular extent) of the
           production red mask along the GT side
  cover  - the coverage of the IDEAL side line (1 px wide) along the side
  runs   - the number of contiguous covered stretches along the side

The real annotation is a thin hand drawn line: the box is drawn around the
cargo, the label is the ideal rectangle, and the pen does not follow the ideal
edge exactly.  This script quantifies both effects so that the emulation can
reproduce them (proto/gen_emulation.py --stroke-model wobbly).

Usage: python proto/measure_real_line.py [--n 120]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, RAW
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def side_line(mask, g, i, band=6.0):
    """(cover, runs, width_p50, width_p90) of side i of the quad g."""
    H, W = mask.shape
    a = np.asarray(g[i], float)
    b = np.asarray(g[(i + 1) % 4], float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    ns = max(int(L), 2)
    ts = np.linspace(0, 1, ns)
    pts = a[None, :] + ts[:, None] * d[None, :]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    hit = (mask[ys, xs] > 0)
    cover = float(hit.mean())
    runs = 0
    k = 0
    while k < ns:
        if hit[k]:
            runs += 1
            while k + 1 < ns and hit[k + 1]:
                k += 1
        k += 1
    # perpendicular extent of the red band, in 4 px long windows
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
    if not widths:
        return cover, runs, 0.0, 0.0
    w = np.array(widths, float)
    return cover, runs, float(np.percentile(w, 50)), float(np.percentile(w, 90))


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
            r = side_line(mask, g, i)
            if r:
                out.append({"id": iid, "cover": r[0], "runs": r[1],
                            "w50": r[2], "w90": r[3], "L": float(np.linalg.norm(
                                np.asarray(g[(i + 1) % 4], float) - np.asarray(g[i], float)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    ids = sorted(os.path.splitext(f)[0]
                 for f in os.listdir(IMG) if f.endswith(".jpg"))[:args.n]
    with Pool(args.workers) as pool:
        chunks = pool.map(work, ids, chunksize=4)
    rows = [r for c in chunks for r in c]
    print("sides: %d" % len(rows))

    print("\n  band width of the production red mask (px)")
    w = np.array([r["w50"] for r in rows], float)
    w = w[w > 0]
    print("    p10 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f  p99 %.1f"
          % (*np.percentile(w, [10, 25, 50, 75, 90, 99]),))

    print("\n  coverage of the ideal side line, by band width")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        sel = [r for r in rows if lo <= r["w50"] < hi]
        if not sel:
            continue
        c = np.array([r["cover"] for r in sel])
        ru = np.array([r["runs"] for r in sel], float)
        print("    w=%d-%d  n=%5d  cover p25 %.2f p50 %.2f p75 %.2f   runs p50 %.1f p90 %.1f"
              % (lo, hi, len(sel), np.percentile(c, 25), np.percentile(c, 50),
                 np.percentile(c, 75), np.percentile(ru, 50), np.percentile(ru, 90)))

    print("\n  number of covered runs vs coverage")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        sel = [r for r in rows if lo <= r["cover"] < hi]
        if not sel:
            continue
        ru = np.array([r["runs"] for r in sel], float)
        print("    cover [%.1f,%.1f)  n=%5d  runs p25 %.0f p50 %.0f p75 %.0f p90 %.0f"
              % (lo, hi, len(sel), np.percentile(ru, 25), np.percentile(ru, 50),
                 np.percentile(ru, 75), np.percentile(ru, 90)))


if __name__ == "__main__":
    sys.exit(main())