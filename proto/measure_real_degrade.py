"""How does a real GT box lose red support, and why is it still detected?

The real GT (Label@CARS_202609_raw) is the hand-drawn annotation burned into
Image@CARS_202609_annotated.  proto/scope73_ceiling.py showed that a box with
a low red support along its own sides is usually still found (9% miss at
support 0.5-0.7), so a "missing chunk of the outline" is NOT the whole story.

This probe measures, for every GT box of the 500 image benchmark:
  * sup[4]      - red support along the 4 sides (denoised red mask)
  * runs[4]     - the contiguous red runs along each side, with their
                   position given as a fraction of the side length
  * enclosed    - is the box interior still sealed by red (flood fill from the
                   interior does not reach the image border)?  A hole needs a
                   sealed interior, an edge fit does not.
  * tp           - matched by the production pipeline, and the corner distance

Usage: python proto/measure_real_degrade.py [--n 200] [--out benchmark/real_degrade.json]
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
from eval_pointmatch import corner_match_distance
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"


def side_runs(mask, g):
    """Per side: (support, [(start_frac, end_frac), ...]) of the red runs."""
    H, W = mask.shape
    out = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        L = float(np.linalg.norm(b - a))
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        hit = (mask[ys, xs] > 0)
        sup = float(hit.mean())
        runs = []
        k = 0
        while k < ns:
            if hit[k]:
                j = k
                while j + 1 < ns and hit[j + 1]:
                    j += 1
                runs.append((round(k / ns, 3), round((j + 1) / ns, 3)))
                k = j + 1
            else:
                k += 1
        out.append((sup, runs))
    return out


def interior_enclosed(mask, g, inset=3.0):
    """Is the box interior sealed by red?  (the free interior must not reach
    the image border)"""
    H, W = mask.shape
    c = np.asarray(g, float)
    ctr = c.mean(axis=0)
    # shrink the quad towards the centre
    q = ctr[None, :] + (c - ctr[None, :]) * (1.0 - inset / 60.0)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [q.astype(np.int32)], 255)
    free = (mask == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, 4)
    seed = tuple(int(round(v)) for v in ctr)
    if not (0 <= seed[0] < W and 0 <= seed[1] < H):
        return None
    l = lab[seed[1], seed[0]]
    if l == 0:
        return None                      # centre is red -> not a box interior
    comp = (lab == l)
    # touches the image border?
    touches = (comp[0, :].any() or comp[-1, :].any() or
               comp[:, 0].any() or comp[:, -1].any())
    inside_free = comp & (m > 0)
    if inside_free.sum() < 10:
        return None
    return bool(not touches)


def side_band_stats(mask, g, band=8.0):
    """Per side: how the red sits relative to the IDEAL side line.

    Returns (band_cover, off_p50, off_p90):
      band_cover - fraction of the side length that has red within +-band px
                   of the ideal line (1.0 = the line is red over its length)
      off_p50/90 - the perpendicular distance of that red from the ideal line
    """
    H, W = mask.shape
    out = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            out.append((0.0, 0.0, 0.0))
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        x0 = int(max(0, np.floor(min(a[0], b[0]) - band - 1)))
        x1 = int(min(W, np.ceil(max(a[0], b[0]) + band + 2)))
        y0 = int(max(0, np.floor(min(a[1], b[1]) - band - 1)))
        y1 = int(min(H, np.ceil(max(a[1], b[1]) + band + 2)))
        if x1 <= x0 or y1 <= y0:
            out.append((0.0, 0.0, 0.0))
            continue
        roi = mask[y0:y1, x0:x1] > 0
        ys, xs = np.nonzero(roi)
        if len(xs) == 0:
            out.append((0.0, 0.0, 0.0))
            continue
        pts = np.stack([xs + x0, ys + y0], axis=1).astype(float)
        rel = pts - a[None, :]
        t = rel @ u
        perp = rel @ n
        sel = (t >= -band) & (t <= L + band) & (np.abs(perp) <= band)
        t, perp = t[sel], perp[sel]
        if len(t) == 0:
            out.append((0.0, 0.0, 0.0))
            continue
        # 1 px resolution coverage of the side length
        bins = np.zeros(int(L) + 1, bool)
        idx = np.clip(np.round(t).astype(int), 0, int(L))
        bins[idx] = True
        cover = float(bins[:int(L) + 1].mean())
        out.append((cover, float(np.percentile(np.abs(perp), 50)),
                    float(np.percentile(np.abs(perp), 90))))
    return out


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
    dets = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)
    rows = []
    for g in gts:
        sr = side_runs(mask, g)
        sup = [s for s, _ in sr]
        best = 1e9
        for d in dets:
            best = min(best, corner_match_distance(np.asarray(d, np.int32),
                                                  np.asarray(g, np.int32)))
        rows.append({
            "id": iid,
            "sup": [round(s, 3) for s in sup],
            "sup_min": round(min(sup), 3),
            "runs": [[list(r) for r in rs] for _, rs in sr],
            "band": [[round(v, 3) for v in st] for st in side_band_stats(mask, g)],
            "enclosed": interior_enclosed(mask, g),
            "dist": None if best > 1e8 else round(float(best), 2),
            "tp": bool(best <= 5.0),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = all 500")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "real_degrade.json"))
    args = ap.parse_args()

    ids = get_ids()
    if args.n:
        ids = ids[:args.n]
    with Pool(args.workers) as pool:
        chunks = pool.map(work, ids, chunksize=4)
    rows = [r for c in chunks for r in c]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("rows: %d -> %s" % (len(rows), args.out))

    # --- aggregate -------------------------------------------------------
    def bucket(r):
        s = r["sup_min"]
        for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
            if lo <= s < hi:
                return (lo, hi)
        return None

    print("\n  bucket        n     TP    miss   enclosed(TP)  enclosed(FN)")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        b = bucket(r)
        if not b:
            continue
        a = agg[b]
        a[0] += 1
        if r["tp"]:
            a[1] += 1
            a[3] += 1 if r["enclosed"] else 0
        else:
            a[2] += 1
            a[4] += 1 if r["enclosed"] else 0
    for b in sorted(agg):
        n, tp, fn, enc_tp, enc_fn = agg[b]
        print("  [%.1f,%.1f)  %5d  %5d  %5.1f%%   %8.1f%%      %8.1f%%"
              % (b[0], b[1], n, tp, 100.0 * fn / n,
                 100.0 * enc_tp / max(tp, 1), 100.0 * enc_fn / max(fn, 1)))

    # --- where is the missing red? --------------------------------------
    print("\n  position of the missing part on the worst side")
    print("  (gap = the largest contiguous non-red stretch, as a fraction of the side)")
    stat = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        i = int(np.argmin(r["sup"]))
        runs = r["runs"][i]
        L = 1.0
        gaps = []
        prev = 0.0
        for a, b in runs:
            if a - prev > 1e-6:
                gaps.append((prev, a))
            prev = b
        if prev < 1.0 - 1e-6:
            gaps.append((prev, 1.0))
        if not gaps:
            stat[r["tp"]]["no_gap"] += 1
            continue
        g = max(gaps, key=lambda x: x[1] - x[0])
        at_corner = (g[0] <= 0.02) or (g[1] >= 0.98)
        stat[r["tp"]]["corner" if at_corner else "middle"] += 1
        stat[r["tp"]]["nruns%d" % len(runs)] += 1
    for tp in (True, False):
        c = stat[tp]
        tot = sum(c.values())
        print("    tp=%-5s n=%4d  gap@corner %5.1f%%  gap@middle %5.1f%%  "
              "runs: %s" % (tp, tot, 100.0 * c["corner"] / max(tot, 1),
                             100.0 * c["middle"] / max(tot, 1),
                             {k[5:]: v for k, v in sorted(c.items()) if k.startswith("nruns")}))

    # --- how many sides are broken --------------------------------------
    print("\n  number of sides with support < 0.9 (TP vs FN)")
    c = collections.Counter()
    for r in rows:
        k = sum(1 for s in r["sup"] if s < 0.9)
        c[(r["tp"], k)] += 1
    for k in range(5):
        print("    %d broken sides: TP %4d   FN %4d" % (k, c[(True, k)], c[(False, k)]))

    # --- is the red MISSING or merely OFFSET? ----------------------------
    print("\n  worst side: is the red missing (short) or offset (wobbling)?")
    print("  band_cover = red within +-8px of the ideal line / side length")
    print("  off_p50/p90 = |perpendicular distance| of that red")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        sel = [r for r in rows if lo <= r["sup_min"] < hi]
        if not sel:
            continue
        cov_tp, cov_fn = [], []
        off_tp, off_fn = [], []
        for r in sel:
            i = int(np.argmin(r["sup"]))
            cov, o50, o90 = r["band"][i]
            (cov_tp if r["tp"] else cov_fn).append(cov)
            (off_tp if r["tp"] else off_fn).append(o50)
        def med(v):
            return float(np.median(v)) if v else float("nan")
        print("    [%.1f,%.1f) n=%3d  TP: band_cover %.2f  off_p50 %.1f   |"
              "  FN: band_cover %.2f  off_p50 %.1f"
              % (lo, hi, len(sel), med(cov_tp), med(off_tp),
                 med(cov_fn), med(off_fn)))


if __name__ == "__main__":
    sys.exit(main())