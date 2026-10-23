"""LOOP-59: hole-NESTED line candidates.

The hole pipeline misses boxes whose interior is not one clean hole:
  * merged interior  - the interior is connected to a neighbour's interior
                       through a gap, so the hole is TOO BIG
  * split interior   - a stroke crosses the interior, so the box is covered by
                       several hole fragments

In both cases the 4 strokes of the box are present in the mask, so the box can
be rebuilt from the strokes (the line detector of scope51).  The global line
detector is far too noisy (thousands of candidates), but a candidate that is
NESTED with a hole region is a strong signal:

  * merged case : the true box lies INSIDE the merged hole
  * split case  : the true box CONTAINS the fragments

So: keep the line candidates whose interior is mostly hole and whose
perimeter is red, and add them to the hole detections.

Usage:
  python proto/scope59_nested.py [--perim 0.9] [--cov 0.6] [--inter 0.35] [--n 500]
"""
import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from cv_hybrid import dedup_keep_order
from quad_ortho import is_orthogonal_quad
import scope51_linerect as lr
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, detect_holes, detect_edgefit,
                           evaluate, mask_cache)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
DIST_VALUES = [3, 5, 8, 12]


def line_candidates(mask, img, cfg):
    """All rectangles built from the red strokes (no filtering)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return np.zeros((0, 4, 2))
    pts = np.stack([xs, ys], 1).astype(float)
    strokes = []
    for ang in lr._angles(pts, cfg):
        strokes += lr._strokes_at(pts, ang, cfg)
    strokes = lr._dedup(strokes, cfg)
    fams = lr._families(strokes, cfg)
    allc = []
    for fi, fa in enumerate(fams):
        for fb in fams[fi + 1:]:
            diff = min(abs(fa["ang"] - fb["ang"]) % 180.0,
                       180 - abs(fa["ang"] - fb["ang"]) % 180.0)
            if abs(diff - 90.0) > 8.0:
                continue
            PA = np.array([L["point"] for L in fa["lines"]])
            DA = np.array([L["dir"] for L in fa["lines"]])
            PB = np.array([L["point"] for L in fb["lines"]])
            DB = np.array([L["dir"] for L in fb["lines"]])
            oA = PA @ lr._perp(DA[0])
            oB = PB @ lr._perp(DB[0])
            ia, ja = np.triu_indices(len(PA), 1)
            ib, jb = np.triu_indices(len(PB), 1)
            sA = np.abs(oA[ja] - oA[ia])
            sB = np.abs(oB[jb] - oB[ib])
            okA = (sA >= cfg["lr_min_side"]) & (sA <= cfg["lr_max_side"])
            okB = (sB >= cfg["lr_min_side"]) & (sB <= cfg["lr_max_side"])
            if not okA.any() or not okB.any():
                continue
            AA, BB = np.meshgrid(np.nonzero(okA)[0], np.nonzero(okB)[0], indexing="ij")
            AA = AA.ravel()
            BB = BB.ravel()
            sa = sA[AA]
            sb = sB[BB]
            asp = np.maximum(sa, sb) / np.maximum(np.minimum(sa, sb), 1e-6)
            k = asp <= cfg["lr_max_aspect"]
            AA, BB = AA[k], BB[k]
            X, okm = lr._isect_matrix(PA, DA, PB, DB)
            ia2 = ia[AA]
            ja2 = ja[AA]
            ib2 = ib[BB]
            jb2 = jb[BB]
            c = np.empty((len(AA), 4, 2))
            c[:, 0] = X[ia2, ib2]
            c[:, 1] = X[ia2, jb2]
            c[:, 2] = X[ja2, jb2]
            c[:, 3] = X[ja2, ib2]
            good = okm[ia2, ib2] & okm[ia2, jb2] & okm[ja2, jb2] & okm[ja2, ib2]
            inb = ((c[:, :, 0].min(1) > -5) & (c[:, :, 0].max(1) < img.shape[1] + 5)
                   & (c[:, :, 1].min(1) > -5) & (c[:, :, 1].max(1) < img.shape[0] + 5))
            allc.append(c[good & inb])
    if not allc:
        return np.zeros((0, 4, 2))
    return np.vstack(allc)


def hole_label_mask(mask, cfg):
    """Label image: 0 = background, 1 = red, >=2 = hole id + 1."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    lab[mask > 0] = 1
    holes = _holes_of(mask, cfg)
    for i, h in enumerate(holes):
        pts = h.reshape(-1, 2).astype(np.int32)
        x0 = max(int(pts[:, 0].min()), 0)
        x1 = min(int(pts[:, 0].max()), W - 1)
        y0 = max(int(pts[:, 1].min()), 0)
        y1 = min(int(pts[:, 1].max()), H - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(sub, [pts - np.array([x0, y0])], 1)
        reg = lab[y0:y1 + 1, x0:x1 + 1]
        reg[(sub > 0) & (reg == 0)] = i + 2
    return lab, len(holes)


def nested_stats(lab, corners, nsamp=5):
    """For each candidate: fraction of its interior that is hole, the largest
    fraction covered by a SINGLE hole, and the red fraction inside."""
    if len(corners) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    H, W = lab.shape
    s = (np.arange(nsamp) + 0.5) / nsamp
    S, T = np.meshgrid(s, s, indexing="ij")
    a = corners[:, 0][:, None, None, :]
    b = corners[:, 1][:, None, None, :]
    c = corners[:, 3][:, None, None, :]
    P = (a + S[None, :, :, None] * (b - a)
         + T[None, :, :, None] * (c - a))
    Q = P.reshape(-1, 2)
    xs = np.clip(np.round(Q[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(Q[:, 1]).astype(int), 0, H - 1)
    v = lab[ys, xs].reshape(len(corners), nsamp * nsamp)
    n = len(corners)
    red = (v == 1).mean(axis=1)
    is_hole = v >= 2
    cov = is_hole.mean(axis=1)
    nlab = int(lab.max()) + 1
    M = np.zeros((n, nlab), np.int32)
    rows = np.repeat(np.arange(n), nsamp * nsamp)
    np.add.at(M, (rows, v.ravel()), 1)
    M[:, 0] = 0
    single = M.max(axis=1) / (nsamp * nsamp)
    return cov, single, red


def nested_features(lab, corners, nsamp=5):
    """Full feature set of every candidate (see scope65 for the analysis).

    cov     fraction of the interior covered by hole regions
    single  largest fraction covered by a SINGLE hole
    red     red fraction inside
    nholes  number of distinct hole regions the interior touches
    cont    fraction of the PERIMETER samples that fall inside a hole region
    """
    n = len(corners)
    if n == 0:
        z = np.zeros(0)
        return {"cov": z, "single": z, "red": z, "nholes": z, "cont": z}
    H, W = lab.shape
    s = (np.arange(nsamp) + 0.5) / nsamp
    S, T = np.meshgrid(s, s, indexing="ij")
    a = corners[:, 0][:, None, None, :]
    b = corners[:, 1][:, None, None, :]
    c = corners[:, 3][:, None, None, :]
    Pm = a + S[None, :, :, None] * (b - a) + T[None, :, :, None] * (c - a)
    Q = Pm.reshape(-1, 2)
    xs = np.clip(np.round(Q[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(Q[:, 1]).astype(int), 0, H - 1)
    v = lab[ys, xs].reshape(n, nsamp * nsamp)
    red = (v == 1).mean(axis=1)
    is_hole = v >= 2
    cov = is_hole.mean(axis=1)
    nlab = int(lab.max()) + 1
    M = np.zeros((n, nlab), np.int32)
    rows = np.repeat(np.arange(n), nsamp * nsamp)
    np.add.at(M, (rows, v.ravel()), 1)
    M[:, 0] = 0
    single = M.max(axis=1) / (nsamp * nsamp)
    Mh = M.copy()
    Mh[:, 1] = 0
    nholes = (Mh > 0).sum(axis=1)
    # perimeter samples inside a hole region
    nn = 24
    ts = (np.arange(nn) + 0.5) / nn
    on_hole = np.zeros(n, np.int64)
    for i in range(4):
        p1 = corners[:, i][:, None, :]
        p2 = corners[:, (i + 1) % 4][:, None, :]
        pts = p1 + ts[None, :, None] * (p2 - p1)
        xx = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
        yy = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
        on_hole += (lab[yy, xx] >= 2).sum(axis=1)
    cont = on_hole / float(4 * nn)
    return {"cov": cov, "single": single, "red": red, "nholes": nholes, "cont": cont}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perim", type=float, default=0.90)
    ap.add_argument("--cov", type=float, default=0.60)
    ap.add_argument("--inter", type=float, default=0.35)
    ap.add_argument("--single", type=float, default=0.30)
    ap.add_argument("--nholes", type=int, default=99)
    ap.add_argument("--cont", type=float, default=1.0)
    ap.add_argument("--ortho", type=float, default=0.0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    LRC = dict(lr.LR_DEFAULT)
    LRC["lr_perim"] = args.perim
    LRC["lr_interior"] = args.inter
    CFG = dict(HOLE_DEFAULT)
    PIPE = dict(PIPE_DEFAULT)

    ids = get_ids()
    if args.n:
        ids = ids[:args.n]
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    exp = {}
    from common import load_expected
    exp = load_expected()

    det_map = {}
    stats = collections.Counter()
    t0 = time.time()
    for k, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        base = detect_holes(img, P, CFG, PIPE) + detect_edgefit(img, P, CFG, PIPE)
        base = dedup_keep_order(base, PIPE["dedup_iou"])
        cands = line_candidates(mask, img, LRC)
        extra = []
        if len(cands):
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
            sup = lr._perim_support_batch(mask_d, cands, 48)
            lab, nh = hole_label_mask(mask, CFG)
            f = nested_features(lab, cands)
            keep = (sup >= args.perim) & (f["red"] <= args.inter) & (f["cov"] >= args.cov) \
                & (f["single"] >= args.single) & (f["nholes"] <= args.nholes) \
                & (f["cont"] <= args.cont)
            # geometric sanity: area / side length / orthogonality
            if keep.any():
                kc = cands[keep]
                sides = np.stack([np.linalg.norm(kc[:, (i + 1) % 4] - kc[:, i], axis=1)
                                  for i in range(4)], 1)
                area = 0.5 * np.abs(np.sum(
                    kc[:, :, 0] * np.roll(kc[:, :, 1], 1, axis=1)
                    - kc[:, :, 1] * np.roll(kc[:, :, 0], 1, axis=1), axis=1))
                g = ((area >= P["min_area"]) & (sides.min(axis=1) >= P["min_side"])
                     & (sides.max(axis=1) <= P["max_side"])
                     & (sides.max(axis=1) / np.maximum(sides.min(axis=1), 1e-6)
                        <= P["max_aspect"]))
                if args.ortho > 0:
                    Hh, Ww = img.shape[:2]
                    og = np.array([
                        is_orthogonal_quad(c, Ww, Hh, args.ortho, P["boundary_band"])
                        for c in kc])
                    g = g & og
                keep2 = np.zeros(len(cands), bool)
                idx = np.nonzero(keep)[0]
                keep2[idx] = g
                keep = keep2
            stats["cand"] += len(cands)
            stats["keep"] += int(keep.sum())
            extra = [c.astype(np.int32) for c in cands[keep]]
        det_map[iid] = dedup_keep_order(base + extra, PIPE["dedup_iou"])
        if iid not in mask_cache:
            mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                          np.ones((3, 3), np.uint8), iterations=3) > 0
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)

    from scope45_holes import evaluate as ev
    ndet, total_gt, per_D = ev(ids, gt_map, det_map, exp)
    print(f"\ndet={ndet}  ({time.time()-t0:.0f}s)")
    for D in DIST_VALUES:
        r = per_D[str(D)]
        print(f"  D={D:2d}  TP={r['tp']:4d}/{total_gt}  trueFP={r['true_fp']:4d} "
              f"real_unl={r['real_unl']:3d}  corr_f1={r['corrected_f1']:.1f}")
    print("stats:", dict(stats))
    out = os.path.join(ROOT, "benchmark",
                       "scope59_nested" + (("_" + args.tag) if args.tag else "") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "det": ndet, "total_gt": total_gt,
                   "per_D": per_D, "stats": dict(stats)}, f, ensure_ascii=False, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()