"""LOOP-57: can the hole FRAGMENTS of one box be merged back together?

Visual evidence (tmp_fn50/raw_*.png): a hand-drawn red box whose interior
is crossed by ANOTHER box's side line is cut into 2+ holes.  The union of
those fragments is the original box, so a pairwise merge of two holes that
share several collinear side lines should recover it.

The risk is the opposite case: two GT boxes that merely TOUCH share exactly
one wall, and merging them would destroy a true positive.  This script
measures both sides of the trade-off on the real 500-image benchmark:

  * how many GT box pairs actually share a wall (the danger zone)
  * for every pair of holes that share >= N collinear side lines: is the
    merged quad a GT box, a fragment of one, or a merge of two GT boxes?

Usage: python proto/scope57_merge_diag.py [--share 2] [--limit N]
"""
import os, sys, json, time, collections, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, _shared_lines, _group_quad,
                           _red_support, _line_isect, _line_angle, _ang_diff)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
DIST = 5.0
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)


# ------------------------------------------------------------------ geometry

def gt_share_wall(a, b, dist_tol=3.0, ang_tol=3.0):
    """Number of (nearly) collinear side pairs between two GT quads."""
    n = 0
    for i in range(4):
        p1 = a[i].astype(float)
        d1 = a[(i + 1) % 4].astype(float) - p1
        for j in range(4):
            p2 = b[j].astype(float)
            d2 = b[(j + 1) % 4].astype(float) - p2
            if _ang_diff(_line_angle(d1), _line_angle(d2)) > ang_tol:
                continue
            nn = np.array([-d1[1], d1[0]])
            L1 = np.linalg.norm(d1)
            nn = nn / max(L1, 1e-6)
            if abs(float(np.dot(p2 - p1, nn))) <= dist_tol:
                n += 1
                break
    return n


def gt_pair_stats(gt_map, ids):
    """How many GT box pairs share a wall (i.e. would be damaged by merging)."""
    hist = collections.Counter()
    pairs = []
    for iid in ids:
        gt = gt_map[iid]
        for i in range(len(gt)):
            for j in range(i + 1, len(gt)):
                n = gt_share_wall(gt[i], gt[j])
                hist[n] += 1
                if n >= 1:
                    pairs.append((iid, i, j, n))
    return hist, pairs


def hole_boxlike(hole, cfg):
    """Is this hole shaped like a box interior?"""
    area = cv2.contourArea(hole)
    if area < cfg["hole_min_comp_area"]:
        return False
    rect = cv2.minAreaRect(hole)
    (cx, cy), (rw, rh), _ = rect
    if rw < cfg["hole_min_side"] or rh < cfg["hole_min_side"]:
        return False
    if area / max(rw * rh, 1e-6) < cfg["hole_fill"]:
        return False
    return True


def merge_pair(corners_a, lines_a, corners_b, lines_b):
    """Rectangle spanned by two holes: outermost line of each direction."""
    d0 = lines_a[0][1]
    d1 = lines_a[1][1]
    n0 = np.array([-d0[1], d0[0]])
    n1 = np.array([-d1[1], d1[0]])
    c = np.mean([corners_a.mean(axis=0), corners_b.mean(axis=0)], axis=0)
    cls0, cls1 = [], []
    for lines in (lines_a, lines_b):
        for p, d in lines:
            if _ang_diff(_line_angle(d), _line_angle(d0)) <= 45.0:
                cls0.append((float(np.dot(p - c, n1)), p, d))
            else:
                cls1.append((float(np.dot(p - c, n0)), p, d))
    if len(cls0) < 2 or len(cls1) < 2:
        return None
    cls0.sort(key=lambda t: t[0])
    cls1.sort(key=lambda t: t[0])
    a0, b0 = cls0[0], cls0[-1]
    a1, b1 = cls1[0], cls1[-1]
    q = []
    for (_, pa, da), (_, pb, db) in ((a0, a1), (a1, b0), (b0, b1), (b1, a0)):
        x = _line_isect(pa, da, pb, db)
        if x is None:
            return None
        q.append(x)
    return np.array(q, float)


def best_gt(q, gt, dist=DIST):
    best = None
    for gj, g in enumerate(gt):
        d = corner_match_distance(q.astype(np.int32), g)
        if best is None or d < best[0]:
            best = (d, gj)
    return best or (1e9, -1)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=int, default=2, help="min shared collinear lines")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = get_ids()
    if args.limit:
        ids = ids[:args.limit]
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}

    print("== GT pair wall sharing (all pairs of GT boxes) ==")
    hist, pairs = gt_pair_stats(gt_map, ids)
    for k in sorted(hist):
        print(f"  shared_lines={k}: {hist[k]} pairs")
    print(f"  -> pairs sharing >=1 wall: {len(pairs)}")

    stats = collections.Counter()
    rows = []
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        holes = _holes_of(mask, CFG)
        infos = []
        for h in holes:
            r = _hole_side_lines(h, mask, CFG)
            if r is not None and hole_boxlike(h, CFG):
                infos.append((h, r[0], r[1]))
        if len(infos) < 2:
            continue
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                sh = _shared_lines(infos[i][2], infos[j][2])
                if sh < args.share:
                    continue
                mq = merge_pair(infos[i][1], infos[i][2], infos[j][1], infos[j][2])
                if mq is None:
                    stats["no_merge"] += 1
                    continue
                dm, gm = best_gt(mq, gt)
                da, ga = best_gt(np.asarray(infos[i][1]), gt)
                db, gb = best_gt(np.asarray(infos[j][1]), gt)
                if dm <= DIST:
                    # merged quad is a GT box: does it damage the pieces?
                    a_ok = da <= DIST
                    b_ok = db <= DIST
                    if a_ok and b_ok:
                        stats["merge_ok_both_pieces_ok"] += 1
                    elif a_ok or b_ok:
                        stats["merge_ok_one_piece_lost"] += 1
                        rows.append({"id": iid, "sh": sh, "dm": round(dm, 1),
                                     "dA": round(da, 1), "dB": round(db, 1),
                                     "gt": gm, "kind": "one_piece_lost"})
                    else:
                        stats["merge_ok_no_piece_was"] += 1
                        rows.append({"id": iid, "sh": sh, "dm": round(dm, 1),
                                     "dA": round(da, 1), "dB": round(db, 1),
                                     "gt": gm, "kind": "recovered"})
                else:
                    if da <= DIST or db <= DIST:
                        stats["merge_wrong_piece_was_ok"] += 1
                    else:
                        stats["merge_wrong_nothing_ok"] += 1
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] {time.time()-t0:.0f}s")

    print(f"\n== pairwise hole merges (share >= {args.share}) ==")
    for k, v in stats.most_common():
        print(f"  {k:28s} {v:6d}")
    out = os.path.join(ROOT, "benchmark", f"scope57_merge_diag_{args.share}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"gt_pair_shared_hist": {str(k): v for k, v in hist.items()},
                   "gt_pairs_with_wall": len(pairs),
                   "merge_stats": dict(stats), "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()