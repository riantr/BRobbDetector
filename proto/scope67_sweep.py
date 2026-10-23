"""LOOP-67: end-to-end sweep of the nested-candidate rule (fast version).

scope66 showed the nested line candidates can recover only ~5 of 25 missed GT
boxes (20%) on 40 images.  Before spending more effort on this lever we need
the END-TO-END effect of a rule:

  extras   = candidates that pass the rule
  merged   = dedup(base + extras)
  new      = merged detections that the base did not already produce
  dTP      = new detections that match a GT box the base missed
  dFP      = new detections that match nothing

The cache (built once) stores, per image:
  base detections, GT boxes, the candidates that pass the geometric filter,
  their features, their IoU against every base detection, and the distance
  from every candidate to every GT box.  The sweep itself is then pure
  numpy / small loops.

Usage:
  python proto/scope67_sweep.py --n 40 --rebuild
"""
import os, sys, json, pickle, argparse, itertools, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask, iou_obb
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
from cv_hybrid import dedup_keep_order
import scope51_linerect as lr
import scope59_nested as sn
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect_holes, detect_edgefit
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
PERMS = list(itertools.permutations(range(4)))


def corner_dist_batch(cands, gts):
    """(N,M) matrix of corner_match_distance (min over the 24 permutations of
    the MAX corner distance)."""
    N = len(cands)
    M = len(gts)
    if N == 0 or M == 0:
        return np.zeros((N, M))
    C = np.asarray(cands, np.float64)
    G = np.asarray(gts, np.float64)
    out = np.full((N, M), 1e9)
    for p in PERMS:
        d = np.linalg.norm(C[:, None, :, :] - G[None, :, p, :], axis=3).max(axis=2)
        out = np.minimum(out, d)
    return out


def build_cache(ids, min_sup):
    LRC = dict(lr.LR_DEFAULT)
    CFG = dict(HOLE_DEFAULT)
    PIPE = dict(PIPE_DEFAULT)
    cache = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        base = detect_holes(img, P, CFG, PIPE) + detect_edgefit(img, P, CFG, PIPE)
        base = dedup_keep_order(base, PIPE["dedup_iou"])
        gts = [np.asarray(g, np.int32) for g in gt]
        # GT boxes the base misses (replicates the greedy matcher)
        pairs = []
        for di in range(len(base)):
            for gj in range(len(gts)):
                d = corner_match_distance(base[di], gts[gj])
                if d <= 5.0:
                    pairs.append((d, di, gj))
        pairs.sort()
        used_det, used_gt = set(), set()
        for d, di, gj in pairs:
            if di in used_det or gj in used_gt:
                continue
            used_det.add(di)
            used_gt.add(gj)
        missed = np.array([g not in used_gt for g in range(len(gts))])

        cands = sn.line_candidates(mask, img, LRC)
        entry = {"id": iid, "base": base, "gt": gts, "missed": missed, "cand": None}
        if not len(cands):
            cache.append(entry)
            continue
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
        sup = lr._perim_support_batch(mask_d, cands, 48)
        lab, nh = sn.hole_label_mask(mask, CFG)
        f = sn.nested_features(lab, cands)
        sides = np.stack([np.linalg.norm(cands[:, (i + 1) % 4] - cands[:, i], axis=1)
                          for i in range(4)], 1)
        area = 0.5 * np.abs(np.sum(
            cands[:, :, 0] * np.roll(cands[:, :, 1], 1, axis=1)
            - cands[:, :, 1] * np.roll(cands[:, :, 0], 1, axis=1), axis=1))
        geom = ((area >= P["min_area"]) & (sides.min(axis=1) >= P["min_side"])
                & (sides.max(axis=1) <= P["max_side"])
                & (sides.max(axis=1) / np.maximum(sides.min(axis=1), 1e-6)
                   <= P["max_aspect"]))
        sel = (sup >= min_sup) & geom
        q = cands[sel].astype(np.float32)
        entry["cand"] = {
            "q": q,
            "sup": sup[sel], "red": f["red"][sel], "cov": f["cov"][sel],
            "single": f["single"][sel], "nholes": f["nholes"][sel],
            "cont": f["cont"][sel],
            "iou_base": np.array([[iou_obb(x.astype(np.int32), b) for b in base]
                                  for x in q], np.float32).reshape(len(q), len(base)),
            "dgt": corner_dist_batch(q, gts),
        }
        cache.append(entry)
    return cache


def eval_rule(cache, rule, iou_thr=0.40, D=5.0):
    dTP = dFP = 0
    nkept = 0
    for e in cache:
        c = e["cand"]
        if c is None:
            continue
        k = ((c["sup"] >= rule["sup"]) & (c["red"] <= rule["red"])
             & (c["cov"] >= rule["cov"]) & (c["single"] >= rule["single"])
             & (c["nholes"] <= rule["nholes"]) & (c["cont"] <= rule["cont"]))
        if not k.any():
            continue
        nkept += int(k.sum())
        q = c["q"][k]
        ib = c["iou_base"][k]
        dg = c["dgt"][k]
        # drop candidates that duplicate a base detection
        keep2 = (ib < iou_thr).all(axis=1) if ib.shape[1] else np.ones(len(q), bool)
        q = q[keep2]
        dg = dg[keep2]
        if not len(q):
            continue
        # dedup the survivors against each other (order = cache order)
        used = []
        for i in range(len(q)):
            if all(iou_obb(q[i].astype(np.int32), q[j].astype(np.int32)) < iou_thr
                   for j in used):
                used.append(i)
        q = q[used]
        dg = dg[used]
        # match survivors to the GT boxes the base missed
        order = np.argsort(dg, axis=None)
        uq = set(range(len(q)))
        ug = set()
        for flat in order:
            di, gj = divmod(int(flat), dg.shape[1])
            if dg[di, gj] > D:
                continue
            if di not in uq or gj in ug:
                continue
            if not e["missed"][gj]:
                continue
            uq.discard(di)
            ug.add(gj)
            dTP += 1
        dFP += len(uq)
    return dTP, dFP, nkept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-sup", type=float, default=0.85)
    ap.add_argument("--cache", type=str, default="benchmark/scope67_cache.pkl")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    ids = get_ids()[:args.n]
    cpath = os.path.join(ROOT, args.cache)
    if args.rebuild or not os.path.exists(cpath):
        t0 = time.time()
        cache = build_cache(ids, args.min_sup)
        with open(cpath, "wb") as f:
            pickle.dump(cache, f, protocol=4)
        print(f"cache built: {len(cache)} images in {time.time()-t0:.0f}s -> {cpath}")
    else:
        with open(cpath, "rb") as f:
            cache = pickle.load(f)
    tot = sum(len(e["gt"]) for e in cache)
    nmiss = sum(int(e["missed"].sum()) for e in cache)
    print(f"images={len(cache)}  gt={tot}  base-missed={nmiss}")

    grid = {
        "sup": [0.85, 0.90, 0.93, 0.95, 0.97],
        "red": [0.05, 0.10, 0.20, 0.35, 1.0],
        "cov": [0.40, 0.60, 0.80, 0.90],
        "single": [0.0, 0.30, 0.70, 0.90],
        "nholes": [1, 2, 3, 5],
        "cont": [0.2, 0.5, 1.0],
    }
    keys = list(grid)
    results = []
    t0 = time.time()
    for combo in itertools.product(*[grid[k] for k in keys]):
        rule = dict(zip(keys, combo))
        tp, fp, nk = eval_rule(cache, rule)
        results.append({"dTP": tp, "dFP": fp, "kept": nk, "rule": rule})
    print(f"swept {len(results)} rules in {time.time()-t0:.0f}s")
    good = [r for r in results if r["dTP"] > 0]
    good.sort(key=lambda r: (-(r["dTP"] * 2 - r["dFP"] * 3), -r["dTP"]))
    print(f"\n-- {len(good)} rules with dTP>0, top 25 by (2*dTP-3*dFP) --")
    for r in good[:25]:
        print(f"  dTP={r['dTP']:3d} dFP={r['dFP']:3d} kept={r['kept']:4d}  {r['rule']}")
    print("\n-- top 15 by dTP --")
    for r in sorted(good, key=lambda r: -r["dTP"])[:15]:
        print(f"  dTP={r['dTP']:3d} dFP={r['dFP']:3d} kept={r['kept']:4d}  {r['rule']}")
    out = os.path.join(ROOT, "benchmark", "scope67_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()