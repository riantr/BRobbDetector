"""P1: per-image config selection and config ENSEMBLES.

Motivation (measured): LOOP-24 found Stage-2 denoise significant PER IMAGE
(p=0.005) while LOOP-28 found it flat IN AGGREGATE -> the effect is
heterogeneous (helps some images, hurts others), so a single global config
cannot harvest it.

This runs K configs on every image of the real 500 set and evaluates several
fusion rules that use NO ground truth:

  base            the shipped pm_v10 config alone
  union_all       union of the K configs, re-deduped
  vote2 / vote3   keep a quad only when >=2 / >=3 of the K configs emit a
                  matching quad (corner distance <= TAU) - deployable consensus
  select_sup      per image, the config with the highest mean own red support
  select_mass     per image, the config with the highest total red support mass

and two ORACLE upper bounds (GT used, not deployable) that tell us whether
there is anything to win at all:

  oracle_pick     per image, the single config with the most TP at D=5
  oracle_union    the union with every non-TP removed  (best any FP filter can do)

Scored with the GT-corrected metric on the real 500 images.

Usage: python proto/p1_per_image.py --workers 12 --out benchmark/p1_per_image.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from cv_hybrid import dedup_keep_order
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]
TAU = 3.0

CONFIGS = {
    "A_shipped": {},
    "B_denoise0": {"denoise": 0},
    "C_denoise120": {"denoise": 120},
    "D_support035": {"support": 0.35},
}
ORDER = list(CONFIGS)


def mask_of(img, p):
    m = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
    if p["denoise_min_area"] > 0:
        m = denoise(m, p["denoise_min_area"])
    return m


def cfg_of(over):
    p = dict(P)
    cfg = dict(HOLE_DEFAULT)
    if "denoise" in over:
        p["denoise_min_area"] = over["denoise"]
    if "support" in over:
        cfg["min_red_support"] = over["support"]
    return p, cfg


def qdist(a, b):
    """Greedy 4-corner max distance (the point-match notion of 'same quad')."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dm = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    used = set()
    worst = 0.0
    for i in range(4):
        order = np.argsort(dm[i])
        for j in order:
            if j not in used:
                used.add(int(j))
                worst = max(worst, float(dm[i, j]))
                break
    return worst


def _work(task):
    iid, = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, {}
    out = {}
    for name in ORDER:
        p, cfg = cfg_of(CONFIGS[name])
        m = mask_of(img, p)
        det = detect(img, p, cfg, PIPE_DEFAULT, mask=m)
        rec = []
        for q in det:
            qf = np.asarray(q, float)
            rec.append((np.asarray(q, np.int32).tolist(), float(_red_support(m, qf))))
        out[name] = rec
    return iid, out


def fused(union_in, votes, tau=TAU):
    """Dedup the union, then keep quads supported by >= votes distinct configs."""
    keep = []
    for q in union_in["quads"]:
        sup_names = set()
        for name in ORDER:
            for qq, _s in union_in["per"][name]:
                if qdist(q, qq) <= tau:
                    sup_names.add(name)
                    break
        if len(sup_names) >= votes:
            keep.append(q)
    return keep


def score(det_map, ids, gt_map, exp):
    masks = {}
    for iid in ids:
        if not det_map.get(iid):
            continue
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        masks[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                np.ones((3, 3), np.uint8), iterations=3) > 0
    res = {}
    for D in DIST_VALUES:
        tp = fp = unl = ndet = 0
        for iid in ids:
            det = det_map.get(iid, [])
            gt = gt_map.get(iid, [])
            ndet += len(det)
            t, unmatched = match_boxes_point(det, gt, float(D))
            tp += t
            md = masks.get(iid)
            cn = exp.get(iid, 0)
            for di in unmatched:
                if md is not None and is_real_unlabeled(det[di], cn, len(gt), md):
                    unl += 1
                else:
                    fp += 1
        ngt = sum(len(gt_map.get(i, [])) for i in ids)
        rec = tp / max(ngt, 1) * 100
        prec = tp / max(tp + fp, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        res[str(D)] = {"tp": tp, "true_fp": fp, "real_unl": unl, "n_det": ndet,
                       "recall": round(rec, 1), "precision": round(prec, 1),
                       "corr_f1": round(f1, 1)}
    return res


def show(name, r):
    print("  %-16s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
          "| dets=%4d  corr_f1@5=%.1f"
          % (name, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"], r["5"]["true_fp"],
             r["8"]["tp"], r["8"]["true_fp"], r["12"]["tp"], r["12"]["true_fp"],
             r["5"]["n_det"], r["5"]["corr_f1"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/p1_per_image.json")
    ap.add_argument("--cache", default="benchmark/_tmp_p1_cache.json")
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [g for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()
    ngt = sum(len(v) for v in gt_map.values())
    print("%d images, %d GT boxes" % (len(ids), ngt), flush=True)

    raw = {}
    if os.path.exists(args.cache):
        raw = json.load(open(args.cache, encoding="utf-8"))
        print("loaded cache %s (%d images)" % (args.cache, len(raw)), flush=True)
    else:
        with Pool(args.workers) as pool:
            for k, (iid, out) in enumerate(pool.imap_unordered(
                    _work, [(i,) for i in ids], chunksize=2)):
                raw[iid] = out
                if (k + 1) % 100 == 0:
                    print("  %d/%d" % (k + 1, len(ids)), flush=True)
        json.dump(raw, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False)
        print("cached -> %s" % args.cache, flush=True)

    def quads_of(per, name):
        return [np.asarray(q, np.int32) for q, _s in per.get(name, [])]

    strategies = {}
    # 1) base + each single config
    for name in ORDER:
        strategies[name] = {iid: quads_of(raw[iid], name) for iid in ids}

    # 2) union + votes
    union_quads = {}
    for iid in ids:
        per = raw[iid]
        allq = []
        for name in ORDER:
            allq.extend(quads_of(per, name))
        union_quads[iid] = dedup_keep_order(allq, 0.40)
    strategies["union_all"] = union_quads

    # 3) self-supervised single-config selection
    sel_sup, sel_mass = {}, {}
    for iid in ids:
        per = raw[iid]
        best_sup, best_mass = None, None
        bs, bm = -1e9, -1e9
        for name in ORDER:
            rec = per.get(name, [])
            if not rec:
                continue
            s = [v for _q, v in rec]
            if float(np.mean(s)) > bs:
                bs, best_sup = float(np.mean(s)), name
            if float(np.sum(s)) > bm:
                bm, best_mass = float(np.sum(s)), name
        sel_sup[iid] = quads_of(per, best_sup) if best_sup else []
        sel_mass[iid] = quads_of(per, best_mass) if best_mass else []
    strategies["select_sup"] = sel_sup
    strategies["select_mass"] = sel_mass

    # 4) oracles
    or_pick, or_union = {}, {}
    for iid in ids:
        gt = gt_map.get(iid, [])
        best_tp, best_name = -1, None
        for name in ORDER:
            t, _u = match_boxes_point(quads_of(raw[iid], name), gt, 5.0)
            if t > best_tp:
                best_tp, best_name = t, name
        or_pick[iid] = quads_of(raw[iid], best_name) if best_name else []
        t, unmatched = match_boxes_point(union_quads[iid], gt, 5.0)
        keep = [q for i, q in enumerate(union_quads[iid]) if i not in set(unmatched)]
        or_union[iid] = keep
    strategies["oracle_pick"] = or_pick
    strategies["oracle_union"] = or_union

    # 5) vote2 / vote3 on the union
    vote2, vote3 = {}, {}
    for iid in ids:
        per = raw[iid]
        prox = {"quads": union_quads[iid], "per": per}
        vote2[iid] = fused(prox, 2)
        vote3[iid] = fused(prox, 3)

    out = {}
    print("\n== single configs ==")
    for name in ORDER:
        r = score(strategies[name], ids, gt_map, exp)
        out[name] = r
        show(name, r)
    print("\n== fusion (no GT used) ==")
    for name in ("union_all", "vote2", "vote3", "select_sup", "select_mass"):
        det = strategies.get(name)
        if det is None:
            det = vote2 if name == "vote2" else vote3
        r = score(det, ids, gt_map, exp)
        out[name] = r
        show(name, r)
    print("\n== ORACLE upper bounds (GT used, NOT deployable) ==")
    for name in ("oracle_pick", "oracle_union"):
        r = score(strategies[name], ids, gt_map, exp)
        out[name] = r
        show(name, r)

    base = out["A_shipped"]["5"]
    print("\n== delta vs shipped config A at D=5 ==")
    for name in out:
        if name == "A_shipped":
            continue
        d = out[name]["5"]
        print("  %-16s dTP %+4d  dTrueFP %+4d  dcorr_f1 %+5.1f  dD3TP %+4d"
              % (name, d["tp"] - base["tp"], d["true_fp"] - base["true_fp"],
                 d["corr_f1"] - base["corr_f1"],
                 out[name]["3"]["tp"] - out["A_shipped"]["3"]["tp"]))

    json.dump({"strategies": out, "ngt": ngt, "configs": CONFIGS},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)
    print("(union quads per image were also written to the strategy tables above;"
          " delete %s to re-run from scratch)" % args.cache)


if __name__ == "__main__":
    main()