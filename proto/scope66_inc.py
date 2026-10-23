"""LOOP-66: incremental potential of the nested line candidates.

scope59 adds the nested candidates to the base detections and dedups them
away - the base already covers that quad (IoU ~0.96).  But a base detection
that overlaps a candidate with IoU 0.96 can still be a FALSE box (an FN of
the base pipeline) while the candidate is the correct one.

So the question is not "are the candidates good" but "for which GT boxes
that the BASE misses is there a candidate that matches, and what does that
candidate cost in false positives?".

This script, for every image:
  * computes the base detections (holes + edgefit)
  * determines which GT boxes the base misses (point match, D=5)
  * finds, among ALL line candidates, the one that matches each missed GT
    best, and reports its features
  * counts how many missed GTs are recoverable at all (a candidate with d<=5)

Usage: python proto/scope66_inc.py [--n 40] [--perim 0.90] ...
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance, match_boxes_point
from cv_hybrid import dedup_keep_order
import scope51_linerect as lr
import scope59_nested as sn
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect_holes,
                            detect_edgefit)
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--perim", type=float, default=0.90)
    ap.add_argument("--cov", type=float, default=0.60)
    ap.add_argument("--inter", type=float, default=0.35)
    ap.add_argument("--single", type=float, default=0.30)
    args = ap.parse_args()

    ids = get_ids()[:args.n]
    LRC = dict(lr.LR_DEFAULT)
    CFG = dict(HOLE_DEFAULT)
    PIPE = dict(PIPE_DEFAULT)

    rows = []
    nfn = 0
    nrec = 0
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
        # GT boxes the base misses (replicates the greedy matcher)
        cand = []
        for di in range(len(base)):
            for gj in range(len(gt)):
                d = corner_match_distance(base[di], np.asarray(gt[gj], np.int32))
                if d <= 5.0:
                    cand.append((d, di, gj))
        cand.sort()
        used_det, used_gt = set(), set()
        for d, di, gj in cand:
            if di in used_det or gj in used_gt:
                continue
            used_det.add(di)
            used_gt.add(gj)
        missed = [(gi, g) for gi, g in enumerate(gt) if gi not in used_gt]
        nfn += len(missed)
        cands = sn.line_candidates(mask, img, LRC)
        if not len(cands) or not missed:
            continue
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
        sup = lr._perim_support_batch(mask_d, cands, 48)
        lab, nh = sn.hole_label_mask(mask, CFG)
        f = sn.nested_features(lab, cands)
        # best candidate per missed GT
        for gi, g in missed:
            gg = np.asarray(g, np.int32)
            ds = np.array([corner_match_distance(c.astype(np.int32), gg)
                           for c in cands], np.float64)
            order = np.argsort(ds)
            best = None
            for k in order[:40]:
                if ds[k] > 12.0:
                    break
                cand = cands[k]
                sides = np.stack([np.linalg.norm(cand[(i + 1) % 4] - cand[i]) for i in range(4)])
                area = 0.5 * abs(np.sum(cand[:, 0] * np.roll(cand[:, 1], 1)
                                        - cand[:, 1] * np.roll(cand[:, 0], 1)))
                if area < P["min_area"] or sides.min() < P["min_side"]:
                    continue
                if sides.max() / max(sides.min(), 1e-6) > P["max_aspect"]:
                    continue
                best = k
                break
            if best is None:
                rows.append({"id": iid, "gt": gi, "d": None, "recoverable": 0})
                continue
            nrec += 1
            k = best
            rows.append({"id": iid, "gt": gi, "d": round(float(ds[k]), 2),
                         "recoverable": int(ds[k] <= 5.0),
                         "sup": round(float(sup[k]), 3),
                         "red": round(float(f["red"][k]), 3),
                         "cov": round(float(f["cov"][k]), 3),
                         "single": round(float(f["single"][k]), 3),
                         "nholes": int(f["nholes"][k]),
                         "cont": round(float(f["cont"][k]), 3),
                         "nbase": len(base)})
    print(f"images={len(ids)}  base FN(GT)={nfn}  with candidate={nrec}  "
          f"recoverable(d<=5)={sum(r['recoverable'] for r in rows)}")
    ok = [r for r in rows if r.get("d") is not None]
    for key, lim in (("sup", 0.97), ("red", 0.05), ("single", 0.90),
                     ("cov", 0.80), ("cont", 0.2)):
        for op, cmp in (("<=", "<="), (">=", ">=")):
            pass
    # how many recoverable candidates pass each single threshold
    print("\n-- single thresholds on the recoverable candidates --")
    for key, ths in (("sup", [0.85, 0.90, 0.93, 0.95, 0.97]),
                     ("red", [0.02, 0.05, 0.10, 0.20, 0.35]),
                     ("single", [0.5, 0.7, 0.8, 0.9]),
                     ("cov", [0.5, 0.7, 0.8, 0.9, 0.95]),
                     ("cont", [0.1, 0.2, 0.3, 0.5, 1.0])):
        s = []
        for t in ths:
            if key == "red" or key == "cont":
                c = sum(1 for r in ok if r["recoverable"] and r[key] <= t)
            else:
                c = sum(1 for r in ok if r["recoverable"] and r[key] >= t)
            s.append(f"{key}{'<' if key in ('red', 'cont') else '>'}{t}:{c}")
        print("  " + "  ".join(s))
    out = os.path.join(ROOT, "benchmark", "scope66_inc.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()