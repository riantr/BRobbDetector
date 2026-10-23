"""LOOP-54: quality-aware merge of the hole quads and the line rectangles.

The hole quad of a MERGED region (two boxes whose interiors merge through a
gap in the annotation) cuts across the background -> its perimeter support is
low.  The correct rectangle built from the strokes has a support of ~1.0.
So when a hole quad and a line rectangle overlap, the one with the higher
perimeter support is the right one: sort by support and let the IoU dedup
keep the first.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_hybrid import dedup_keep_order
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, load_all, evaluate,
                           detect_holes, mask_cache)
from scope51_linerect import LR_DEFAULT, detect_linrect, _perim_support_batch

ROOT = r"D:\src\CodeArts\BRobbDetector"


def supports(mask, quads):
    if len(quads) == 0:
        return np.zeros(0)
    mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
    return _perim_support_batch(mask_d, np.asarray(quads, np.float64), 48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sup",
                    choices=["sup", "sup_nolines", "sup_linfirst"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--dedup", type=float, default=0.40)
    args = ap.parse_args()

    cfg = dict(LR_DEFAULT)
    for kv in filter(None, args.cfg.split(",")):
        k, v = kv.split("=")
        cfg[k] = float(v)
    pipe = dict(PIPE_DEFAULT)

    ids, gt_map, exp = load_all()
    if args.limit:
        ids = ids[:args.limit]

    det_map = {}
    t0 = time.time()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        if iid not in mask_cache:
            mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                          np.ones((3, 3), np.uint8),
                                          iterations=3) > 0
        holes = detect_holes(img, P, dict(HOLE_DEFAULT), pipe)
        lins = [np.asarray(q, np.int32) for q in detect_linrect(img, P, cfg)]
        sh = supports(mask, holes)
        sl = supports(mask, lins)
        allq = holes + lins
        sall = np.concatenate([sh, sl]) if len(allq) else np.zeros(0)
        order = np.argsort(-sall)
        det_map[iid] = dedup_keep_order([allq[i] for i in order], args.dedup)

    ndet, total_gt, per_D = evaluate(ids, gt_map, det_map, exp)
    print(f"LOOP-54 mode={args.mode} dedup={args.dedup} cfg={cfg}")
    print(f"det={ndet}  ({time.time()-t0:.0f}s)")
    for D in (3, 5, 8, 12):
        r = per_D[str(D)]
        print(f"  D={D:2d}  TP={r['tp']:4d}/{total_gt}  trueFP={r['true_fp']:4d} "
              f"real_unl={r['real_unl']:3d}  corr_f1={r['corrected_f1']:.1f}")

    out = os.path.join(ROOT, "benchmark", f"scope54_{args.mode}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "lr_cfg": cfg, "det": ndet,
                   "total_gt": total_gt, "per_D": per_D}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()