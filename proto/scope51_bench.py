"""LOOP-51 bench: evaluate the line-based rectangle detector (alone and
combined with the hole pipeline) on the 500-image true-GT benchmark."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG
from canny_quad import get_ids
from cv_hybrid import dedup_keep_order
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, load_all, evaluate,
                           detect_holes, detect_edgefit, mask_cache)
from scope51_linerect import LR_DEFAULT, detect_linrect

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lin",
                    choices=["lin", "holes", "both", "both_sep", "both_linfirst",
                             "both_edge"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cfg", default="")
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
        if iid not in mask_cache:
            mask_cache[iid] = cv2.dilate(
                cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR),
                np.ones((3, 3), np.uint8), iterations=3) > 0
        holes = []
        lins = []
        if args.mode in ("holes", "both", "both_sep", "both_linfirst", "both_edge"):
            holes = detect_holes(img, P, dict(HOLE_DEFAULT), pipe)
        if args.mode in ("lin", "both", "both_sep", "both_linfirst", "both_edge"):
            lins = [np.asarray(q, np.int32) for q in detect_linrect(img, P, cfg)]
        if args.mode == "both_sep":
            det_map[iid] = (dedup_keep_order(holes, pipe["dedup_iou"])
                            + dedup_keep_order(lins, pipe["dedup_iou"]))
        elif args.mode == "both_linfirst":
            det_map[iid] = dedup_keep_order(lins + holes, pipe["dedup_iou"])
        elif args.mode == "both":
            det_map[iid] = dedup_keep_order(holes + lins, pipe["dedup_iou"])
        else:
            det_map[iid] = dedup_keep_order(holes + lins, pipe["dedup_iou"])

    ndet, total_gt, per_D = evaluate(ids, gt_map, det_map, exp)
    print(f"LOOP-51 mode={args.mode} cfg={cfg}")
    print(f"det={ndet}  ({time.time()-t0:.0f}s)")
    for D in (3, 5, 8, 12):
        r = per_D[str(D)]
        print(f"  D={D:2d}  TP={r['tp']:4d}/{total_gt}  trueFP={r['true_fp']:4d} "
              f"real_unl={r['real_unl']:3d}  corr_f1={r['corrected_f1']:.1f}")

    out = os.path.join(ROOT, "benchmark", f"scope51_{args.mode}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "lr_cfg": cfg, "det": ndet,
                   "total_gt": total_gt, "per_D": per_D}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()