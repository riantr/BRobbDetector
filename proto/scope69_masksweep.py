"""LOOP-69: re-tune the RED MASK thresholds for the hole pipeline.

The mask thresholds (hue_max 4 / sat_min 100 / val_min 140) were tuned on
2026-10-21 for the OLD edgefit pipeline.  The visual inspection of the
remaining FNs shows that many annotation strokes are FADED: the bottom and
left sides of the GT box have no red pixels in the mask at all, so the box
never gets an enclosed interior and can never produce a hole.

A more permissive mask (lower sat_min / val_min) closes those faded strokes
and creates the missing interior.  The risk is that the orange X-ray object
texture (hue 9-10) floods the mask - but that is controlled by hue_max, not
by sat/val, so the hue window stays narrow.

Usage:
  python proto/scope69_masksweep.py --n 500 --sat 60,80,100 --val 90,110,130,140
"""
import os, sys, json, time, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, DIST_VALUES,
                            detect, evaluate, mask_cache)

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sat", default="100")
    ap.add_argument("--val", default="140")
    ap.add_argument("--hue", default="4")
    ap.add_argument("--out", default="benchmark/scope69_masksweep.json")
    args = ap.parse_args()

    ids = get_ids()
    if args.n:
        ids = ids[:args.n]
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    from common import load_expected
    exp = load_expected()
    cfg = dict(HOLE_DEFAULT)
    pipe = dict(PIPE_DEFAULT)

    sats = [float(x) for x in args.sat.split(",")]
    vals = [float(x) for x in args.val.split(",")]
    hues = [float(x) for x in args.hue.split(",")]
    rows = []
    for hue, sat, val in itertools.product(hues, sats, vals):
        p = dict(P)
        p["hue_max"] = hue
        p["sat_min"] = sat
        p["val_min"] = val
        t0 = time.time()
        det_map = {}
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                det_map[iid] = []
                continue
            if iid not in mask_cache:
                mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                              np.ones((3, 3), np.uint8),
                                              iterations=3) > 0
            det_map[iid] = detect(img, p, cfg, pipe)
        ndet, total_gt, per_D = evaluate(ids, gt_map, det_map, exp)
        r5 = per_D["5"]
        r3 = per_D["3"]
        row = {"hue": hue, "sat": sat, "val": val, "det": ndet,
               "tp3": r3["tp"], "tp5": r5["tp"], "fp5": r5["true_fp"],
               "f1_5": round(r5["corrected_f1"], 1),
               "f1_3": round(r3["corrected_f1"], 1), "sec": round(time.time() - t0)}
        rows.append(row)
        print(f"  hue={hue} sat={sat} val={val}: D3={r3['tp']:4d} D5={r5['tp']:4d} "
              f"FP={r5['true_fp']:4d} f1_5={r5['corrected_f1']:.1f} "
              f"f1_3={r3['corrected_f1']:.1f}  ({row['sec']}s)", flush=True)
    rows.sort(key=lambda r: -r["f1_5"])
    print("\n-- best by f1_5 --")
    for r in rows[:10]:
        print(f"   hue={r['hue']} sat={r['sat']} val={r['val']}: tp5={r['tp5']} "
              f"fp5={r['fp5']} f1_5={r['f1_5']} f1_3={r['f1_3']}")
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()