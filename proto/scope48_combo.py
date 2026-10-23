"""LOOP-48: combination check for the hole pipeline (TRUE GT).

Primary objective is RECALL (the user asked to reduce missed detections),
so TP at D=5 leads; corrected_f1 at D=5 is the tie-breaker and must not
regress.  Reported for every D so the precision/recall trade-off is visible.

Usage: python proto/scope48_combo.py
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scope45_holes import HOLE_DEFAULT, PIPE_DEFAULT, load_all, run, DIST_VALUES

ROOT = r"D:\src\CodeArts\BRobbDetector"

COMBOS = [
    ("BASE(b5,d3,f55)", {}),
    ("dedup0.4", {"dedup_iou": 0.40}),
    ("band3", {"hole_band": 3.0}),
    ("band4", {"hole_band": 4.0}),
    ("band3+dedup0.4", {"hole_band": 3.0, "dedup_iou": 0.40}),
    ("band4+dedup0.4", {"hole_band": 4.0, "dedup_iou": 0.40}),
    ("band3+dedup0.4+fill0.65", {"hole_band": 3.0, "dedup_iou": 0.40,
                                 "hole_fill": 0.65}),
    ("band4+dedup0.4+fill0.65", {"hole_band": 4.0, "dedup_iou": 0.40,
                                 "hole_fill": 0.65}),
    ("band3+dedup0.4+minside12", {"hole_band": 3.0, "dedup_iou": 0.40,
                                  "hole_min_side": 12.0}),
    ("band3+dedup0.45", {"hole_band": 3.0, "dedup_iou": 0.45}),
    ("band2.5+dedup0.4", {"hole_band": 2.5, "dedup_iou": 0.40}),
    ("band3+dedup0.4+over0.12", {"hole_band": 3.0, "dedup_iou": 0.40,
                                 "hole_overshoot": 0.12}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scope48_combo.json")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()
    results = {}
    print(f"LOOP-48 combos on {len(ids)} images, "
          f"{sum(len(v) for v in gt_map.values())} GT boxes")
    for name, over in COMBOS:
        cfg = dict(HOLE_DEFAULT)
        pipe = dict(PIPE_DEFAULT)
        for k, v in over.items():
            if k in pipe:
                pipe[k] = v
            else:
                cfg[k] = v
        t0 = time.time()
        ndet, total_gt, per_D, dt = run(ids, gt_map, exp, cfg, pipe)
        results[name] = {"over": over, "det": ndet, "per_D": per_D}
        print(f"  {name:26s} det={ndet:4d}  " +
              "  ".join(f"D{D}:{per_D[str(D)]['tp']:4d}/{per_D[str(D)]['corrected_f1']:5.1f}"
                       for D in DIST_VALUES) + f"  ({dt:.0f}s)")

    best = max(results, key=lambda k: (results[k]["per_D"]["5"]["tp"],
                                        results[k]["per_D"]["5"]["corrected_f1"]))
    print(f"\nbest by TP@D5: {best}  "
          f"TP={results[best]['per_D']['5']['tp']} "
          f"corr_f1={results[best]['per_D']['5']['corrected_f1']}")

    path = os.path.join(ROOT, "benchmark", args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()