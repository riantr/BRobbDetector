"""LOOP-46: one-axis-at-a-time tuning of the hole pipeline (TRUE GT).

GATE (unchanged from LOOP-40/42): strict improvement in corrected_f1 at D=5
wins, a TIE is an exact match, MARGINAL is within 5 TP and 1.0 corrected_f1.

Usage:
  python proto/scope46_tune.py --axis hole_band
  python proto/scope46_tune.py --axis all
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from scope45_holes import (HOLE_DEFAULT, PIPE_DEFAULT, load_all, run, P,
                           DIST_VALUES)

ROOT = r"D:\src\CodeArts\BRobbDetector"

# axis -> list of values (the first one is the current default)
AXES = {
    "strategy":   ["both", "holes", "edgefit"],
    "ortho":      ["ortho_edge", "none", "ortho_blend"],
    "gate_first": [1, 0],
    "hole_band":  [5.0, 3.0, 4.0, 6.0, 7.0, 9.0],
    "hole_fill":  [0.55, 0.40, 0.50, 0.65, 0.75],
    "hole_pts":   [12, 6, 8, 16, 24],
    "hole_min_side": [5.0, 3.0, 8.0, 12.0],
    "hole_min_comp_area": [60.0, 0.0, 30.0, 120.0],
    "hole_overshoot": [0.06, 0.0, 0.12, 0.20],
    "dedup_iou":  [0.30, 0.15, 0.20, 0.40, 0.50],
}

PIPE_AXES = {"strategy", "ortho", "gate_first", "dedup_iou"}


def fmt(name, ndet, per_D):
    r5 = per_D["5"]
    s = f"  {name:22s} det={ndet:4d}  "
    s += "  ".join(f"D{D}:{per_D[str(D)]['tp']:4d}/{per_D[str(D)]['corrected_f1']:5.1f}"
                   for D in DIST_VALUES)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="all")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()
    cfg = dict(HOLE_DEFAULT)
    pipe = dict(PIPE_DEFAULT)
    results = {}

    axes = list(AXES) if args.axis == "all" else [args.axis]
    for axis in axes:
        print(f"\n=== axis: {axis} ===")
        base = None
        for val in AXES[axis]:
            c = dict(cfg)
            pp = dict(pipe)
            if axis in PIPE_AXES:
                pp[axis] = val
            else:
                c[axis] = val
            ndet, total_gt, per_D, dt = run(ids, gt_map, exp, c, pp)
            key = f"{axis}={val}"
            results[key] = {"det": ndet, "per_D": per_D}
            print(fmt(key, ndet, per_D) + f"  ({dt:.0f}s)")
            if base is None:
                base = per_D["5"]["corrected_f1"]
        best = max((k for k in results if k.startswith(axis + "=")),
                   key=lambda k: results[k]["per_D"]["5"]["corrected_f1"])
        print(f"  best: {best}  "
              f"(corr_f1 {results[best]['per_D']['5']['corrected_f1']} "
              f"vs default {base})")

    path = os.path.join(ROOT, "benchmark", "scope46_tune.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()