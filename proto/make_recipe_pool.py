"""Build the emulation recipe pool from the measured real degradation.

proto/measure_real_degrade.py measures, for every GT box of the real data,
the red coverage of each side and whether the red is *missing* over a stretch
of that side (band_cover < 1).  The two facts together explain the recall of
the detector on the real data:

  worst side present   -> the box is found (recall 84-99%)
  worst side missing   -> the box is lost  (recall 0-25%)

This script turns those measurements into a pool of per-box recipes that the
generator can replay:  [[coverage, missing, band_cover] x 4] per box.

Usage: python proto/make_recipe_pool.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"D:\src\CodeArts\BRobbDetector"
SRC = os.path.join(ROOT, "benchmark", "real_degrade_full.json")
DST = os.path.join(ROOT, "benchmark", "real_gt_recipe.json")


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    recipes = []
    for r in rows:
        rec = []
        for i in range(4):
            cov = float(r["sup"][i])
            bc = float(r["band"][i][0])
            missing = 1.0 if bc < 0.999 else 0.0
            if missing:
                bc = max(0.05, min(1.0, bc))
            rec.append([round(cov, 4), missing, round(bc, 4)])
        recipes.append(rec)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False)
    n_missing = sum(1 for r in recipes for s in r if s[1] > 0.5)
    print("wrote %s: %d boxes, %d sides, %d of them 'missing' (%.1f%%)"
          % (DST, len(recipes), 4 * len(recipes), n_missing,
             100.0 * n_missing / (4 * len(recipes))))
    mins = [min(s[0] for s in r) for r in recipes]
    hist = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    print("\nmin-side coverage of the pool (the real distribution):")
    for lo, hi in hist:
        n = sum(1 for m in mins if lo <= m < hi)
        print("  [%.1f,%.1f)  %5d  %5.1f%%" % (lo, hi, n, 100.0 * n / len(mins)))
    return 0


if __name__ == "__main__":
    sys.exit(main())