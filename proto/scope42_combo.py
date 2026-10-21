"""LOOP-42: combine the LOOP-40 marginal winners and re-check the GATE.

The LOOP-40 sweep on the TRUE GT found no single axis that strictly wins at
D=5, but three axes were marginal:

  sat_min  80 -> 100   592/67.9 -> 594/68.1
  min_side 15 -> 12    592/67.9 -> 594/68.0
  val_min 110 -> 140   592/67.9 -> 593/68.0
  hue_max   5 -> 4     D=3 547 -> 555 (D=5 tie)

This script evaluates the combinations to see whether the marginals stack
into a real win (GATE: strict improvement in corrected_f1 at D=5).

Usage: python proto/scope42_combo.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import load_expected
from canny_quad import get_ids
from scope40_retune import BASE, run_config, Cache, fmt

ROOT = r"D:\src\CodeArts\BRobbDetector"
DIST_VALUES = [3, 5, 8, 12]

COMBOS = [
    ("BASE", {}),
    ("sat=100", {"sat_min": 100}),
    ("min_side=12", {"min_side": 12.0}),
    ("val=140", {"val_min": 140}),
    ("hue=4", {"hue_max": 4}),
    ("sat=100+minside=12", {"sat_min": 100, "min_side": 12.0}),
    ("sat=100+minside=12+val=140", {"sat_min": 100, "min_side": 12.0, "val_min": 140}),
    ("sat=100+minside=12+val=140+hue=4",
     {"sat_min": 100, "min_side": 12.0, "val_min": 140, "hue_max": 4}),
    ("sat=100+minside=12+val=140+hue=4+ma=500",
     {"sat_min": 100, "min_side": 12.0, "val_min": 140, "hue_max": 4, "min_area": 500.0}),
    ("sat=90+minside=12+hue=4",
     {"sat_min": 90, "min_side": 12.0, "hue_max": 4}),
    ("ALL-minimal", {"sat_min": 100, "min_side": 12.0, "val_min": 140,
                     "hue_max": 4, "min_area": 500.0, "ortho_tol": 10.0}),
    # round-2 marginals: fit_band 6->8 (+4 D3 TP), min_area 500->300 (+1 TP)
    ("ALL+fb8", {"sat_min": 100, "min_side": 12.0, "val_min": 140,
                 "hue_max": 4, "min_area": 500.0, "fit_band": 8.0}),
    ("ALL+fb8+ma300", {"sat_min": 100, "min_side": 12.0, "val_min": 140,
                       "hue_max": 4, "min_area": 300.0, "fit_band": 8.0}),
    ("ALL+fb8+ma300+ma8", {"sat_min": 100, "min_side": 12.0, "val_min": 140,
                           "hue_max": 4, "min_area": 300.0, "fit_band": 8.0,
                           "max_aspect": 8.0}),
    ("ALL+fb8+ma300+ortho8", {"sat_min": 100, "min_side": 12.0, "val_min": 140,
                              "hue_max": 4, "min_area": 300.0, "fit_band": 8.0,
                              "ortho_tol": 8.0}),
    ("ALL+fb8+ma300+fb10", {"sat_min": 100, "min_side": 12.0, "val_min": 140,
                            "hue_max": 4, "min_area": 300.0, "fit_band": 10.0}),
]


def main():
    ids = get_ids()
    exp = load_expected()
    cache = Cache()
    results = {}
    print(f"LOOP-42 combination check on the TRUE GT, {len(ids)} images")
    for name, over in COMBOS:
        p = dict(BASE)
        p.update(over)
        t0 = time.time()
        ndet, per_D = run_config(ids, p, cache, exp)
        results[name] = {"params": over, "det": ndet, "per_D": per_D}
        print(fmt(name, ndet, per_D) + f"  ({time.time()-t0:.0f}s)")

    path = os.path.join(ROOT, "benchmark", "scope42_combo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "combos": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()