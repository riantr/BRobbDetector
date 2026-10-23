"""Parameter sweep on the emulation benchmark.

Runs the production pipeline (proto/scope45_holes.py) over the emulation
dataset for a list of configuration variants and scores each of them with the
project point-match metric (D=3/5/8/12).

The emulation GT is COMPLETE, so a detection that matches no GT box is a plain
false positive - no "real unlabeled" heuristic is needed.

Usage:
    python proto/sweep_emulation.py --variants benchmark/sweep_variants.json
    python proto/sweep_emulation.py --variants ... --n 2000
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT
from eval_emulation import (detect_one, load_gt, score_view, DIST_VALUES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def apply_overrides(base, text):
    cfg = dict(base)
    if not text:
        return cfg
    for kv in filter(None, text.split(",")):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            cfg[k] = float(v)
        except ValueError:
            cfg[k] = v
    return cfg


def run_variant(stems, img_dir, cfg, pipe, p, workers, colors=("red",)):
    tasks = [(s, list(colors), p, cfg, pipe, None, False, img_dir) for s in stems]
    det_map = {}
    with Pool(workers) as pool:
        for stem, res in pool.imap_unordered(detect_one, tasks, chunksize=8):
            det_map[stem] = res
    return det_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=os.path.join(ROOT, "data", "Image@emul4"))
    ap.add_argument("--lab-dir", default=os.path.join(ROOT, "data", "Label@emul4"))
    ap.add_argument("--variants", required=True,
                    help="JSON list of {name, cfg, pipe} objects")
    ap.add_argument("--n", type=int, default=0, help="use only the first N images")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--colors", default="red")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark",
                                                  "sweep_emul4.json"))
    ap.add_argument("--baseline", default="base",
                    help="name of the reference variant")
    args = ap.parse_args()

    with open(args.variants, encoding="utf-8") as f:
        variants = json.load(f)
    colors = [c for c in args.colors.split(",") if c]
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in os.listdir(args.lab_dir) if p.endswith(".json"))
    if args.n:
        stems = stems[:args.n]
    gt_map = {s: load_gt(s, args.lab_dir) for s in stems}
    n_gt = sum(len(v) for v in gt_map.values())
    print("emulation: %d images, %d GT boxes" % (len(stems), n_gt), flush=True)

    rows = []
    for v in variants:
        cfg = apply_overrides(HOLE_DEFAULT, v.get("cfg", ""))
        pipe = apply_overrides(PIPE_DEFAULT, v.get("pipe", ""))
        t0 = time.time()
        det_map = run_variant(stems, args.img_dir, cfg, pipe, P, args.workers,
                              colors)
        sc = score_view(det_map, gt_map, stems, colors, colors[0])
        dt = time.time() - t0
        rec = {"name": v.get("name", "?"), "cfg": v.get("cfg", ""),
               "pipe": v.get("pipe", ""), "sec": round(dt, 1),
               "ndet": sc["ndet"], "ngt": sc["ngt"], "per_D": sc["per_D"]}
        rows.append(rec)
        print("%-26s %5.1fs  " % (rec["name"], dt) +
              "  ".join("D%d %4d/%4d r%5.1f p%5.1f" %
                        (D, sc["per_D"][str(D)]["tp"], sc["per_D"][str(D)]["fp"],
                         sc["per_D"][str(D)]["recall"],
                         sc["per_D"][str(D)]["precision"])
                        for D in DIST_VALUES), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("saved %s" % args.out)


if __name__ == "__main__":
    main()