"""LOOP-49 Phase 0: classify the 124 overlapping-GT FN that LOOP-46 excluded.

Loads the frozen LOOP-46 census rows with `excluded_reason == "overlapping"`
(the 124 overlapping D=5 FN), re-instruments each image with the shipped pm_v10
pipeline, and assigns each FN to the stage that killed its second-box candidate:

  no_proposal / gate_side / gate_ortho / gate_support / dedup / corner_gt5

Also records the GT-overlap cluster size, min/mean side support, and whether a
second-box candidate was ever proposed (matched at D=5 or D=12).

No lever is run here. The stop object implements the LOOP-49 rules:
  - largest bucket < 20             -> STOP
  - no proposal fraction > 0.80    -> STOP (overlap class definitional)

Usage: python proto/loop49_overlap_census.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, iou_obb
from canny_quad import denoise
from scope45_holes import P
from sweep_real import build_data
import loop46_fn_census as CEN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "benchmark", "loop46_fn_census.json")
OUT = os.path.join(ROOT, "benchmark", "loop49_overlap_census.json")


def cluster_of(gt_all, gj):
    n = len(gt_all)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if iou_obb(gt_all[i], gt_all[j]) > 0.01:
                union(i, j)
    r = find(gj)
    members = [k for k in range(n) if find(k) == r]
    max_iou = 0.0
    for i in members:
        for j in members:
            if i != j:
                max_iou = max(max_iou, iou_obb(gt_all[i], gt_all[j]))
    return len(members), max_iou


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    overlap_rows = [r for r in src["rows"]
                    if r.get("excluded_reason") == "overlapping"]
    print("overlapping FN rows:", len(overlap_rows), flush=True)
    ids, gt_map, exp, _ = build_data(0, 500)
    by_iid = {}
    for r in overlap_rows:
        by_iid.setdefault(r["iid"], []).append(r)
    rows = []
    buckets = {k: 0 for k in ("no_proposal", "gate_side", "gate_ortho",
                              "gate_support", "dedup", "corner_gt5")}
    d5 = d12 = 0
    for k, iid in enumerate(sorted(by_iid)):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        raw = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(raw, P["denoise_min_area"]) if P["denoise_min_area"] > 0 else raw
        events, final = CEN.instrument(img, mask)
        gt_all = [np.asarray(g, np.int32) for g in gt_map[iid]]
        for r in by_iid[iid]:
            gj = int(r["gt_index"])
            gb = np.asarray(r["gt"], np.int32)
            bucket, best_d, best = CEN.classify(gb, events, gt_all, gj, mask)
            csize, cmax_iou = cluster_of(gt_all, gj)
            prop5 = bool(best is not None and best_d <= 5.0)
            prop12 = bool(best is not None and best_d <= 12.0)
            d5 += int(prop5)
            d12 += int(prop12)
            buckets[bucket] += 1
            rows.append({
                "iid": iid, "gt_index": gj, "gt": gb.tolist(),
                "cluster_size": csize,
                "cluster_max_iou": round(float(cmax_iou), 4),
                "min_side_support": r["min_side_support"],
                "mean_side_support": r["mean_side_support"],
                "max_iou_with_other_gt": r["max_iou_with_other_gt"],
                "bucket": bucket,
                "second_box_proposed_d5": prop5,
                "second_box_proposed_d12": prop12,
                "best_candidate_distance": (None if best is None
                                            else round(float(best_d), 4)),
                "best_candidate_family": (None if best is None else best["family"]),
                "best_candidate_stage": (None if best is None else best["stage"]),
                "best_candidate_reason": (None if best is None else best["reason"]),
            })
        if k and k % 50 == 0:
            print("  %d/%d" % (k, len(by_iid)), flush=True)
    largest = max(buckets.values())
    n = len(rows)
    no_prop_frac = buckets["no_proposal"] / max(n, 1)
    stop = {
        "triggered": (largest < 20) or (no_prop_frac > 0.80),
        "largest_bucket": largest,
        "rule_small_bucket": "fewer than 20 overlapping FN in every bucket",
        "no_proposal_fraction": round(no_prop_frac, 4),
        "rule_no_proposal": "no second-box candidate for >80% of the overlap class",
        "action": ("STOP; no lever may be run" if (largest < 20 or no_prop_frac > 0.80)
                   else "proceed to the dominant bucket"),
    }
    out = {
        "loop": "LOOP-49", "phase": "overlapping-FN census", "seed": 0,
        "anchor": {"ngt": 1076, "detections": 1070,
                   "tp": {"3": 857, "5": 901, "8": 926, "12": 938},
                   "source": "benchmark/pm_v10_refine.json"},
        "source_rows": len(overlap_rows),
        "n_overlap_fn": n,
        "n_images": len(by_iid),
        "buckets": buckets,
        "bucket_sum": sum(buckets.values()),
        "second_box": {"proposed_d5": d5, "proposed_d12": d12,
                       "no_proposal_fraction": round(no_prop_frac, 4)},
        "stop": stop,
        "rows": rows,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("buckets:", buckets, flush=True)
    print("second box proposed D5=%d D12=%d  no_proposal_frac=%.3f"
          % (d5, d12, no_prop_frac), flush=True)
    print("STOP:", stop, flush=True)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()