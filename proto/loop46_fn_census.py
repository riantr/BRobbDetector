"""LOOP-46 Phase 0: per-FN census.

For every D=5 FN in the real 500-image benchmark, classify where the box died:

  no_proposal  : no pre-gate candidate ever matched the GT at D=5
  gate_side    : a candidate matched at D=5 but failed side_ok
  gate_ortho   : a candidate matched at D=5 but failed the orthogonality/edge rule
  gate_support : a candidate matched at D=5 but failed min_red_support
  dedup        : a candidate matched at D=5 but was removed by dedup
  corner_gt5   : a candidate matched only at D=8 or D=12 (near miss)

Restricted to the realistic headroom: min-side red support >= 0.6 and no
overlap with another GT box. The bucket totals are the routing input for the
LOOP-46 lever choice; no lever is run by this script.

Usage: python proto/loop46_fn_census.py
"""
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, iou_obb
from canny_quad import _side_ok
from cv_hybrid import outer_contours, _edgefit_corners
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of,
                           _hole_side_lines, _quad_of, _refine_sides, _enforce,
                           _on_edge, _red_support, detect)
from quad_ortho import is_orthogonal_quad
from eval_pointmatch import corner_match_distance
from sweep_real import build_data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmark", "loop46_fn_census.json")
D = 5
WELL_SUP = 0.6


def side_sups(mask, quad, step=2):
    q = np.asarray(quad, float)
    H, W = mask.shape
    out = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        n = max(int(L / step), 2)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        out.append(float(np.count_nonzero(mask[ys, xs] > 0)) / n)
    return out


def instrument(img, mask):
    """Run the shipped families step-by-step and record every candidate's fate."""
    H, W = img.shape[:2]
    events = []
    # ---- hole family (shipped: group mode off, refine on, support 0.5)
    if PIPE_DEFAULT["strategy"] in ("holes", "both"):
        cfg = HOLE_DEFAULT
        for h in _holes_of(mask, cfg):
            r = _hole_side_lines(h, mask, cfg)
            if r is None:
                events.append({"q": None, "family": "hole", "stage": "quad",
                               "reason": "side_lines_none"})
                continue
            q = _quad_of(h, mask, cfg)
            if q is None:
                events.append({"q": None, "family": "hole", "stage": "quad",
                               "reason": "quad_none"})
                continue
            if cfg.get("hole_refine", 0):
                rr = _refine_sides(mask, q, band=cfg.get("hole_refine_band", 4.0),
                                   min_pts=int(cfg.get("hole_refine_pts", 8)),
                                   tol=cfg.get("hole_refine_tol", 2.0))
                if rr is not None:
                    q = rr
            q = np.asarray(q, float)
            if not _side_ok(q, P):
                events.append({"q": q, "family": "hole", "stage": "side",
                               "reason": "side_ok"})
                continue
            if not is_orthogonal_quad(q, W, H, tol=P["ortho_tol"],
                                      edge_band=P.get("edge_band", 0.0),
                                      waive_boundary=False):
                events.append({"q": q, "family": "hole", "stage": "ortho",
                               "reason": "ortho_or_edge"})
                continue
            q2 = _enforce(q, PIPE_DEFAULT["ortho"])
            if _on_edge(q2, W, H, P):
                events.append({"q": q2, "family": "hole", "stage": "ortho",
                               "reason": "on_edge"})
                continue
            if cfg.get("min_red_support", 0.0) > 0 and \
                    _red_support(mask, q2) < cfg["min_red_support"]:
                events.append({"q": q2, "family": "hole", "stage": "support",
                               "reason": "min_red_support"})
                continue
            events.append({"q": np.asarray(q2, np.int32), "family": "hole",
                           "stage": "pass", "reason": ""})
    # ---- edgefit family
    if PIPE_DEFAULT["strategy"] in ("edgefit", "both"):
        for c in outer_contours(mask, P):
            q = _edgefit_corners(c, P)
            if q is None:
                events.append({"q": None, "family": "edgefit", "stage": "quad",
                               "reason": "corners_none"})
                continue
            q = np.asarray(q, float)
            if not _side_ok(q, P):
                events.append({"q": q, "family": "edgefit", "stage": "side",
                               "reason": "side_ok"})
                continue
            if not is_orthogonal_quad(q, W, H, tol=P["ortho_tol"],
                                      edge_band=P.get("edge_band", 0.0),
                                      waive_boundary=False):
                events.append({"q": q, "family": "edgefit", "stage": "ortho",
                               "reason": "ortho_or_edge"})
                continue
            q2 = _enforce(q, PIPE_DEFAULT["ortho"])
            if _on_edge(q2, W, H, P):
                events.append({"q": q2, "family": "edgefit", "stage": "ortho",
                               "reason": "on_edge"})
                continue
            events.append({"q": np.asarray(q2, np.int32), "family": "edgefit",
                           "stage": "pass", "reason": ""})
    # ---- dedup (same iterative rule as dedup_keep_order, but index-aware)
    pass_idx = [i for i, e in enumerate(events) if e["stage"] == "pass"]
    keep, removed = [], []
    for i in pass_idx:
        q = events[i]["q"]
        if all(iou_obb(q, events[k]["q"]) < PIPE_DEFAULT["dedup_iou"] for k in keep):
            keep.append(i)
        else:
            removed.append(i)
        events[i]["dedup_kept"] = i in keep
    for i in removed:
        events[i]["stage"] = "dedup"
        events[i]["reason"] = "dedup_iou"
    final = [events[i]["q"] for i in keep]
    return events, final


def match_pairs(det, gt, dist):
    cand = []
    for di in range(len(det)):
        for gj in range(len(gt)):
            d = corner_match_distance(det[di], gt[gj])
            if d <= dist:
                cand.append((d, di, gj))
    cand.sort(key=lambda t: t[0])
    usedd, usedg = set(), set()
    pairs = []
    for d, di, gj in cand:
        if di in usedd or gj in usedg:
            continue
        usedd.add(di); usedg.add(gj); pairs.append((di, gj, d))
    return pairs


def classify(gt_box, events, gt_all, gj_index, mask):
    def dist_of(q):
        return corner_match_distance(q, gt_box) if q is not None else 1e18

    # 1) a pass candidate removed by dedup?
    ded = [e for e in events if e["stage"] == "dedup" and dist_of(e["q"]) <= D]
    if ded:
        best = min(ded, key=lambda e: dist_of(e["q"]))
        return "dedup", dist_of(best["q"]), best
    # 2) a rejected candidate at D=5?
    reject = [e for e in events if e["stage"] in ("side", "ortho", "support")
              and dist_of(e["q"]) <= D]
    if reject:
        best = min(reject, key=lambda e: dist_of(e["q"]))
        stage = best["stage"]
        bucket = {"side": "gate_side", "ortho": "gate_ortho",
                  "support": "gate_support"}[stage]
        return bucket, dist_of(best["q"]), best
    # 3) a near miss at D=8/12?
    near = [e for e in events if e["q"] is not None and dist_of(e["q"]) <= 12.0]
    if near:
        best = min(near, key=lambda e: dist_of(e["q"]))
        return "corner_gt5", dist_of(best["q"]), best
    return "no_proposal", 1e18, None


def main():
    ids, gt_map, exp, _ = build_data(0, 500)
    rows, total_fn, eligible = [], 0, 0
    buckets = {k: 0 for k in ("no_proposal", "gate_side", "gate_ortho",
                              "gate_support", "dedup", "corner_gt5")}
    excluded = {"weak_red": 0, "overlapping": 0}
    diag = {"min_ge_06_nonoverlap": 0, "min_ge_06_any": 0,
            "mean_ge_06_nonoverlap": 0, "mean_ge_06_any": 0,
            "max_iou_le_010": 0}
    for k, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        if P["denoise_min_area"] > 0:
            from canny_quad import denoise
            mask = denoise(mask, P["denoise_min_area"])
        events, final = instrument(img, mask)
        gt = [np.asarray(g, np.int32) for g in gt_map[iid]]
        pairs5 = match_pairs(final, gt, D)
        matched_gt = {gj for _, gj, _ in pairs5}
        for gj, gb in enumerate(gt):
            if gj in matched_gt:
                continue
            total_fn += 1
            sups = side_sups(mask, gb)
            min_sup = min(sups)
            mean_sup = float(np.mean(sups))
            max_iou = max([iou_obb(gb, gt[o]) for o in range(len(gt))
                           if o != gj] + [0.0])
            overlap = max_iou > 0.01
            if min_sup >= WELL_SUP:
                diag["min_ge_06_any"] += 1
            if mean_sup >= WELL_SUP:
                diag["mean_ge_06_any"] += 1
            if max_iou <= 0.10:
                diag["max_iou_le_010"] += 1
            if min_sup >= WELL_SUP and not overlap:
                diag["min_ge_06_nonoverlap"] += 1
            if mean_sup >= WELL_SUP and not overlap:
                diag["mean_ge_06_nonoverlap"] += 1
            base = {
                "iid": iid, "gt_index": gj, "gt": gb.tolist(),
                "min_side_support": round(min_sup, 4),
                "mean_side_support": round(mean_sup, 4),
                "max_iou_with_other_gt": round(float(max_iou), 4),
                "overlapping": bool(overlap),
                "detected_d5": False,
            }
            if overlap:
                excluded["overlapping"] += 1
                rows.append({**base, "eligible": False,
                             "excluded_reason": "overlapping", "bucket": None})
                continue
            if mean_sup < WELL_SUP:
                excluded["weak_red"] += 1
                rows.append({**base, "eligible": False,
                             "excluded_reason": "weak_red", "bucket": None})
                continue
            eligible += 1
            bucket, best_d, best = classify(gb, events, gt, gj, mask)
            buckets[bucket] += 1
            rows.append({
                **base, "eligible": True, "excluded_reason": None,
                "bucket": bucket,
                "best_candidate_distance": (None if best is None
                                            else round(float(best_d), 4)),
                "best_candidate_family": (None if best is None else best["family"]),
                "best_candidate_stage": (None if best is None else best["stage"]),
                "best_candidate_reason": (None if best is None else best["reason"]),
            })
        if k and k % 100 == 0:
            print("  %d/%d  FN=%d eligible=%d" % (k, len(ids), total_fn, eligible),
                  flush=True)
    stop = {
        "triggered": max(buckets.values()) < 20,
        "rule": "Phase 0 finds fewer than 20 eligible FN in every bucket",
        "eligible": eligible,
        "largest_bucket": max(buckets.values()),
        "action": ("STOP; no lever may be run" if max(buckets.values()) < 20
                   else "proceed to the dominant bucket"),
    }
    out = {
        "loop": "LOOP-46", "phase": "FN census", "seed": 0,
        "anchor": {"ngt": 1076, "detections": 1070,
                   "tp": {"3": 857, "5": 901, "8": 926, "12": 938},
                   "source": "benchmark/pm_v10_refine.json"},
        "n_images": len(ids), "n_gt": sum(len(gt_map[i]) for i in ids),
        "n_fn_d5": total_fn,
        "eligibility_rule": "mean-side red support >= 0.6 and no overlap with another GT box (revised by user 2026-11-05 from min-side)",
        "eligibility_metric": "mean_side_support",
        "well_drawn_nonoverlap_fn": eligible,
        "excluded": excluded,
        "buckets": buckets,
        "bucket_sum": sum(buckets.values()),
        "assumed_headroom": 61,
        "headroom_note": ("Revised to the MEAN-side >= 0.6 definition chosen by the user; "
                          "the prior min-side run gave 3 eligible, the mean-side count is %d." % eligible),
        "threshold_sensitivity": diag,
        "stop": stop,
        "rows": rows,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("FN=%d eligible=%d bucket_sum=%d" % (total_fn, eligible, sum(buckets.values())))
    print("buckets:", buckets)
    print("excluded:", excluded)
    print("threshold_sensitivity:", diag)
    print("STOP:", stop)
    print("saved", OUT)


if __name__ == "__main__":
    main()