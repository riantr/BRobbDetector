"""LOOP-49 Phase 1: dense-cluster second-box proposal (the only eligible lever).

Census: 98/124 overlapping FN are `no_proposal` (79.0%, just under the 80% stop
line); 26 are corner_gt5. The lever is a concave-corner split of an outer red
component when the shipped count is below the CSV expected count.

For each image with `len(base) < expected`:
  - take outer red contours with 5-12 approx vertices and a concave vertex;
  - split the vertices at the concave vertex along the long axis of the
    min-area rectangle;
  - fit a min-area rect to each lobe, gate it (side, orthogonality/edge,
    red support) and add the accepted quads;
  - union with the shipped detections and dedup.

Acceptance: theta>0 and p<0.05 on seed 0, trueFP increase <= 0.5 per gained TP;
if seed 0 passes, repeat on seeds 1/2 with the same sign/significance.

Usage: python proto/loop49_levers.py
"""
import json
import os
import sys
import time

import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, iou_obb
from canny_quad import denoise, _side_ok
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _enforce,
                           _on_edge, _red_support, dedup_keep_order)
from quad_ortho import is_orthogonal_quad
from eval_pointmatch import match_boxes_point
from s1_inward_complete import score
from sweep_real import build_data
import loop45_moonbit_dml as MB

MB.MBT = os.path.join(MB.ROOT, "tmp_loop45_mbt")
OUT_JSON = os.path.join(MB.ROOT, "benchmark", "loop49_levers.json")
OUT_MD = os.path.join(MB.ROOT, "benchmark", "loop49_levers_verdict.md")
OUT_PANEL = os.path.join(MB.ROOT, "benchmark", "loop49_panels")


def gate_quad(q, mask):
    H, W = mask.shape
    q = np.asarray(q, float)
    if not _side_ok(q, P):
        return None
    if not is_orthogonal_quad(q, W, H, tol=P["ortho_tol"],
                              edge_band=P.get("edge_band", 0.0),
                              waive_boundary=False):
        return None
    q2 = _enforce(q, PIPE_DEFAULT["ortho"])
    if _on_edge(q2, W, H, P):
        return None
    if P.get("min_red_support", 0.0) > 0 and \
            _red_support(mask, q2) < P["min_red_support"]:
        return None
    return np.asarray(q2, np.int32)


def split_outer(img, mask, expected):
    base = [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]
    if expected is None or len(base) >= expected:
        return base, 0
    out = list(base)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    added = 0
    H, W = img.shape[:2]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < P.get("min_area", 300) * 1.5:
            continue
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.02 * peri, True).reshape(-1, 2).astype(float)
        if len(ap) < 5 or len(ap) > 12:
            continue
        # most concave vertex, orientation-aware (OpenCV outer contours can be
        # clockwise or counter-clockwise; a concave vertex has the OPPOSITE
        # cross-product sign to the contour orientation).
        orient = cv2.contourArea(c, oriented=True)
        sgn = 1.0 if orient > 0 else -1.0
        best = None
        for i in range(len(ap)):
            a, b, cc = ap[i - 1], ap[i], ap[(i + 1) % len(ap)]
            v1, v2 = b - a, cc - b
            cross = (v1[0] * v2[1] - v1[1] * v2[0]) * sgn
            if cross < 0 and (best is None or cross < best[0]):
                best = (cross, i, b)
        if best is None:
            continue
        _, i, v = best
        # split at the concave vertex along the long axis of the union box
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        theta = np.deg2rad(ang)
        if rw >= rh:
            d = np.array([np.cos(theta), np.sin(theta)])
        else:
            d = np.array([-np.sin(theta), np.cos(theta)])
        proj = (ap - v) @ d
        g1, g2 = ap[proj <= 0], ap[proj > 0]
        if len(g1) < 4 or len(g2) < 4:
            continue
        for g in (g1, g2):
            r = cv2.minAreaRect(g.astype(np.float32))
            if min(r[1]) < P.get("min_side", 12):
                continue
            q = gate_quad(cv2.boxPoints(r), mask)
            if q is not None:
                out.append(q)
                added += 1
    return dedup_keep_order(out, PIPE_DEFAULT["dedup_iou"]), added


def run_seed(seed):
    ids, gt_map, exp, _ = build_data(seed, 500)
    metric = {a: {D: {"tp": 0, "fp": 0} for D in (3, 5, 8, 12)}
              for a in ("shipped", "split")}
    rows, maps = [], {"shipped": {}, "split": {}}
    added_total = 0
    os.makedirs(OUT_PANEL, exist_ok=True)
    for k, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        raw = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(raw, P["denoise_min_area"]) if P["denoise_min_area"] > 0 else raw
        gt = gt_map[iid]
        base = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]
        split, added = split_outer(img, mask, exp.get(iid))
        added_total += added
        dets = {"shipped": base, "split": split}
        vals = {}
        red_ratio = float(np.count_nonzero(raw)) / max(H * W, 1)
        for a, ds in dets.items():
            maps[a][iid] = ds
            v = {}
            for D in (3, 5, 8, 12):
                tp, unm = match_boxes_point(ds, gt, float(D))
                metric[a][D]["tp"] += int(tp)
                metric[a][D]["fp"] += int(len(unm))
                v["y_tp%d" % D] = int(tp)
            vals[a] = v
        for a in ("shipped", "split"):
            rows.append({"unit": "image", "block": iid, "cell": a,
                         "t_split": 1.0 if a == "split" else 0.0,
                         **vals[a], "x_n_gt": len(gt), "x_img_w": W,
                         "x_img_h": H, "x_red_ratio": round(red_ratio, 6)})
        if k and k % 100 == 0:
            print("  %d/%d" % (k, len(ids)), flush=True)
    agg = {a: score(maps[a], ids, gt_map, exp) for a in maps}
    return metric, rows, agg, ids, added_total


def estimate(rows):
    df = pd.DataFrame(rows)
    df = df.sort_values(["block", "cell"], kind="stable").reset_index(drop=True)
    blocks = sorted(df["block"].astype(str).unique())
    bmap = {b: i for i, b in enumerate(blocks)}
    df["_cl"] = df["block"].astype(str).map(bmap)
    cols = ["t_split", "y_tp5", "x_n_gt", "x_img_w", "x_img_h", "x_red_ratio",
            "_cl", "block"]
    MB.write_mbt(df[cols], "t_split", "y_tp5",
                 ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio"])
    res, _ = MB.run_mbt()
    return {"treatment": "t_split", "outcome": "y_tp5",
            "n_units": int(len(df)), "n_blocks": int(df["_cl"].nunique()),
            "theta": round(float(res["c"]), 6),
            "se": round(float(res["se"]), 6),
            "ci95": [round(float(res["lo"]), 6), round(float(res["hi"]), 6)],
            "p": float(res["p"]), "learner": "linear_regression",
            "fold_seed": MB.FOLD_SEED}


def main():
    census = json.load(open(os.path.join(MB.ROOT, "benchmark",
                                        "loop49_overlap_census.json"), encoding="utf-8"))
    if census["buckets"].get("no_proposal", 0) != max(census["buckets"].values()):
        raise SystemExit("dominant bucket is not no_proposal; lever not eligible")
    if census["stop"]["triggered"]:
        raise SystemExit("census stop triggered: %s" % census["stop"])
    out = {"loop": "LOOP-49", "lever": "dense-cluster second-box proposal",
           "name": "concave-corner split of the outer red component",
           "target_bucket": "no_proposal",
           "census": "benchmark/loop49_overlap_census.json",
           "census_buckets": census["buckets"],
           "acceptance": "theta>0 and p<0.05 on seed 0; trueFP <= 0.5 per TP; "
                         "seeds 1/2 same sign/significance",
           "seeds": {}}
    seed0_pass = False
    for seed in (0, 1, 2):
        t0 = time.time()
        metric, rows, agg, ids, added = run_seed(seed)
        est = estimate(rows)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_PANEL, "seed%d_split.csv" % seed), index=False)
        s, l = agg["shipped"], agg["split"]
        base = metric["shipped"][5]
        d_tp = l["5"]["tp"] - s["5"]["tp"]
        d_fp = l["5"]["true_fp"] - s["5"]["true_fp"]
        print("  seed %d base D5 %d TP / %d rawFP | split %d TP / %d rawFP "
              "| theta %+0.4f p=%.3g"
              % (seed, base["tp"], base["fp"], metric["split"][5]["tp"],
                 metric["split"][5]["fp"], est["theta"], est["p"]), flush=True)
        res = {"status": "DONE", "n_images": len(ids),
               "base_d5_tp": s["5"]["tp"], "base_d5_rawFP": base["fp"],
               "base_d5_dets": s["5"]["tp"] + base["fp"],
               "lever_d5_tp": l["5"]["tp"],
               "lever_d5_rawFP": metric["split"][5]["fp"],
               "delta_tp": d_tp, "delta_trueFP": d_fp,
               "trueFP_per_gain": (None if d_tp <= 0 else round(d_fp / d_tp, 3)),
               "estimate": est,
               "added_quads": int(added),
               "metrics": {a: metric[a] for a in ("shipped", "split")},
               "corrected": {a: {D: {"tp": agg[a][str(D)]["tp"],
                                     "true_fp": agg[a][str(D)]["true_fp"],
                                     "corr_f1": agg[a][str(D)]["corr_f1"]}
                                 for D in ("3","5","8","12")} for a in agg},
               "wall_clock_s": round(time.time() - t0, 1)}
        out["seeds"][str(seed)] = res
        if seed == 0:
            seed0_pass = bool(est["theta"] > 0 and est["p"] < 0.05 and d_tp > 0
                              and (d_fp / d_tp if d_tp > 0 else 1e9) <= 0.5)
            res["seed0_pass"] = seed0_pass
    if not seed0_pass:
        out["verdict"] = "DEAD"
    else:
        s1, s2 = out["seeds"].get("1", {}), out["seeds"].get("2", {})
        ok = all(x.get("status") == "DONE" and x["estimate"]["theta"] > 0
                 and x["estimate"]["p"] < 0.05 for x in (s1, s2))
        out["verdict"] = "CONFIRMED" if ok else "NOT-REPLICATED"
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    s0 = out["seeds"]["0"]
    lines = ["# LOOP-49 — dense-cluster second-box proposal",
             "",
             "Census: **98/124 no_proposal** (79.0%), 26 corner_gt5; no second box",
             "proposed at D=5. The only eligible lever is the concave-corner split",
             "of the outer red component (cardinality-triggered).",
             "",
             "| seed | status | theta | p | D5 TP shipped -> lever | D5 trueFP | trueFP/TP |",
             "|---|---|---|---|---|---|---|"]
    for seed in ("0", "1", "2"):
        r = out["seeds"].get(seed, {})
        if r.get("status") == "DONE":
            e = r["estimate"]; c = r["corrected"]
            lines.append("| %s | DONE | %+0.4f | %.3g | %d -> %d | %d -> %d | %s |"
                         % (seed, e["theta"], e["p"], r["base_d5_tp"],
                            r["lever_d5_tp"], c["shipped"]["5"]["true_fp"],
                            c["split"]["5"]["true_fp"], r["trueFP_per_gain"]))
        else:
            lines.append("| %s | %s | - | - | - | - | - |"
                         % (seed, r.get("status", "NOT-ATTEMPTED")))
    lines += ["", "**Verdict: %s.**" % out["verdict"], "",
              "Base arm (seed 0): %s TP / %s rawFP / %s dets (anchor 901/169/1070)."
              % (s0.get("base_d5_tp"), s0.get("base_d5_rawFP"),
                 s0.get("base_d5_dets")), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print("VERDICT:", out["verdict"], flush=True)
    print("saved", OUT_JSON, OUT_MD, flush=True)


if __name__ == "__main__":
    main()