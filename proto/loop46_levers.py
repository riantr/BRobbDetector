"""LOOP-46 lever C: hysteresis red mask (the only eligible lever).

Phase 0 (mean-side support >= 0.6) found 33 eligible FN and the dominant bucket
is `no_proposal` (21). Per the pre-registered routing, ONLY lever C may run.

Hysteresis: the shipped mask is the strong seed; a broader weak mask
(hue<=6, sat>=70, val>=110) is kept only on connected components that touch a
strong seed pixel. This grows faint strokes without letting isolated object
texture in.

Acceptance:
  seed 0: theta > 0 and p < 0.05 and trueFP increase <= 0.5 per gained TP;
  seeds 1/2: same sign and p < 0.05. Otherwise NOT-REPLICATED.

Usage: python proto/loop46_levers.py
"""
import json
import os
import sys
import time

import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import denoise
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect
from eval_pointmatch import match_boxes_point
from s1_inward_complete import score
from sweep_real import build_data
import loop45_moonbit_dml as MB

MB.MBT = os.path.join(MB.ROOT, "tmp_loop46_mbt")
OUT_JSON = os.path.join(MB.ROOT, "benchmark", "loop46_levers.json")
OUT_MD = os.path.join(MB.ROOT, "benchmark", "loop46_levers_verdict.md")
OUT_PANEL = os.path.join(MB.ROOT, "benchmark", "loop46_panels")
SEEDS = (0, 1, 2)
LEVER = "C"
NAME = "hysteresis red mask"
TARGET = "no_proposal"


def hyst_mask(img):
    strong = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    weak = red_mask(img, max(60, P["sat_min"] - 30),
                    max(100, P["val_min"] - 30), min(8, P["hue_max"] + 2))
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.unique(labels[strong > 0])
    keep = keep[keep != 0]
    if keep.size == 0:
        return strong
    out = np.isin(labels, keep).astype(np.uint8) * 255
    if P["denoise_min_area"] > 0:
        out = denoise(out, P["denoise_min_area"])
    return out


def arm_shipped(img):
    return [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]


def arm_hyst(img):
    m = hyst_mask(img)
    return [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)]


def run_seed(seed, limit=None):
    ids, gt_map, exp, _ = build_data(seed, 500)
    if limit:
        ids = ids[:limit]
    print("seed=%d images=%d" % (seed, len(ids)), flush=True)
    metric = {a: {D: {"tp": 0, "fp": 0} for D in (3, 5, 8, 12)}
              for a in ("shipped", "hyst")}
    rows, maps = [], {"shipped": {}, "hyst": {}}
    os.makedirs(OUT_PANEL, exist_ok=True)
    for k, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        H, W = img.shape[:2]
        m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        red_ratio = float(np.count_nonzero(m)) / max(H * W, 1)
        dets = {"shipped": arm_shipped(img), "hyst": arm_hyst(img)}
        vals = {}
        for a, ds in dets.items():
            maps[a][iid] = ds
            v = {}
            for D in (3, 5, 8, 12):
                tp, unm = match_boxes_point(ds, gt, float(D))
                metric[a][D]["tp"] += int(tp)
                metric[a][D]["fp"] += int(len(unm))
                v["y_tp%d" % D] = int(tp)
            vals[a] = v
        for a in ("shipped", "hyst"):
            rows.append({
                "unit": "image", "block": iid, "cell": a,
                "t_hyst": 1.0 if a == "hyst" else 0.0,
                **vals[a],
                "x_n_gt": len(gt), "x_img_w": W, "x_img_h": H,
                "x_red_ratio": round(red_ratio, 6),
            })
        if k and k % 100 == 0:
            print("  %d/%d" % (k, len(ids)), flush=True)
    agg = {a: score(maps[a], ids, gt_map, exp) for a in maps}
    return metric, rows, agg, ids


def estimate(rows):
    df = pd.DataFrame(rows)
    df = df.sort_values(["block", "cell"], kind="stable").reset_index(drop=True)
    blocks = sorted(df["block"].astype(str).unique())
    bmap = {b: i for i, b in enumerate(blocks)}
    df["_cl"] = df["block"].astype(str).map(bmap)
    cols = ["t_hyst", "y_tp5", "x_n_gt", "x_img_w", "x_img_h",
            "x_red_ratio", "_cl", "block"]
    MB.write_mbt(df[cols], "t_hyst", "y_tp5",
                 ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio"])
    res, _ = MB.run_mbt()
    return {
        "treatment": "t_hyst", "outcome": "y_tp5",
        "n_units": int(len(df)), "n_blocks": int(df["_cl"].nunique()),
        "theta": round(res["c"], 6), "se": round(res["se"], 6),
        "ci95": [round(res["lo"], 6), round(res["hi"], 6)], "p": res["p"],
        "learner": "linear_regression", "fold_seed": MB.FOLD_SEED,
    }


def main():
    census = json.load(open(os.path.join(MB.ROOT, "benchmark",
                                        "loop46_fn_census.json"), encoding="utf-8"))
    buckets = census["buckets"]
    dominant = max(buckets, key=buckets.get)
    if dominant != TARGET:
        raise SystemExit("census dominant bucket is %s, not %s; lever C not eligible"
                         % (dominant, TARGET))
    out = {"loop": "LOOP-46", "lever": LEVER, "name": NAME,
           "target_bucket": TARGET, "census": "benchmark/loop46_fn_census.json",
           "n_eligible": census["well_drawn_nonoverlap_fn"],
           "census_buckets": buckets,
           "acceptance": "theta>0 and p<0.05 on seed 0 + same on seeds 1/2; "
                         "trueFP increase <= 0.5 per gained TP",
           "seeds": {}}
    seed0_pass = False
    for si, seed in enumerate(SEEDS):
        if si > 0 and not seed0_pass:
            out["seeds"][str(seed)] = {"status": "NOT-ATTEMPTED",
                                       "reason": "seed 0 did not pass"}
            continue
        t0 = time.time()
        metric, rows, agg, ids = run_seed(seed)
        est = estimate(rows)
        with open(os.path.join(OUT_PANEL, "seed%d_hyst.csv" % seed), "w",
                  newline="", encoding="utf-8") as f:
            w = pd.DataFrame(rows).to_csv(f, index=False)
        s, h = agg["shipped"], agg["hyst"]
        base = metric["shipped"][5]
        print("  base: D5 %d TP / %d rawFP | lever D5 %d TP / %d rawFP"
              % (base["tp"], base["fp"], metric["hyst"][5]["tp"],
                 metric["hyst"][5]["fp"]), flush=True)
        print("  theta %+0.4f p=%.3g  trueFP %d -> %d"
              % (est["theta"], est["p"], s["5"]["true_fp"], h["5"]["true_fp"]),
              flush=True)
        d_tp = h["5"]["tp"] - s["5"]["tp"]
        d_fp = h["5"]["true_fp"] - s["5"]["true_fp"]
        seed_res = {
            "status": "DONE", "n_images": len(ids),
            "base_d5_tp": s["5"]["tp"], "base_d5_rawFP": base["fp"],
            "base_d5_dets": s["5"]["tp"] + base["fp"],
            "lever_d5_tp": h["5"]["tp"], "lever_d5_rawFP": metric["hyst"][5]["fp"],
            "delta_tp": d_tp, "delta_trueFP": d_fp,
            "trueFP_per_gain": (None if d_tp <= 0 else round(d_fp / d_tp, 3)),
            "estimate": est,
            "metrics": {a: metric[a] for a in ("shipped", "hyst")},
            "corrected": {a: {D: {"tp": agg[a][str(D)]["tp"],
                                  "true_fp": agg[a][str(D)]["true_fp"],
                                  "corr_f1": agg[a][str(D)]["corr_f1"]}
                              for D in ("3","5","8","12")} for a in agg},
            "wall_clock_s": round(time.time() - t0, 1),
        }
        out["seeds"][str(seed)] = seed_res
        if seed == 0:
            ok = (est["theta"] > 0 and est["p"] < 0.05 and d_tp > 0
                  and (d_fp / d_tp if d_tp > 0 else 1e9) <= 0.5)
            seed0_pass = bool(ok)
            seed_res["seed0_pass"] = seed0_pass
    # overall verdict
    ss = out["seeds"]
    if not seed0_pass:
        out["verdict"] = "DEAD"
    else:
        s1, s2 = ss.get("1", {}), ss.get("2", {})
        ok1 = (s1.get("status") == "DONE" and s1["estimate"]["theta"] > 0
               and s1["estimate"]["p"] < 0.05)
        ok2 = (s2.get("status") == "DONE" and s2["estimate"]["theta"] > 0
               and s2["estimate"]["p"] < 0.05)
        out["verdict"] = "CONFIRMED" if (ok1 and ok2) else "NOT-REPLICATED"
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # verdict md
    lines = ["# LOOP-46 — lever C (hysteresis red mask)", "",
             "Phase 0 (mean-side >= 0.6) gave **%d eligible FN**; dominant bucket "
             "`%s` = %d. This is the only eligible lever." %
             (census["well_drawn_nonoverlap_fn"], TARGET, buckets[TARGET]), "",
             "| seed | status | theta | p | D5 TP shipped -> lever | D5 trueFP | trueFP/TP |",
             "|---|---|---|---|---|---|---|"]
    for seed in ("0", "1", "2"):
        s = ss.get(seed, {})
        if s.get("status") == "DONE":
            e = s["estimate"]
            lines.append("| %s | DONE | %+0.4f | %.3g | %d -> %d | %d -> %d | %s |"
                         % (seed, e["theta"], e["p"], s["base_d5_tp"],
                            s["lever_d5_tp"], s["corrected"]["shipped"]["5"]["true_fp"],
                            s["corrected"]["hyst"]["5"]["true_fp"],
                            s["trueFP_per_gain"]))
        else:
            lines.append("| %s | %s | - | - | - | - | - |"
                         % (seed, s.get("status", "NOT-ATTEMPTED")))
    lines += ["", "**Verdict: %s.**" % out["verdict"], "",
              "Base arm (seed 0): %s TP / %s rawFP / %s dets "
              "(frozen anchor = 901 / 169 / 1070)." %
              (ss.get("0", {}).get("base_d5_tp"), ss.get("0", {}).get("base_d5_rawFP"),
               ss.get("0", {}).get("base_d5_dets")), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print("VERDICT:", out["verdict"])
    print("saved", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()