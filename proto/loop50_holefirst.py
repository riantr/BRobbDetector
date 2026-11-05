"""LOOP-50: candidate-clutter lever — hole-first / edgefit-fallback.

All-cause analysis: `S4_quads` -10.2 pp/SD and `S5_edgefit_share` -9.0 pp/SD
(pm_v10, mutually adjusted). The lever keeps the shipped pipeline but uses the
hole family first; edgefit runs ONLY when the hole family emits nothing.

Arms (one-at-a-time, same 500 images):
  shipped = both families (pm_v10)
  holefirst = holes if holes non-empty else edgefit

Outcome: per-image TP at D=5 (+ D=3/8/12); PLR adjusted for n_gt / W / H /
red ratio / GT-overlap. Acceptance: theta>0 and p<0.05 on seed 0, repeated on
seeds 1/2; trueFP increase <= 0.5 per gained TP.

Usage: python proto/loop50_holefirst.py [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, iou_obb
from canny_quad import denoise
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect
from eval_pointmatch import match_boxes_point
from s1_inward_complete import score
from sweep_real import build_data
import loop45_moonbit_dml as MB

MB.MBT = os.path.join(MB.ROOT, "tmp_loop45_mbt")
OUT_JSON = os.path.join(MB.ROOT, "benchmark", "loop50_holefirst.json")
OUT_MD = os.path.join(MB.ROOT, "benchmark", "loop50_holefirst_verdict.md")
OUT_PANEL = os.path.join(MB.ROOT, "benchmark", "loop50_panels")
PIPE_H = dict(PIPE_DEFAULT, strategy="holes")
PIPE_E = dict(PIPE_DEFAULT, strategy="edgefit")


def arm_shipped(img, mask, exp_n):
    return [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]


def arm_holefirst(img, mask, exp_n):
    holes = detect(img, P, HOLE_DEFAULT, PIPE_H, mask=mask)
    if holes:
        return [np.asarray(q, np.int32) for q in holes]
    return [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_E, mask=mask)]


def run_seed(seed, limit=None):
    ids, gt_map, exp, _ = build_data(seed, 500)
    if limit:
        ids = ids[:limit]
    metric = {a: {D: {"tp": 0, "fp": 0} for D in (3, 5, 8, 12)}
              for a in ("shipped", "holefirst")}
    rows, maps = [], {"shipped": {}, "holefirst": {}}
    os.makedirs(OUT_PANEL, exist_ok=True)
    for k, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        raw = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(raw, P["denoise_min_area"]) if P["denoise_min_area"] > 0 else raw
        gt = gt_map[iid]
        n_gt = len(gt)
        ovl = 0.0
        if n_gt:
            ovl = sum(1 for i in range(n_gt)
                      if any(iou_obb(gt[i], gt[j]) > 0.01
                             for j in range(n_gt) if j != i)) / n_gt
        dets = {"shipped": arm_shipped(img, mask, exp.get(iid)),
                "holefirst": arm_holefirst(img, mask, exp.get(iid))}
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
        for a in ("shipped", "holefirst"):
            rows.append({"unit": "image", "block": iid, "cell": a,
                         "t_holefirst": 1.0 if a == "holefirst" else 0.0,
                         **vals[a],
                         "x_n_gt": n_gt, "x_img_w": W, "x_img_h": H,
                         "x_red_ratio": round(red_ratio, 6),
                         "x_ovl": round(ovl, 6)})
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
    cols = ["t_holefirst", "y_tp5", "x_n_gt", "x_img_w", "x_img_h",
            "x_red_ratio", "x_ovl", "_cl", "block"]
    MB.write_mbt(df[cols], "t_holefirst", "y_tp5",
                 ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio", "x_ovl"])
    res, _ = MB.run_mbt()
    return {"treatment": "t_holefirst", "outcome": "y_tp5",
            "n_units": int(len(df)), "n_blocks": int(df["_cl"].nunique()),
            "theta": round(float(res["c"]), 6),
            "se": round(float(res["se"]), 6),
            "ci95": [round(float(res["lo"]), 6), round(float(res["hi"]), 6)],
            "p": float(res["p"]), "learner": "linear_regression",
            "fold_seed": MB.FOLD_SEED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = {"loop": "LOOP-50", "lever": "hole-first / edgefit-fallback",
           "target": "candidate clutter (S4_quads -10.2 pp/SD, S5_edgefit_share -9.0)",
           "acceptance": "theta>0 and p<0.05 on seed 0, same on seeds 1/2; trueFP <= 0.5 per TP",
           "seeds": {}}
    seed0_pass = False
    for seed in (0, 1, 2):
        t0 = time.time()
        metric, rows, agg, ids = run_seed(seed, args.limit)
        est = estimate(rows)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_PANEL, "seed%d_holefirst.csv" % seed), index=False)
        s, l = agg["shipped"], agg["holefirst"]
        base = metric["shipped"][5]
        d_tp = l["5"]["tp"] - s["5"]["tp"]
        d_fp = l["5"]["true_fp"] - s["5"]["true_fp"]
        print("  seed %d base D5 %d TP / %d rawFP | holefirst %d TP / %d rawFP "
              "| theta %+0.4f p=%.3g"
              % (seed, base["tp"], base["fp"], metric["holefirst"][5]["tp"],
                 metric["holefirst"][5]["fp"], est["theta"], est["p"]), flush=True)
        res = {"status": "DONE", "n_images": len(ids),
               "base_d5_tp": s["5"]["tp"], "base_d5_rawFP": base["fp"],
               "base_d5_dets": s["5"]["tp"] + base["fp"],
               "lever_d5_tp": l["5"]["tp"],
               "lever_d5_rawFP": metric["holefirst"][5]["fp"],
               "delta_tp": d_tp, "delta_trueFP": d_fp,
               "trueFP_per_gain": (None if d_tp <= 0 else round(d_fp / d_tp, 3)),
               "estimate": est,
               "metrics": {a: metric[a] for a in ("shipped", "holefirst")},
               "corrected": {a: {str(D): {"tp": agg[a][str(D)]["tp"],
                                          "true_fp": agg[a][str(D)]["true_fp"],
                                          "corr_f1": agg[a][str(D)]["corr_f1"]}
                                  for D in (3, 5, 8, 12)} for a in agg},
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
    base0 = out["seeds"]["0"]
    out["base_arm"] = {"seed": 0, "D5_tp": base0["base_d5_tp"],
                       "D5_rawFP": base0["base_d5_rawFP"],
                       "D5_dets": base0["base_d5_dets"],
                       "frozen_anchor": (base0["base_d5_tp"] == 901
                                         and base0["base_d5_dets"] == 1070)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    lines = ["# LOOP-50 — hole-first / edgefit-fallback", "",
             "All-cause target: `S4_quads` -10.2 pp/SD, `S5_edgefit_share` -9.0.",
             "One lever: holes first; edgefit only when holes emit nothing.", "",
             "| seed | status | theta | p | D5 TP shipped -> holefirst | D5 rawFP | trueFP | trueFP/TP |",
             "|---|---|---|---|---|---|---|---|"]
    for seed in ("0", "1", "2"):
        r = out["seeds"].get(seed, {})
        if r.get("status") == "DONE":
            e = r["estimate"]; c = r["corrected"]
            lines.append("| %s | DONE | %+0.4f | %.3g | %d -> %d | %d -> %d | %d -> %d | %s |"
                         % (seed, e["theta"], e["p"], r["base_d5_tp"],
                            r["lever_d5_tp"], r["base_d5_rawFP"],
                            r["lever_d5_rawFP"],
                            c["shipped"]["5"]["true_fp"],
                            c["holefirst"]["5"]["true_fp"],
                            r["trueFP_per_gain"]))
        else:
            lines.append("| %s | %s | - | - | - | - | - | - |"
                         % (seed, r.get("status", "NOT-ATTEMPTED")))
    lines += ["", "**Verdict: %s.**" % out["verdict"], "",
              "Base arm (seed 0): %s TP / %s rawFP / %s dets (frozen anchor 901/169/1070)."
              % (out["base_arm"]["D5_tp"], out["base_arm"]["D5_rawFP"],
                 out["base_arm"]["D5_dets"]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print("VERDICT:", out["verdict"], flush=True)
    print("saved", OUT_JSON, OUT_MD, flush=True)


if __name__ == "__main__":
    main()