"""LOOP-51: D=3 corner-precision lever — validity-constrained corner snap.

The shipped quad is the base. For each corner, search a +-2 px window on the
redness channel R - max(G,B) for the ridge peak (sub-pixel parabolic), move the
corner, and keep the snapped quad only if it passes the shipped validity gate
(side, orthogonality/edge, red support); otherwise keep the original corner.

One lever, post-pass, no detector change. Primary outcome: per-image TP at D=3
(base 857). D=5 non-inferiority is required (base 901) so the lever cannot buy
D=3 by losing D=5.

Usage: python proto/loop51_corner_snap.py [--limit N]
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
from loop49_levers import gate_quad
import loop45_moonbit_dml as MB

MB.MBT = os.path.join(MB.ROOT, "tmp_loop45_mbt")
OUT_JSON = os.path.join(MB.ROOT, "benchmark", "loop51_corner_snap.json")
OUT_MD = os.path.join(MB.ROOT, "benchmark", "loop51_corner_snap_verdict.md")
OUT_PANEL = os.path.join(MB.ROOT, "benchmark", "loop51_panels")


def redness(img):
    b, g, r = cv2.split(img.astype(np.float32))
    return r - np.maximum(g, b)


def _parabolic(p0, p1, p2):
    den = p0 - 2.0 * p1 + p2
    if abs(den) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (p0 - p2) / den, -0.5, 0.5))


def snap_quad(q, red, mask, W, H):
    q = np.asarray(q, float)
    q2 = q.copy()
    for i in range(4):
        x, y = q[i]
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - 2), min(W - 1, xi + 2)
        y0, y1 = max(0, yi - 2), min(H - 1, yi + 2)
        patch = red[y0:y1 + 1, x0:x1 + 1]
        if patch.size == 0:
            continue
        yy, xx = np.unravel_index(int(np.argmax(patch)), patch.shape)
        if patch[yy, xx] <= 0:
            continue
        dx = _parabolic(patch[yy, xx - 1], patch[yy, xx], patch[yy, xx + 1]) \
            if 0 < xx < patch.shape[1] - 1 else 0.0
        dy = _parabolic(patch[yy - 1, xx], patch[yy, xx], patch[yy + 1, xx]) \
            if 0 < yy < patch.shape[0] - 1 else 0.0
        nx = x0 + xx + dx
        ny = y0 + yy + dy
        if (nx - x) ** 2 + (ny - y) ** 2 <= 4.0:
            q2[i] = [nx, ny]
    qg = gate_quad(q2, mask)
    return (qg if qg is not None else np.asarray(q, np.int32)), qg is not None


def arm_snap(img, mask):
    red = redness(img)
    H, W = mask.shape
    base = [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]
    out = []
    changed = 0
    for q in base:
        qs, ok = snap_quad(q, red, mask, W, H)
        out.append(qs)
        changed += int(ok)
    return out, changed


def run_seed(seed, limit=None):
    ids, gt_map, exp, _ = build_data(seed, 500)
    if limit:
        ids = ids[:limit]
    metric = {a: {D: {"tp": 0, "fp": 0} for D in (3, 5, 8, 12)}
              for a in ("shipped", "snap")}
    rows, maps = [], {"shipped": {}, "snap": {}}
    changed_total = 0
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
        base = [np.asarray(q, np.int32) for q in
                detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]
        snap, changed = arm_snap(img, mask)
        changed_total += changed
        dets = {"shipped": base, "snap": snap}
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
        for a in ("shipped", "snap"):
            rows.append({"unit": "image", "block": iid, "cell": a,
                         "t_snap": 1.0 if a == "snap" else 0.0,
                         **vals[a],
                         "x_n_gt": n_gt, "x_img_w": W, "x_img_h": H,
                         "x_red_ratio": round(red_ratio, 6),
                         "x_ovl": round(ovl, 6)})
        if k and k % 100 == 0:
            print("  %d/%d" % (k, len(ids)), flush=True)
    agg = {a: score(maps[a], ids, gt_map, exp) for a in maps}
    return metric, rows, agg, ids, changed_total


def estimate(rows, outcome):
    df = pd.DataFrame(rows)
    df = df.sort_values(["block", "cell"], kind="stable").reset_index(drop=True)
    blocks = sorted(df["block"].astype(str).unique())
    bmap = {b: i for i, b in enumerate(blocks)}
    df["_cl"] = df["block"].astype(str).map(bmap)
    cols = ["t_snap", outcome, "x_n_gt", "x_img_w", "x_img_h",
            "x_red_ratio", "x_ovl", "_cl", "block"]
    MB.write_mbt(df[cols], "t_snap", outcome,
                 ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio", "x_ovl"])
    res, _ = MB.run_mbt()
    return {"treatment": "t_snap", "outcome": outcome,
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
    out = {"loop": "LOOP-51", "lever": "validity-constrained corner snap",
           "target": "D=3 corner precision (base 857 TP), D=5 non-inferiority (901)",
           "acceptance": "theta>0 and p<0.05 on y_tp3 seed 0, same on seeds 1/2; "
                         "D5 TP not lower; D3 trueFP <= 0.5 per gained TP",
           "seeds": {}}
    seed0_pass = False
    for seed in (0, 1, 2):
        t0 = time.time()
        metric, rows, agg, ids, changed = run_seed(seed, args.limit)
        est3 = estimate(rows, "y_tp3")
        est5 = estimate(rows, "y_tp5")
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_PANEL, "seed%d_snap.csv" % seed), index=False)
        s, l = agg["shipped"], agg["snap"]
        base3, base5 = metric["shipped"][3], metric["shipped"][5]
        d3 = l["3"]["tp"] - s["3"]["tp"]
        d5 = l["5"]["tp"] - s["5"]["tp"]
        d3fp = l["3"]["true_fp"] - s["3"]["true_fp"]
        print("  seed %d: D3 %d -> %d (%+d), trueFP %d -> %d; D5 %d -> %d (%+d); "
              "theta3 %+0.4f p=%.3g; changed %d"
              % (seed, base3["tp"], l["3"]["tp"], d3,
                 s["3"]["true_fp"], l["3"]["true_fp"],
                 base5["tp"], l["5"]["tp"], d5, est3["theta"], est3["p"], changed),
              flush=True)
        res = {"status": "DONE", "n_images": len(ids), "snapped_quads": int(changed),
               "base_d3_tp": s["3"]["tp"], "lever_d3_tp": l["3"]["tp"],
               "delta_d3_tp": d3, "delta_d3_trueFP": d3fp,
               "base_d5_tp": s["5"]["tp"], "lever_d5_tp": l["5"]["tp"],
               "delta_d5_tp": d5,
               "base_d5_rawFP": base5["fp"],
               "lever_d5_rawFP": metric["snap"][5]["fp"],
               "estimate_d3": est3, "estimate_d5": est5,
               "metrics": {a: metric[a] for a in ("shipped", "snap")},
               "corrected": {a: {str(D): {"tp": agg[a][str(D)]["tp"],
                                          "true_fp": agg[a][str(D)]["true_fp"],
                                          "corr_f1": agg[a][str(D)]["corr_f1"]}
                                  for D in (3, 5, 8, 12)} for a in agg},
               "wall_clock_s": round(time.time() - t0, 1)}
        out["seeds"][str(seed)] = res
        if seed == 0:
            seed0_pass = bool(est3["theta"] > 0 and est3["p"] < 0.05 and d3 > 0
                              and d5 >= 0
                              and (d3fp / d3 if d3 > 0 else 1e9) <= 0.5)
            res["seed0_pass"] = seed0_pass
    if not seed0_pass:
        out["verdict"] = "DEAD"
    else:
        s1, s2 = out["seeds"].get("1", {}), out["seeds"].get("2", {})
        ok = all(x.get("status") == "DONE"
                 and x["estimate_d3"]["theta"] > 0
                 and x["estimate_d3"]["p"] < 0.05
                 and x["delta_d5_tp"] >= 0 for x in (s1, s2))
        out["verdict"] = "CONFIRMED" if ok else "NOT-REPLICATED"
    s0 = out["seeds"]["0"]
    out["base_arm"] = {"seed": 0, "D3_tp": s0["base_d3_tp"],
                       "D5_tp": s0["base_d5_tp"],
                       "D5_rawFP": s0["base_d5_rawFP"],
                       "D5_dets": s0["base_d5_tp"] + s0["base_d5_rawFP"],
                       "frozen_anchor": (s0["base_d3_tp"] == 857
                                         and s0["base_d5_tp"] == 901
                                         and s0["base_d5_tp"] + s0["base_d5_rawFP"] == 1070)}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    lines = ["# LOOP-51 — D=3 validity-constrained corner snap", "",
             "Target: D=3 corner precision (base 857 TP); D=5 must not regress (base 901).",
             "One lever: snap each corner to the local `R-max(G,B)` ridge peak within",
             "+-2 px, keep only if the shipped validity gate passes.", "",
             "| seed | D3 TP shipped -> lever | D3 trueFP | D5 TP shipped -> lever | theta3 | p | snapped |",
             "|---|---|---|---|---|---|---|"]
    for seed in ("0", "1", "2"):
        r = out["seeds"].get(seed, {})
        if r.get("status") == "DONE":
            e = r["estimate_d3"]; c = r["corrected"]
            lines.append("| %s | %d -> %d | %d -> %d | %d -> %d | %+0.4f | %.3g | %d |"
                         % (seed, r["base_d3_tp"], r["lever_d3_tp"],
                            c["shipped"]["3"]["true_fp"], c["snap"]["3"]["true_fp"],
                            r["base_d5_tp"], r["lever_d5_tp"],
                            e["theta"], e["p"], r["snapped_quads"]))
        else:
            lines.append("| %s | %s | - | - | - | - | - |"
                         % (seed, r.get("status", "NOT-ATTEMPTED")))
    lines += ["", "**Verdict: %s.**" % out["verdict"], "",
              "Base arm (seed 0): D3 %s / D5 %s TP / %s rawFP / %s dets "
              "(frozen 857 / 901 / 169 / 1070)."
              % (out["base_arm"]["D3_tp"], out["base_arm"]["D5_tp"],
                 out["base_arm"]["D5_rawFP"], out["base_arm"]["D5_dets"]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print("VERDICT:", out["verdict"], flush=True)
    print("saved", OUT_JSON, OUT_MD, flush=True)


if __name__ == "__main__":
    main()