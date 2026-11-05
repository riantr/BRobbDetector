"""LOOP-45 new arms: cardinality-triggered admission, 3-of-4 side-support gate,
local ROI adaptive redness.  Each lever is one-at-a-time vs the shipped pm_v10
pipeline on the real benchmark, scored at D=3/5/8/12, and estimated with the
MoonBit DoubleML PLR (sign + p<0.05 acceptance; held-out seeds 1/2 decide).

Usage:
  python proto/loop45_levers.py --seed 0
  python proto/loop45_levers.py --seed 0 --limit 20   # smoke test
  python proto/loop45_levers.py --seeds 0 1 2
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect, detect_holes,
                           detect_edgefit, dedup_keep_order, denoise)
from eval_pointmatch import match_boxes_point
from sweep_real import build_data
import loop45_moonbit_dml as MB

ROOT = MB.ROOT
OUT_PANEL = os.path.join(ROOT, "benchmark", "loop45_panels")
OUT_JSON = os.path.join(ROOT, "benchmark", "loop45_levers.json")
OUT_MD = os.path.join(ROOT, "benchmark", "loop45_levers_verdict.md")
LEVERS = ("card", "side3", "local")


def _xywh(img):
    H, W = img.shape[:2]
    return H, W


def _base(img):
    return [np.asarray(q, np.int32) for q in detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]


def _side_support(mask, quad, step=2):
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
        out.append(float(np.count_nonzero(mask[ys, xs] > 0)) / max(n, 1))
    return out


def arm_card(img, expected_n):
    """If the shipped count is below the CSV prior, re-admit low-support quads."""
    base = _base(img)
    if expected_n is None or len(base) >= expected_n:
        return base, 0
    cfg = dict(HOLE_DEFAULT)
    cfg["min_red_support"] = 0.0
    extra = [np.asarray(q, np.int32) for q in detect(img, P, cfg, PIPE_DEFAULT)]
    merged = dedup_keep_order(base + extra, PIPE_DEFAULT["dedup_iou"])
    return merged, len(merged) - len(base)


def arm_side3(img):
    """Hole family: accept a quad when >=3 of its 4 sides have red support >=0.5."""
    mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        mask = denoise(mask, P["denoise_min_area"])
    cfg = dict(HOLE_DEFAULT)
    cfg["min_red_support"] = 0.0
    holes = [np.asarray(q, np.int32) for q in
             detect_holes(img, P, cfg, PIPE_DEFAULT, mask=mask)]
    kept = []
    for q in holes:
        s = _side_support(mask, q)
        if sum(1 for v in s if v >= 0.5) >= 3:
            kept.append(q)
    edges = [np.asarray(q, np.int32) for q in
             detect_edgefit(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=mask)]
    return dedup_keep_order(kept + edges, PIPE_DEFAULT["dedup_iou"])


def _local_mask(crop):
    b, g, r = cv2.split(crop.astype(np.float32))
    redness = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (redness > thr).astype(np.uint8) * 255


def arm_local(img):
    """Second proposal source: Otsu on redness inside each shipped-box ROI."""
    base = _base(img)
    H, W = img.shape[:2]
    out = list(base)
    for q in base:
        x0 = max(0, int(q[:, 0].min()) - 20)
        x1 = min(W, int(q[:, 0].max()) + 20)
        y0 = max(0, int(q[:, 1].min()) - 20)
        y1 = min(H, int(q[:, 1].max()) + 20)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = img[y0:y1, x0:x1]
        m = _local_mask(crop)
        if P["denoise_min_area"] > 0:
            m = denoise(m, P["denoise_min_area"])
        for qq in detect(crop, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m):
            qq = np.asarray(qq, np.int32).copy()
            qq[:, 0] += x0
            qq[:, 1] += y0
            out.append(qq)
    return dedup_keep_order(out, PIPE_DEFAULT["dedup_iou"])


ARMS = {"shipped": lambda img, exp: _base(img), "card": arm_card,
        "side3": arm_side3, "local": arm_local}


def run_seed(seed, limit=None, workers=1):
    ids, gt_map, exp, _ = build_data(seed, 500)
    if limit:
        ids = ids[:limit]
    print("seed=%d images=%d" % (seed, len(ids)), flush=True)
    metric = {arm: {D: {"tp": 0, "fp": 0} for D in (3, 5, 8, 12)}
              for arm in ("shipped",) + LEVERS}
    panels = {lever: [] for lever in LEVERS}
    os.makedirs(OUT_PANEL, exist_ok=True)
    for k, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        H, W = img.shape[:2]
        m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        red_ratio = float(np.count_nonzero(m)) / max(H * W, 1)
        shipped = _base(img)
        arm_dets = {"shipped": shipped}
        arm_dets["card"], _ = arm_card(img, exp.get(iid))
        arm_dets["side3"] = arm_side3(img)
        arm_dets["local"] = arm_local(img)
        rows = {}
        for arm, dets in arm_dets.items():
            vals = {}
            for D in (3, 5, 8, 12):
                tp, unm = match_boxes_point(dets, gt, float(D))
                metric[arm][D]["tp"] += int(tp)
                metric[arm][D]["fp"] += int(len(unm))
                vals["y_tp%d" % D] = int(tp)
            rows[arm] = vals
        for lever in LEVERS:
            for arm in ("shipped", lever):
                panels[lever].append({
                    "unit": "image", "block": iid, "cell": arm,
                    "t_" + lever: 1.0 if arm == lever else 0.0,
                    **rows[arm],
                    "x_n_gt": len(gt), "x_img_w": W, "x_img_h": H,
                    "x_red_ratio": round(red_ratio, 6),
                })
        if k and k % 100 == 0:
            print("  %d/%d" % (k, len(ids)), flush=True)
    return metric, panels


def estimate(panel_rows, lever):
    import pandas as pd
    df = pd.DataFrame(panel_rows)
    df = df.sort_values(["block", "cell"], kind="stable").reset_index(drop=True)
    blocks = sorted(df["block"].astype(str).unique())
    bmap = {b: i for i, b in enumerate(blocks)}
    df["_cl"] = df["block"].astype(str).map(bmap)
    cols = ["t_" + lever, "y_tp5", "x_n_gt", "x_img_w", "x_img_h", "x_red_ratio",
            "_cl"]
    MB.write_mbt(df[cols + ["block"]], "t_" + lever, "y_tp5",
                 ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio"])
    res, raw = MB.run_mbt()
    return {
        "treatment": "t_" + lever, "outcome": "y_tp5",
        "n_units": int(len(df)), "n_blocks": int(df["_cl"].nunique()),
        "theta": round(res["c"], 6), "se": round(res["se"], 6),
        "ci95": [round(res["lo"], 6), round(res["hi"], 6)], "p": res["p"],
        "learner": "linear_regression", "fold_seed": MB.FOLD_SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    seeds = args.seeds or [args.seed]
    out = {"loop": "LOOP-45", "tool": "riantr/moonbit_doubleML@0.164.62",
           "acceptance": "theta>0 and p<0.05 on seed 0, repeated on seeds 1/2",
           "seeds": {}}
    for seed in seeds:
        t0 = time.time()
        metric, panels = run_seed(seed, args.limit)
        base = metric["shipped"][5]
        print("  shipped base: D5 %d TP / %d rawFP (expect 901/169 at seed 0)"
              % (base["tp"], base["fp"]), flush=True)
        levers = {}
        for lever in LEVERS:
            from collections import Counter
            with open(os.path.join(OUT_PANEL, "seed%d_%s.csv" % (seed, lever)),
                      "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(panels[lever][0]))
                w.writeheader(); w.writerows(panels[lever])
            levers[lever] = {
                "panel": "benchmark/loop45_panels/seed%d_%s.csv" % (seed, lever),
                "metrics": metric["shipped"], "metrics_lever": metric[lever],
                "estimate": estimate(panels[lever], lever),
            }
            e = levers[lever]["estimate"]
            print("  %-6s theta %+0.4f p=%.3g  D5 %d -> %d TP"
                  % (lever, e["theta"], e["p"], metric["shipped"][5]["tp"],
                     metric[lever][5]["tp"]), flush=True)
        out["seeds"][str(seed)] = {
            "n_images": len(panels["card"]) // 2,
            "wall_clock_s": round(time.time() - t0, 1),
            "levers": levers,
        }
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # verdict md
    lines = ["# LOOP-45 — new recall-lever arms (MoonBit DoubleML PLR)", "",
             "Acceptance: theta > 0 and p < 0.05 on seed 0, and the same sign/"
             "significance on seeds 1/2; otherwise NOT-REPLICATED.", ""]
    for seed, sd in out["seeds"].items():
        lines += ["## seed %s (%d images)" % (seed, sd["n_images"]), "",
                  "| lever | theta | p | D5 TP shipped -> lever | D5 rawFP shipped -> lever |",
                  "|---|---|---|---|---|"]
        for lever, ld in sd["levers"].items():
            e = ld["estimate"]; s = ld["metrics"][5]; l = ld["metrics_lever"][5]
            lines.append("| %s | %+0.4f | %.3g | %d -> %d | %d -> %d |"
                         % (lever, e["theta"], e["p"], s["tp"], l["tp"], s["fp"], l["fp"]))
        lines.append("")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print("saved %s and %s" % (OUT_JSON, OUT_MD), flush=True)


if __name__ == "__main__":
    main()