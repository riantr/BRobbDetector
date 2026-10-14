"""5-axis orthogonal CV scope — adds exactly ONE new axis on top of pm_v4.

Baseline (pm_v4 winner): hybrid(hull -> edgefit -> mar) = hy_chm.
Axes A (mask=raw), B (proposal=outer CC), C (corners=hull/edgefit/mar) and
D (no refinement) are all FROZEN at the pm_v4 config. The one new axis E varies
ONLY the detection scale:

  E = scale : s1 / s05 / s2   (+ ms union, priority s1 -> s05 -> s2)
    s1  : 1x   (loaded from frozen pm_v4 det dirs -> exact anchor)
    s05 : image resized x0.5 (INTER_AREA), detect, corners scaled back x2
    s2  : image resized x2.0 (INTER_CUBIC), detect, corners scaled back x0.5

Pixel-unit params (areas x s^2; lengths x s) are scaled with the image so the
axis changes only the sampling scale, not the algorithm.

Hybrid re-built per scale: hy = hull -> edgefit -> mar (dedup IoU<0.30).
ms = priority union of [s1, s05, s2] (precise 1x first).

Usage:
  python proto/cv_scope5.py
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, DEFAULT as CQ_DEFAULT
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                       FIT_DEFAULT, evaluate, ROOT, DIST_VALUES)
from eval_pointmatch import load_boxes
from cv_scope3 import SC3_DEFAULT
from cv_scope4 import _base_quads, NONE_REUSE, HY_ORDER

SCALES = {"s05": 0.5, "s1": 1.0, "s2": 2.0}
UNIONS = {"ms": ["s1", "s05", "s2"]}
FROZEN_HY = r"D:\src\CodeArts\BRobbDetector\tmp_sc3_hy_chm"


def scale_params(p, s):
    """Scale pixel-unit params by image scale s (areas x s^2, lengths x s)."""
    q = dict(p)
    q["denoise_min_area"] = p["denoise_min_area"] * s * s
    q["min_area"] = p["min_area"] * s * s
    q["min_side"] = p["min_side"] * s
    q["max_side"] = p["max_side"] * s
    q["boundary_band"] = p["boundary_band"] * s
    q["fit_band"] = p["fit_band"] * s
    q["fit_jump"] = p["fit_jump"] * s
    q["fit_min_pts"] = max(6, int(round(p["fit_min_pts"] * s)))
    return q


def detect_at_scale(img, s, p):
    """hull/edgefit/mar quads detected at scale s, returned in original coords."""
    H, W = img.shape[:2]
    if s == 1.0:
        im, ps = img, p
    else:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        im = cv2.resize(img, (int(round(W * s)), int(round(H * s))),
                        interpolation=interp)
        ps = scale_params(p, s)
    Hs, Ws = im.shape[:2]
    mask = _prep(im, ps)
    cnts = outer_contours(mask, ps)
    out = {m: [] for m in HY_ORDER}
    for c in cnts:
        base = _base_quads(c, ps)
        for m in HY_ORDER:
            qq = _pass(base[m], Ws, Hs, ps)
            if qq is not None:
                out[m].append(np.round(qq.astype(np.float64) / s).astype(np.int32))
    for m in HY_ORDER:
        out[m] = dedup_keep_order(out[m], p["dedup_iou"])
    return out


def run_scales(ids, p):
    """variant -> method -> {iid: [quads]} for scales != 1x."""
    det = {v: {m: {} for m in HY_ORDER} for v, s in SCALES.items() if s != 1.0}
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        for v, s in SCALES.items():
            if s == 1.0:
                continue
            if img is None:
                for m in HY_ORDER:
                    det[v][m][iid] = []
                continue
            q = detect_at_scale(img, s, p)
            for m in HY_ORDER:
                det[v][m][iid] = q[m]
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] multi-scale")
    return det


def build_hybrid(det_v, ids, p):
    return {iid: dedup_keep_order(
        [q for m in HY_ORDER for q in det_v[m].get(iid, [])], p["dedup_iou"])
        for iid in ids}


def write_dir(name, hy, ids):
    out_dir = os.path.join(ROOT, f"tmp_sc5_{name}")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for iid in ids:

        quads = hy[iid]
        total += len(quads)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in quads]), f,
                      ensure_ascii=False)
    print(f"  {name}: {total} boxes -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    p.update(SC3_DEFAULT)

    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"5-axis CV scope (new axis E: scale) on {len(ids)} images, "
          f"{total_gt} GT boxes, D={DIST_VALUES}")

    # s1 anchor: frozen pm_v4 per-method dirs
    var_det = {"s1": {m: {iid: [q.astype(np.int32) for q in load_boxes(d, iid)]
                          for iid in ids}
                      for m, d in NONE_REUSE.items()}}
    var_det.update(run_scales(ids, p))

    hy = {v: build_hybrid(var_det[v], ids, p) for v in SCALES}
    # ms union over scale hybrids (precise s1 first)
    hy_ms = {iid: dedup_keep_order(
        [q for v in UNIONS["ms"] for q in hy[v].get(iid, [])], p["dedup_iou"])
        for iid in ids}

    for v, s in SCALES.items():
        if s != 1.0:
            write_dir(f"hy_{v}", hy[v], ids)
    write_dir("hy_s1", hy["s1"], ids)
    write_dir("ms", hy_ms, ids)

    frozen = sum(len(hy["s1"][iid]) for iid in ids)
    frozen_ref = sum(len(load_boxes(FROZEN_HY, iid)) for iid in ids)
    print(f"  cross-check: s1-hybrid={frozen} vs tmp_sc3_hy_chm={frozen_ref}")

    results = {}
    for v in SCALES:
        results[v] = {}
        for m in HY_ORDER + ["hybrid"]:
            dm = hy[v] if m == "hybrid" else var_det[v][m]
            ndet = sum(len(x) for x in dm.values())
            row = {}
            for D in DIST_VALUES:
                tp, fp = evaluate(ids, gt_map, dm, D)
                row[str(D)] = {"tp": tp, "raw_fp": fp}
            results[v][m] = {"det": ndet, "per_D": row}
    results["ms"] = {}
    for m in HY_ORDER + ["hybrid"]:
        if m == "hybrid":
            dm = hy_ms
        else:
            dm = {iid: dedup_keep_order(
                [q for v in UNIONS["ms"] for q in var_det[v][m].get(iid, [])],
                p["dedup_iou"]) for iid in ids}
        ndet = sum(len(x) for x in dm.values())
        row = {}
        for D in DIST_VALUES:
            tp, fp = evaluate(ids, gt_map, dm, D)
            row[str(D)] = {"tp": tp, "raw_fp": fp}
        results["ms"][m] = {"det": ndet, "per_D": row}

    print(f"\n{'variant':6s} {'method':9s} {'det':>5s}  " +
          "  ".join(f"D={d:<14d}" for d in DIST_VALUES))
    for v in list(SCALES) + ["ms"]:
        for m in HY_ORDER + ["hybrid"]:
            r = results[v][m]
            print(f"{v:6s} {m:9s} {r['det']:5d}  " + "  ".join(
                f"TP={r['per_D'][str(d)]['tp']:4d} FP={r['per_D'][str(d)]['raw_fp']:4d}"
                for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_scope5.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(ids), "total_gt": total_gt,
            "dist_values": DIST_VALUES,
            "new_axis": "E: scale in {s05,s1,s2} + ms union",
            "scales": SCALES, "unions": UNIONS,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()