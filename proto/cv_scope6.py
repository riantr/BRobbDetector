"""6-axis orthogonal CV scope — adds exactly ONE new axis on top of pm_v4.

Baseline (pm_v4 winner): hybrid(hull -> edgefit -> mar) = hy_chm at 1x, no refinement.
Axes A/B/C/D/E frozen. The one new axis F varies ONLY the detection-stage admission
filter applied in `_pass` (side_ok + is_orthogonal_quad):

  F = filter : base / ortho10 / ortho35 / bnd12 / minside8 / aspect12 / minarea500 / denoise20

Each option changes exactly ONE filter lever; all other params stay at pm_v4.
min_area / denoise_min_area change the contour set (group key), the rest act after
contour extraction (base quads reused, only `_pass` re-run).

Hybrid re-built per option: hy = hull -> edgefit -> mar (dedup IoU<0.30).

Usage:
  python proto/cv_scope6.py
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, load_expected
from canny_quad import get_ids, to_json, DEFAULT as CQ_DEFAULT
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                       FIT_DEFAULT, evaluate, ROOT, DIST_VALUES)
from eval_pointmatch import load_boxes
from cv_scope3 import SC3_DEFAULT
from cv_scope4 import _base_quads, NONE_REUSE, HY_ORDER

FILTERS = {
    "base":         {},
    "ortho10":      {"ortho_tol": 10.0},
    "ortho15":      {"ortho_tol": 15.0},
    "ortho17":      {"ortho_tol": 17.0},
    "ortho35":      {"ortho_tol": 35.0},
    "bnd12":        {"boundary_band": 12.0},
    "bnd16":        {"boundary_band": 16.0},
    "aspect12":     {"max_aspect": 12.0},
    "aspect16":     {"max_aspect": 16.0},
    "bnd_aspect":   {"boundary_band": 12.0, "max_aspect": 12.0},
    "ba16":         {"boundary_band": 16.0, "max_aspect": 16.0},
    "o15_ba":       {"ortho_tol": 15.0, "boundary_band": 12.0, "max_aspect": 12.0},
    "o16_ba":       {"ortho_tol": 16.0, "boundary_band": 12.0, "max_aspect": 12.0},
    "o17_ba":       {"ortho_tol": 17.0, "boundary_band": 12.0, "max_aspect": 12.0},
    "o18_ba":       {"ortho_tol": 18.0, "boundary_band": 12.0, "max_aspect": 12.0},
    "o17_b10":      {"ortho_tol": 17.0, "boundary_band": 10.0, "max_aspect": 12.0},
    "o17_b14":      {"ortho_tol": 17.0, "boundary_band": 14.0, "max_aspect": 12.0},
    "o17_a10":      {"ortho_tol": 17.0, "boundary_band": 12.0, "max_aspect": 10.0},
    "minside8":     {"min_side": 8.0},
    "minarea500":   {"min_area": 500.0},
    "denoise20":    {"denoise_min_area": 20},
}
FROZEN_HY = r"D:\src\CodeArts\BRobbDetector\tmp_sc3_hy_chm"


def groups(p):
    """contour-group key (denoise_min_area, min_area) -> [filter variants]."""
    g = {}
    for v, ov in FILTERS.items():
        key = (ov.get("denoise_min_area", p["denoise_min_area"]),
               ov.get("min_area", p["min_area"]))
        g.setdefault(key, []).append(v)
    return g


def run_filters(ids, p):
    """variant -> method -> {iid: [quads]}; base quads cached per contour group."""
    g = groups(p)
    det = {v: {m: {} for m in HY_ORDER} for v in FILTERS}
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            for v in FILTERS:
                for m in HY_ORDER:
                    det[v][m][iid] = []
            continue
        H, W = img.shape[:2]
        for (dma, ma), variants in g.items():
            pg = dict(p)
            pg["denoise_min_area"], pg["min_area"] = dma, ma
            cnts = outer_contours(_prep(img, pg), pg)
            cache = [_base_quads(c, pg) for c in cnts]
            for v in variants:
                pv = dict(pg)
                pv.update(FILTERS[v])
                res = {m: [] for m in HY_ORDER}
                for base in cache:
                    for m in HY_ORDER:
                        q = _pass(base[m], W, H, pv)
                        if q is not None:
                            res[m].append(q)
                for m in HY_ORDER:
                    det[v][m][iid] = dedup_keep_order(res[m], pv["dedup_iou"])
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] filter sweep")
    return det


def build_hybrid(det_v, ids, p):
    return {iid: dedup_keep_order(
        [q for m in HY_ORDER for q in det_v[m].get(iid, [])], p["dedup_iou"])
        for iid in ids}


def write_dir(name, hy, ids):
    out_dir = os.path.join(ROOT, f"tmp_sc6_hy_{name}")
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
    ap.parse_args()

    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    p.update(SC3_DEFAULT)

    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"6-axis CV scope (new axis F: admission filter) on {len(ids)} images, "
          f"{total_gt} GT boxes, D={DIST_VALUES}")

    det = run_filters(ids, p)
    hy = {v: build_hybrid(det[v], ids, p) for v in FILTERS}

    # override the base anchor with the frozen pm_v4 hybrid for exactness
    hy["base"] = {iid: [q.astype(np.int32) for q in load_boxes(FROZEN_HY, iid)]
                  for iid in ids}
    for v in FILTERS:
        if v != "base":
            write_dir(v, hy[v], ids)
    write_dir("base", hy["base"], ids)

    frozen = sum(len(hy["base"][iid]) for iid in ids)
    frozen_ref = sum(len(load_boxes(FROZEN_HY, iid)) for iid in ids)
    print(f"  cross-check: base-hybrid={frozen} vs tmp_sc3_hy_chm={frozen_ref}")

    results = {}
    for v in FILTERS:
        results[v] = {}
        for m in HY_ORDER + ["hybrid"]:
            dm = hy[v] if m == "hybrid" else det[v][m]
            ndet = sum(len(x) for x in dm.values())
            row = {}
            for D in DIST_VALUES:
                tp, fp = evaluate(ids, gt_map, dm, D)
                row[str(D)] = {"tp": tp, "raw_fp": fp}
            results[v][m] = {"det": ndet, "per_D": row}

    print(f"\n{'variant':11s} {'method':9s} {'det':>5s}  " +
          "  ".join(f"D={d:<14d}" for d in DIST_VALUES))
    for v in FILTERS:
        for m in HY_ORDER + ["hybrid"]:
            r = results[v][m]
            print(f"{v:11s} {m:9s} {r['det']:5d}  " + "  ".join(
                f"TP={r['per_D'][str(d)]['tp']:4d} FP={r['per_D'][str(d)]['raw_fp']:4d}"
                for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_scope6.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(ids), "total_gt": total_gt,
            "dist_values": DIST_VALUES,
            "new_axis": "F: admission filter",
            "filters": FILTERS,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()