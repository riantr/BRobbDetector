"""LOOP-12: Combinations of marginal CV algorithms vs pm_v5.

GATE: D=5 TP >= 502 AND corr_f1 >= 71.3.

Each variant combines 2+ methods that were MARGINAL when tested alone
(LOOP-10 tophat mask, LOOP-11 strict RANSAC), on the hypothesis that
they recover DIFFERENT missed boxes and their union exceeds either alone.

Variants:
  12a tophat+ransac:   LOOP-10 tophat mask + LOOP-11 strict RANSAC 4-line
  12b union+ransac:    Union of (base, tophat, CLAHE) masks + strict RANSAC
  12c tophat+multi:    tophat mask + epq + paq + strict RANSAC (4 sources)
  12d or_of_dets:      pm_v5(base) ∪ pm_v5(tophat) ∪ pm_v5(CLAHE) + RANSAC

All keep pm_v5 base: hull+edgefit+mar hybrid, o15_ba admission filter.
RANSAC uses strict params: parallel_tol=0.10, min_size=25, min_support=0.50.

Usage: python proto/cv_scope12.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import (
    outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES,
)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from cv_scope10 import red_mask_tophat, red_mask_clahe, make_epq, make_paq
from cv_scope11 import ransac_4line_rects
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

OUT_ROOT = os.path.join(ROOT, "tmp_sc12")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope12.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]

# RANSAC strict params (LOOP-11 MARGINAL winner)
RANSAC_PARAMS = {
    "n_iter": 200, "parallel_tol": 0.10, "min_size": 25,
    "min_support": 0.50, "max_per_img": 20,
}


# ── Mask functions ──

def union_mask_3(img, smin=80, vmin=110):
    """Union of base HSV, tophat-modulated HSV, and CLAHE-modulated HSV masks."""
    return cv2.bitwise_or(
        cv2.bitwise_or(red_mask(img), red_mask_tophat(img, smin=smin, vmin=vmin)),
        red_mask_clahe(img, smin=smin, vmin=vmin),
    )


# ── RANSAC family wrapper (matches cv_scope4 family_fn pattern) ──

def make_ransac(mask, p, n_iter=200, parallel_tol=0.10, min_size=25,
                 min_support=0.50, max_per_img=20, seed=42):
    """Generate RANSAC 4-line rectangles (as a proposal list)."""
    H, W = mask.shape
    quads = ransac_4line_rects(
        mask, p, n_iter=n_iter, parallel_tol=parallel_tol,
        min_size=min_size, min_support=min_support,
        max_per_img=max_per_img, seed=seed,
    )
    if not quads:
        return None
    return quads[0] if len(quads) == 1 else quads


# ── Pipeline variants ──

def run_with_mask_and_families(name, ids, p, mask_fn, families, out_dir):
    """Run pm_v5 hybrid with custom mask + extra families.
    families: dict of {family_name: family_fn} to add after HY_ORDER.
    """
    os.makedirs(out_dir, exist_ok=True)
    det = {}
    total = 0
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det[iid] = []
            continue
        H, W = img.shape[:2]
        mask = mask_fn(img)
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
        cnts = outer_contours(mask, p)
        order = HY_ORDER + list(families.keys())
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p)
            for fname, ffn in families.items():
                if ffn is not None:
                    base[fname] = ffn(c)
            for m in order:
                q = _pass(base.get(m), W, H, p)
                if q is not None:
                    hy_quads.append(q)
        rq = ransac_4line_rects(
            mask, p,
            n_iter=RANSAC_PARAMS["n_iter"],
            parallel_tol=RANSAC_PARAMS["parallel_tol"],
            min_size=RANSAC_PARAMS["min_size"],
            min_support=RANSAC_PARAMS["min_support"],
            max_per_img=RANSAC_PARAMS["max_per_img"],
            seed=42 + n,
        )
        hy_quads.extend(rq)
        try:
            hy = dedup_keep_order(hy_quads, p["dedup_iou"])
        except cv2.error as e:
            print(f"  [{name} {iid}] dedup error: {e}", flush=True)
            hy = dedup_keep_order(
                [q for q in hy_quads if q is not None and len(q) == 4],
                p["dedup_iou"],
            )
        det[iid] = hy
        total += len(hy)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in hy]), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  [{name} {n+1}/{len(ids)}] boxes={total} elapsed={dt:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"  {name}: {total} boxes in {dt:.0f}s -> {out_dir}", flush=True)
    return det


def run_or_of_dets(name, ids, p, mask_fns, out_dir):
    """Run pm_v5 on each mask, union all detections, add RANSAC strict."""
    os.makedirs(out_dir, exist_ok=True)
    det = {}
    total = 0
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det[iid] = []
            continue
        H, W = img.shape[:2]
        all_quads = []
        for mask_fn in mask_fns:
            mask = mask_fn(img)
            if p["denoise_min_area"] > 0:
                mask = denoise(mask, p["denoise_min_area"])
            cnts = outer_contours(mask, p)
            for c in cnts:
                base = _base_quads(c, p)
                for m in HY_ORDER:
                    q = _pass(base.get(m), W, H, p)
                    if q is not None:
                        all_quads.append(q)
        rq = ransac_4line_rects(
            mask, p,
            n_iter=RANSAC_PARAMS["n_iter"],
            parallel_tol=RANSAC_PARAMS["parallel_tol"],
            min_size=RANSAC_PARAMS["min_size"],
            min_support=RANSAC_PARAMS["min_support"],
            max_per_img=RANSAC_PARAMS["max_per_img"],
            seed=42 + n,
        )
        all_quads.extend(rq)
        try:
            hy = dedup_keep_order(all_quads, p["dedup_iou"])
        except cv2.error as e:
            print(f"  [{name} {iid}] dedup error: {e}", flush=True)
            hy = dedup_keep_order(
                [q for q in all_quads if q is not None and len(q) == 4],
                p["dedup_iou"],
            )
        det[iid] = hy
        total += len(hy)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in hy]), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  [{name} {n+1}/{len(ids)}] boxes={total} elapsed={dt:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"  {name}: {total} boxes in {dt:.0f}s -> {out_dir}", flush=True)
    return det


# ── Evaluation ──

def evaluate_det_map(det_map, ids, dist_thr=5.0):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)
        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp
        FN += len(gt) - tp
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    return {
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
        "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "corrected_f1": round(f1, 1),
    }


def evaluate_pm5_ref(ids, dist_thr=5.0):
    det_map = {iid: load_boxes(PM5_REF, iid) for iid in ids}
    return evaluate_det_map(det_map, ids, dist_thr)


def make_verdict(r, pm5):
    tp_b, f1_b = pm5["tp"], pm5["corrected_f1"]
    tp_n, f1_n = r["tp"], r["corrected_f1"]
    if tp_n >= tp_b and f1_n >= f1_b:
        return "WIN"
    if tp_n >= tp_b - 5 and f1_n >= f1_b - 1.0:
        return "MARGINAL"
    return "DEAD"


# ── Main ──

def main():
    ids = get_ids()
    p = pm5_params()
    print(f"LOOP-12: Combinations of marginal CV algorithms vs pm_v5 (D={D})", flush=True)
    print(f"  ids={len(ids)}  pm5_params: ortho={p['ortho_tol']} bnd={p['boundary_band']} aspect={p['max_aspect']}", flush=True)

    print(f"\nEvaluating pm_v5 baseline at D={D}...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    variants = [
        # name, mask_fn, families, is_or_of_dets
        ("tophat+ransac", red_mask_tophat, {}, False),
        ("union+ransac",  union_mask_3,   {}, False),
        ("tophat+multi",  red_mask_tophat,
            {"epq": make_epq, "paq": make_paq}, False),
        ("or_of_dets",    None, {}, True),
    ]

    results = {"pm5": pm5, "methods": {}, "dist_thr": D, "n_images": len(ids)}
    for name, mask_fn, families, is_or in variants:
        out_dir = os.path.join(OUT_ROOT, name)
        print(f"\nRunning {name} (mask={mask_fn.__name__ if mask_fn else 'or'}, "
              f"families={list(families.keys())})...", flush=True)
        t0 = time.time()
        if is_or:
            det_map = run_or_of_dets(
                name, ids, p,
                [red_mask, red_mask_tophat, red_mask_clahe],
                out_dir,
            )
        else:
            det_map = run_with_mask_and_families(
                name, ids, p, mask_fn, families, out_dir,
            )
        dt = time.time() - t0
        r = evaluate_det_map(det_map, ids, D)
        r["runtime_s"] = round(dt, 1)
        r["verdict"] = make_verdict(r, pm5)
        results["methods"][name] = r
        print(f"  {name}: TP={r['tp']} corr_f1={r['corrected_f1']} recall={r['recall']} ({dt:.0f}s) [{r['verdict']}]", flush=True)

    winners = [n for n, _, _, _ in variants if results["methods"][n]["verdict"] == "WIN"]
    if winners:
        print(f"\nRe-evaluating WINners at all D values: {winners}", flush=True)
        for name in winners:
            det_map = {iid: load_boxes(os.path.join(OUT_ROOT, name), iid) for iid in ids}
            results["methods"][name]["full_D"] = {}
            for dd in DIST_VALUES_FULL:
                results["methods"][name]["full_D"][f"D{dd}"] = evaluate_det_map(det_map, ids, float(dd))

    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY (D={D}, 500 images, pm_v5 baseline TP={pm5['tp']} corr_f1={pm5['corrected_f1']})", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Method':16s}  {'TP':>5s}  {'rawFP':>6s}  {'trueFP':>7s}  {'recall':>6s}  {'corr_f1':>8s}  Verdict", flush=True)
    print(f"{'pm_v5':16s}  {pm5['tp']:>5d}  {pm5['raw_fp']:>6d}  {pm5['true_fp']:>7d}  {pm5['recall']:>6.1f}  {pm5['corrected_f1']:>8.1f}  baseline", flush=True)
    for name, _, _, _ in variants:
        r = results["methods"][name]
        print(f"{name:16s}  {r['tp']:>5d}  {r['raw_fp']:>6d}  {r['true_fp']:>7d}  {r['recall']:>6.1f}  {r['corrected_f1']:>8.1f}  {r['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()