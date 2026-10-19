"""LOOP-11: Global RANSAC 4-line fit on full-image red pixels (not per-contour).

GATE: D=5 TP >= 502 AND corr_f1 >= 71.3 (pm_v5 baseline).

Strategy: Sample 2 pairs of red pixels to define 2 parallel lines, then form
a rectangle by finding the perpendicular closure. Score by red pixel coverage
on the rectangle boundary. Repeat N times per image, keep top-K rectangles
with score >= min_support. Add to pm_v5 hybrid (hull->edgefit->mar) as a 4th
proposal source.

This recovers boxes whose outline is broken into multiple fragments
(per-contour approaches can't recover these because they require a single
connected outer contour).

Variants (each changes ONE dimension):
  11a base:    n_iter=200, parallel_tol=0.15, min_size=20, min_support=0.40
  11b strict:  n_iter=200, parallel_tol=0.10, min_size=25, min_support=0.50
  11c loose:   n_iter=300, parallel_tol=0.20, min_size=15, min_support=0.30

All keep pm_v5 base: hull+edgefit+mar hybrid, o15_ba admission filter.

Usage: python proto/cv_scope11.py
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
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

OUT_ROOT = os.path.join(ROOT, "tmp_sc11")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope11.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]


# ── RANSAC 4-line core ──

def four_points_to_rect(p1, p2, p3, p4, parallel_tol=0.15, min_size=20):
    """Given 4 points, form a rectangle if (p1,p2) and (p3,p4) are on parallel lines.

    The 4 corners are the axis-aligned bounding box in the (long, perp) frame
    defined by the average of the 2 sample directions.

    Returns 4 corners (float32) or None.
    """
    pts = np.array([p1, p2, p3, p4], dtype=np.float32)

    v1 = p2 - p1
    n1 = np.linalg.norm(v1)
    if n1 < 5:
        return None
    d1 = v1 / n1

    v2 = p4 - p3
    n2 = np.linalg.norm(v2)
    if n2 < 5:
        return None
    d2 = v2 / n2

    cross = abs(d1[0] * d2[1] - d1[1] * d2[0])
    if cross > parallel_tol:
        return None

    d_long = (d1 + d2)
    d_norm = np.linalg.norm(d_long)
    if d_norm < 1e-6:
        return None
    d_long /= d_norm

    d_perp = np.array([-d_long[1], d_long[0]], dtype=np.float32)

    centroid = pts.mean(axis=0)
    rel = pts - centroid
    t_along = rel @ d_long
    t_perp = rel @ d_perp

    t_min, t_max = t_along.min(), t_along.max()
    p_min, p_max = t_perp.min(), t_perp.max()

    if (t_max - t_min) < min_size or (p_max - p_min) < min_size:
        return None

    corners_tp = [
        (t_min, p_min), (t_max, p_min), (t_max, p_max), (t_min, p_max),
    ]
    corners = np.array([
        centroid + t * d_long + p * d_perp
        for t, p in corners_tp
    ], dtype=np.float32)
    return corners


def score_rect_boundary(rect, mask):
    """Score rectangle by fraction of boundary pixels that are red."""
    H, W = mask.shape
    total = 0
    red = 0
    for i in range(4):
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        seg_len = float(np.linalg.norm(p2 - p1))
        n_samples = max(int(seg_len), 5)
        xs = np.linspace(p1[0], p2[0], n_samples)
        ys = np.linspace(p1[1], p2[1], n_samples)
        xs = np.clip(xs.astype(int), 0, W - 1)
        ys = np.clip(ys.astype(int), 0, H - 1)
        total += n_samples
        red += int((mask[ys, xs] > 0).sum())
    return red / total if total > 0 else 0.0


def ransac_4line_rects(mask, p, n_iter=200, parallel_tol=0.15, min_size=20,
                        min_support=0.40, max_per_img=20, seed=42):
    """Find rectangles by RANSAC 4-line fit on full red pixel set."""
    H, W = mask.shape
    red_ys, red_xs = np.where(mask > 0)
    if len(red_ys) < 8:
        return []
    red_px = np.stack([red_xs, red_ys], axis=1).astype(np.float32)
    n_red = len(red_px)
    if n_red < 4:
        return []

    rng = np.random.RandomState(seed)
    rects = []

    for _ in range(n_iter):
        idx = rng.choice(n_red, 4, replace=False)
        p1, p2, p3, p4 = red_px[idx]
        rect = four_points_to_rect(p1, p2, p3, p4, parallel_tol, min_size)
        if rect is None:
            continue
        score = score_rect_boundary(rect, mask)
        if score < min_support:
            continue
        rects.append((score, rect))

    if not rects:
        return []

    rects.sort(key=lambda x: -x[0])
    rects = rects[:max_per_img]

    quads = []
    for _, r in rects:
        qi = r.astype(np.int32)
        if cv2.contourArea(qi) < 100:
            continue
        if _pass(r, W, H, p) is not None:
            quads.append(qi)
    return quads


# ── Runner ──

def run_variant(name, ids, p, n_iter, parallel_tol, min_size,
                min_support, max_per_img):
    """Run pm_v5 + RANSAC 4-line as 4th family."""
    out_dir = os.path.join(OUT_ROOT, name)
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
        mask = red_mask(img)
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
        cnts = outer_contours(mask, p)
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base.get(m), W, H, p)
                if q is not None:
                    hy_quads.append(q)
        rq = ransac_4line_rects(mask, p, n_iter=n_iter,
                                 parallel_tol=parallel_tol,
                                 min_size=min_size,
                                 min_support=min_support,
                                 max_per_img=max_per_img,
                                 seed=42 + n)
        hy_quads.extend(rq)
        try:
            hy = dedup_keep_order(hy_quads, p["dedup_iou"])
        except cv2.error as e:
            print(f"  [{name} {iid}] dedup error: {e}; dropping RANSAC rects", flush=True)
            hy = dedup_keep_order(hy_quads[:len(hy_quads) - len(rq)], p["dedup_iou"])
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


def main():
    ids = get_ids()
    p = pm5_params()
    print(f"LOOP-11: Global RANSAC 4-line fit vs pm_v5 (D={D})", flush=True)
    print(f"  ids={len(ids)}  pm5_params: ortho={p['ortho_tol']} bnd={p['boundary_band']} aspect={p['max_aspect']}", flush=True)

    sample_id = ids[0]
    img = cv2.imread(os.path.join(IMG, sample_id + ".jpg"))
    mask = red_mask(img)
    print(f"\nSanity check on {sample_id} (shape {img.shape}):", flush=True)
    print(f"  red pixels: {int((mask > 0).sum())}", flush=True)

    print(f"\nEvaluating pm_v5 baseline at D={D}...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    variants = [
        # name, n_iter, parallel_tol, min_size, min_support, max_per_img
        ("base",   200, 0.15, 20, 0.40, 20),
        ("strict", 200, 0.10, 25, 0.50, 20),
        ("loose",  300, 0.20, 15, 0.30, 30),
    ]

    results = {"pm5": pm5, "methods": {}, "dist_thr": D, "n_images": len(ids)}
    for name, n_iter, ptol, msize, msup, mper in variants:
        print(f"\nRunning {name} (n_iter={n_iter}, parallel_tol={ptol}, min_size={msize}, min_support={msup}, max_per_img={mper})...", flush=True)
        t0 = time.time()
        det_map = run_variant(name, ids, p, n_iter, ptol, msize, msup, mper)
        dt = time.time() - t0
        r = evaluate_det_map(det_map, ids, D)
        r["runtime_s"] = round(dt, 1)
        r["verdict"] = make_verdict(r, pm5)
        results["methods"][name] = r
        print(f"  {name}: TP={r['tp']} corr_f1={r['corrected_f1']} recall={r['recall']} ({dt:.0f}s) [{r['verdict']}]", flush=True)

    winners = [n for n, _, _, _, _, _ in variants if results["methods"][n]["verdict"] == "WIN"]
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
    print(f"{'Method':12s}  {'TP':>5s}  {'rawFP':>6s}  {'trueFP':>7s}  {'recall':>6s}  {'corr_f1':>8s}  Verdict", flush=True)
    print(f"{'pm_v5':12s}  {pm5['tp']:>5d}  {pm5['raw_fp']:>6d}  {pm5['true_fp']:>7d}  {pm5['recall']:>6.1f}  {pm5['corrected_f1']:>8.1f}  baseline", flush=True)
    for name, _, _, _, _, _ in variants:
        r = results["methods"][name]
        print(f"{name:12s}  {r['tp']:>5d}  {r['raw_fp']:>6d}  {r['true_fp']:>7d}  {r['recall']:>6.1f}  {r['corrected_f1']:>8.1f}  {r['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()