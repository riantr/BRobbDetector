"""LOOP-10: 6 untried classical-CV methods, each a single-axis change vs pm_v5.

GATE: D=5 TP >= 502 AND corr_f1 >= 71.3 (pm_v5 baseline).

Methods (each changes ONE dimension):
  1. CLAHE pre-mask         (frontend: CLAHE on V, then HSV red mask)
  2. White top-hat pre-mask (frontend: top-hat on gray, modulate V, then HSV)
  3. Excess-red projection  (frontend: R - max(G,B), HSV-gated)
  4. YCrCb Cr projection    (frontend: Cr threshold, HSV-gated)
  5. Extreme-points quad    (4th corner family: 4 farthest points in 4 cardinal dirs)
  6. PCA-aligned extreme quad (4th corner family: 4 farthest points along PCA axes)

All keep pm_v5 base: hull+edgefit+mar hybrid, o15_ba admission filter.
Frontends (1-4) swap the mask. Corner variants (5-6) add a 4th family to the hybrid.

Usage: python proto/cv_scope10.py
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

OUT_ROOT = os.path.join(ROOT, "tmp_sc10")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope10.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
D = 5.0
DIST_VALUES_FULL = [3, 5, 8, 12]


# ── Frontend mask variants ──

def red_mask_clahe(img, smin=80, vmin=110, clip=2.0, grid=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def red_mask_tophat(img, smin=80, vmin=110, k=15, gain=2.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    boost = np.clip(tophat.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(
        hsv[..., 2].astype(np.int32) + boost.astype(np.int32), 0, 255
    ).astype(np.uint8)
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def _hsv_gate(img, smin=40, vmin=80):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def red_mask_excess_red(img, thr=30, smin=40, vmin=80):
    gate = _hsv_gate(img, smin, vmin)
    B = img[..., 0].astype(np.int32)
    G = img[..., 1].astype(np.int32)
    R = img[..., 2].astype(np.int32)
    er = (R - np.maximum(G, B)) > thr
    return (er & (gate > 0)).astype(np.uint8) * 255


def red_mask_crcb(img, cr_min=145, smin=40, vmin=80):
    gate = _hsv_gate(img, smin, vmin)
    cr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 1]
    mask = (cr > cr_min) & (gate > 0)
    return mask.astype(np.uint8) * 255


# ── 4th corner families ──

def make_epq(c):
    """4 farthest contour points in 4 cardinal directions from centroid."""
    pts = c.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None
    cx, cy = pts.mean(axis=0)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    idxs = [int(np.argmax(dx)), int(np.argmin(dy)),
            int(np.argmin(dx)), int(np.argmax(dy))]
    if len(set(idxs)) < 4:
        return None
    return pts[idxs]


def make_paq(c):
    """4 farthest contour points along PCA principal axes."""
    pts = c.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None
    mu = pts.mean(axis=0)
    X = pts - mu
    cov = X.T @ X / max(len(X) - 1, 1)
    try:
        _, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    p1 = eigvecs[:, -1]
    p2 = eigvecs[:, 0]
    proj1 = X @ p1
    proj2 = X @ p2
    idxs = [int(np.argmax(proj1)), int(np.argmin(proj2)),
            int(np.argmin(proj1)), int(np.argmax(proj2))]
    if len(set(idxs)) < 4:
        return None
    return pts[idxs]


# ── Runner ──

def run_variant(name, mask_fn, family_fn, family_name, ids, p, out_dir):
    """Run pm_v5 pipeline with optional mask_fn and/or 4th family."""
    os.makedirs(out_dir, exist_ok=True)
    order = HY_ORDER + ([family_name] if family_name else [])
    det = {}
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det[iid] = []
            continue
        H, W = img.shape[:2]
        mask = (mask_fn or red_mask)(img)
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
        cnts = outer_contours(mask, p)
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p)
            if family_fn:
                base[family_name] = family_fn(c)
            for m in order:
                q = _pass(base.get(m), W, H, p)
                if q is not None:
                    hy_quads.append(q)
        hy = dedup_keep_order(hy_quads, p["dedup_iou"])
        det[iid] = hy
        total += len(hy)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in hy]), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            print(f"  [{name} {n+1}/{len(ids)}] boxes={total}", flush=True)
    print(f"  {name}: {total} boxes -> {out_dir}", flush=True)
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
    print(f"LOOP-10: 6 untried CV methods vs pm_v5 (D={D})", flush=True)
    print(f"  ids={len(ids)}  pm5_params: ortho={p['ortho_tol']} bnd={p['boundary_band']} aspect={p['max_aspect']}", flush=True)

    sample_id = ids[0]
    img = cv2.imread(os.path.join(IMG, sample_id + ".jpg"))
    print(f"\nSanity check on {sample_id} (shape {img.shape}):", flush=True)
    for name, fn in [("standard", red_mask), ("clahe", red_mask_clahe),
                      ("tophat", red_mask_tophat), ("excess_red", red_mask_excess_red),
                      ("crcb", red_mask_crcb)]:
        m = fn(img)
        print(f"  {name:12s}: {int((m > 0).sum())} px", flush=True)

    print(f"\nEvaluating pm_v5 baseline at D={D}...", flush=True)
    pm5 = evaluate_pm5_ref(ids, D)
    print(f"  pm_v5: TP={pm5['tp']} corr_f1={pm5['corrected_f1']} recall={pm5['recall']}", flush=True)

    methods = [
        ("clahe",      red_mask_clahe,      None,     None),
        ("tophat",     red_mask_tophat,     None,     None),
        ("excess_red", red_mask_excess_red, None,     None),
        ("crcb",       red_mask_crcb,       None,     None),
        ("epq",        None,                make_epq, "epq"),
        ("paq",        None,                make_paq, "paq"),
    ]

    results = {"pm5": pm5, "methods": {}, "dist_thr": D, "n_images": len(ids)}
    for name, mask_fn, family_fn, family_name in methods:
        out_dir = os.path.join(OUT_ROOT, name)
        print(f"\nRunning {name}...", flush=True)
        t0 = time.time()
        det_map = run_variant(name, mask_fn, family_fn, family_name, ids, p, out_dir)
        dt = time.time() - t0
        r = evaluate_det_map(det_map, ids, D)
        r["runtime_s"] = round(dt, 1)
        r["verdict"] = make_verdict(r, pm5)
        results["methods"][name] = r
        print(f"  {name}: TP={r['tp']} corr_f1={r['corrected_f1']} recall={r['recall']} ({dt:.0f}s) [{r['verdict']}]", flush=True)

    winners = [name for name, _, _, _ in methods if results["methods"][name]["verdict"] == "WIN"]
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
    for name, _, _, _ in methods:
        r = results["methods"][name]
        print(f"{name:12s}  {r['tp']:>5d}  {r['raw_fp']:>6d}  {r['true_fp']:>7d}  {r['recall']:>6.1f}  {r['corrected_f1']:>8.1f}  {r['verdict']}", flush=True)
    print("=" * 78, flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()