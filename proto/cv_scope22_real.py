"""LOOP-22 real validation: test close_2 RECT iter=1 min_area=1200 on 500 real images.

The synthetic WIN was marginal (+1 TP, +0.1 f1). Due to the synthetic→real gap
(LOOP-16 close_3 was synth WIN but real DEAD), we MUST validate on real data.

Compares:
  - pm_v5_1 baseline (no close, min_area=1000): frozen TP=502, corr_f1=71.5
  - close_2 candidate (close_2 RECT iter=1, min_area=1200)

Usage: python proto/cv_scope22_real.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_hybrid import outer_contours, _pass, dedup_keep_order, ROOT
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope22_real.json")
DIST_VALUES = [3, 5, 8, 12]


def apply_close(mask, ksize=2, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def run_pipeline(ids, p, use_close=False):
    """Run pm_v5_1 pipeline on real images, return {iid: [quads]}."""
    det_map = {}
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        H, W = img.shape[:2]
        mask = red_mask(img)
        if use_close:
            mask = apply_close(mask, ksize=2, iterations=1)
        if p['denoise_min_area'] > 0:
            mask = denoise(mask, p['denoise_min_area'])
        cnts = outer_contours(mask, p)
        hy_quads = []
        for c in cnts:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base.get(m), W, H, p)
                if q is not None:
                    hy_quads.append(q)
        try:
            hy = dedup_keep_order(hy_quads, p['dedup_iou'])
        except cv2.error:
            hy = dedup_keep_order(
                [q for q in hy_quads if q is not None and len(q) == 4],
                p['dedup_iou'],
            )
        det_map[iid] = hy
    dt = time.time() - t0
    return det_map, dt


def evaluate(det_map, ids, dist_thr):
    exp = load_expected()
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
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
        TP += tp; FN += len(gt) - tp
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec_raw = TP / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
    f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0
    return {
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
        "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "raw_f1": round(f1_raw, 1), "corrected_f1": round(f1_c, 1),
    }


def main():
    ids = get_ids()
    print(f"LOOP-22 real validation: {len(ids)} images, D={DIST_VALUES}", flush=True)

    # Baseline: pm_v5_1 (no close, min_area=1000)
    p_base = pm5_params()
    p_base['min_area'] = 1000
    print(f"\n[Baseline] pm_v5_1 (no close, min_area=1000)...", flush=True)
    det_base, dt = run_pipeline(ids, p_base, use_close=False)
    print(f"  {sum(len(v) for v in det_base.values())} boxes in {dt:.1f}s", flush=True)

    # Candidate: close_2 RECT iter=1, min_area=1200
    p_cand = pm5_params()
    p_cand['min_area'] = 1200
    print(f"\n[Candidate] close_2 RECT iter=1, min_area=1200...", flush=True)
    det_cand, dt = run_pipeline(ids, p_cand, use_close=True)
    print(f"  {sum(len(v) for v in det_cand.values())} boxes in {dt:.1f}s", flush=True)

    # Evaluate at multiple D
    print(f"\n{'='*78}", flush=True)
    print(f"{'D':>4s}  {'base_TP':>8s}  {'cand_TP':>8s}  {'ΔTP':>5s}  "
          f"{'base_f1':>8s}  {'cand_f1':>8s}  {'Δf1':>6s}  Verdict", flush=True)
    print(f"{'='*78}", flush=True)

    results = {'baseline': {}, 'candidate': {}}
    for d in DIST_VALUES:
        m_base = evaluate(det_base, ids, float(d))
        m_cand = evaluate(det_cand, ids, float(d))
        results['baseline'][f'D{d}'] = m_base
        results['candidate'][f'D{d}'] = m_cand
        dtp = m_cand['tp'] - m_base['tp']
        df1 = m_cand['corrected_f1'] - m_base['corrected_f1']
        if m_cand['corrected_f1'] > m_base['corrected_f1']:
            verdict = "WIN"
        elif m_cand['corrected_f1'] == m_base['corrected_f1']:
            verdict = "TIE"
        else:
            verdict = "DEAD"
        print(f"{d:>4d}  {m_base['tp']:>8d}  {m_cand['tp']:>8d}  {dtp:+5d}  "
              f"{m_base['corrected_f1']:>7.1f}%  {m_cand['corrected_f1']:>7.1f}%  "
              f"{df1:+6.1f}  {verdict}", flush=True)
    print(f"{'='*78}", flush=True)

    # D=5 detail
    m5_base = results['baseline']['D5']
    m5_cand = results['candidate']['D5']
    print(f"\nD=5 detail:", flush=True)
    print(f"  baseline: TP={m5_base['tp']} FP={m5_base['raw_fp']} trueFP={m5_base['true_fp']} "
          f"real_unl={m5_base['real_unlabeled']} corr_f1={m5_base['corrected_f1']:.1f}%", flush=True)
    print(f"  candidate: TP={m5_cand['tp']} FP={m5_cand['raw_fp']} trueFP={m5_cand['true_fp']} "
          f"real_unl={m5_cand['real_unlabeled']} corr_f1={m5_cand['corrected_f1']:.1f}%", flush=True)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()