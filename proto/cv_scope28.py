"""LOOP-28 (Path-3): Real-data denoise tuning.

LOOP-24 DoubleML found Stage 2 (denoise) is the only statistically significant
stage (p=0.005), with negative theta on kept_area_ratio — more aggressive
denoising (higher denoise_min_area) should increase TP.

Current default: denoise_min_area=40 (frozen pm_v5_1 baseline: TP=502, corr_f1=71.5).

Method:
  1. Sweep denoise_min_area [0,20,40,60,80,100,120,150,200,300,500] on 500 real images
  2. Run full pm_v5_1 pipeline for each value
  3. Evaluate with point-match metric at D=3,5,8,12
  4. Compute GT-corrected metrics (true_fp, real_unlabeled)
  5. GATE: TP > 502 AND corr_f1 > 71.5 at D=5

Output:
  benchmark/cv_scope28.json
  tmp_sc28_dma<value>/  (detection JSONs for each value)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT, DIST_VALUES,
                       _edgefit_corners)
from cv_scope4 import HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import (load_boxes, match_boxes_point, is_real_unlabeled,
                             red_support)

RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope28.json")
BASELINE_TP = 502
BASELINE_F1 = 71.5
DENOISE_GRID = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 500]


def run_pipeline(img, p):
    """Run pm_v5_1 pipeline and return detected quads."""
    H, W = img.shape[:2]
    mask = red_mask(img)
    if p['denoise_min_area'] > 0:
        mask = denoise(mask, p['denoise_min_area'])
    cnts = outer_contours(mask, p)
    all_cands = []
    for c in cnts:
        qh = _quad_adaptive_hull(c, p)
        qe = _edgefit_corners(c, p)
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        qm = cv2.boxPoints(cv2.minAreaRect(c)) if rw >= 1 and rh >= 1 else None
        for q in [qh, qe, qm]:
            if q is not None and len(q) == 4:
                qq = _pass(q, W, H, p)
                if qq is not None:
                    all_cands.append(qq)
    return dedup_keep_order(all_cands, p['dedup_iou'])


def _quad_adaptive_hull(c, p):
    from canny_quad import _quad_adaptive
    return _quad_adaptive(cv2.convexHull(c), p["approx_eps_list"])


def evaluate(ids, det_map, gt_map, mask_cache, exp, dist_thr):
    """Evaluate at a given D threshold with GT-corrected metrics."""
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    for iid in ids:
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        gt = [g.astype(np.int32) for g in gt_map.get(iid, [])]
        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp; FN += len(gt) - tp
        raw_fp += len(unmatched)
        csv_n = exp.get(iid, 0)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_cache[iid]):
                real_unl += 1
            else:
                true_fp += 1
    n_gt = TP + FN
    n_det = TP + raw_fp
    recall = TP / n_gt * 100 if n_gt else 0
    prec_raw = TP / n_det * 100 if n_det else 0
    f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0
    return {
        'tp': TP, 'fn': FN, 'raw_fp': raw_fp, 'true_fp': true_fp,
        'real_unlabeled': real_unl, 'n_gt': n_gt, 'n_det': n_det,
        'recall': round(recall, 1), 'raw_f1': round(f1_raw, 1),
        'corrected_f1': round(f1_c, 1),
    }


def main():
    ids = get_ids()
    exp = load_expected()
    print(f"LOOP-28: Real-data denoise tuning on {len(ids)} images", flush=True)
    print(f"  denoise_min_area grid: {DENOISE_GRID}", flush=True)
    print(f"  Baseline: dma=40, TP={BASELINE_TP}, corr_f1={BASELINE_F1}", flush=True)

    # Load GT
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"  GT boxes: {total_gt}", flush=True)

    # Build mask cache
    print(f"\nBuilding mask cache...", flush=True)
    mask_cache = {}
    img_cache = {}
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        img_cache[iid] = img
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask_cache[iid] = mask_d
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] cached", flush=True)
    print(f"  {len(mask_cache)} images cached", flush=True)

    # Sweep denoise_min_area
    p = pm5_params()
    all_results = {}

    for dma in DENOISE_GRID:
        print(f"\n--- denoise_min_area = {dma} ---", flush=True)
        p['denoise_min_area'] = dma
        t0 = time.time()

        # Run pipeline
        det_map = {}
        total_det = 0
        out_dir = os.path.join(ROOT, f"tmp_sc28_dma{dma}")
        os.makedirs(out_dir, exist_ok=True)
        for n, iid in enumerate(ids):
            img = img_cache.get(iid)
            if img is None:
                det_map[iid] = []
                continue
            quads = run_pipeline(img, p)
            det_map[iid] = quads
            total_det += len(quads)
            with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, [np.asarray(q, np.int32) for q in quads]), f,
                          ensure_ascii=False)
            if (n + 1) % 100 == 0:
                print(f"  [{n+1}/{len(ids)}] det={total_det}", flush=True)

        elapsed = time.time() - t0
        print(f"  {total_det} detections in {elapsed:.1f}s", flush=True)

        # Evaluate at each D
        dma_results = {'n_det': total_det, 'per_D': {}}
        for dd in DIST_VALUES:
            m = evaluate(ids, det_map, gt_map, mask_cache, exp, float(dd))
            dma_results['per_D'][str(dd)] = m

        # Print D=5 summary
        m5 = dma_results['per_D']['5']
        d_tp = m5['tp'] - BASELINE_TP
        d_f1 = m5['corrected_f1'] - BASELINE_F1
        flag = " *** WIN" if m5['tp'] > BASELINE_TP and m5['corrected_f1'] > BASELINE_F1 else ""
        print(f"  D=5: TP={m5['tp']} ({d_tp:+d})  recall={m5['recall']}%  "
              f"corr_f1={m5['corrected_f1']} ({d_f1:+.1f})  "
              f"trueFP={m5['true_fp']}  real_unl={m5['real_unlabeled']}{flag}", flush=True)

        all_results[str(dma)] = dma_results

    # Summary table
    print(f"\n{'='*90}")
    print(f"LOOP-28 SUMMARY: denoise_min_area sweep on 500 real images")
    print(f"{'='*90}")
    print(f"\n{'dma':>5s} {'n_det':>6s}  " +
          "  ".join(f"D={d:<21d}" for d in DIST_VALUES))
    for dma in DENOISE_GRID:
        r = all_results[str(dma)]
        parts = []
        for dd in DIST_VALUES:
            m = r['per_D'][str(dd)]
            parts.append(f"TP={m['tp']:4d} f1={m['corrected_f1']:5.1f}")
        print(f"{dma:>5d} {r['n_det']:>6d}  " + "  ".join(parts))

    print(f"\n  D=5 detail:")
    print(f"  {'dma':>5s} {'TP':>5s} {'dTP':>5s} {'recall':>7s} "
          f"{'corr_f1':>8s} {'df1':>6s} {'trueFP':>7s} {'real_unl':>9s} {'verdict':>8s}")
    for dma in DENOISE_GRID:
        r = all_results[str(dma)]
        m = r['per_D']['5']
        d_tp = m['tp'] - BASELINE_TP
        d_f1 = m['corrected_f1'] - BASELINE_F1
        if m['tp'] > BASELINE_TP and m['corrected_f1'] > BASELINE_F1:
            verdict = "WIN"
        elif m['tp'] == BASELINE_TP and m['corrected_f1'] == BASELINE_F1:
            verdict = "TIE"
        elif abs(d_tp) < 5 and abs(d_f1) < 1.0:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"  {dma:>5d} {m['tp']:>5d} {d_tp:>+5d} {m['recall']:>6.1f}% "
              f"{m['corrected_f1']:>7.1f}% {d_f1:>+5.1f} {m['true_fp']:>7d} "
              f"{m['real_unlabeled']:>9d} {verdict:>8s}")
    print(f"{'='*90}")

    # Save
    results = {
        'n_images': len(ids),
        'total_gt': total_gt,
        'baseline_tp': BASELINE_TP,
        'baseline_f1': BASELINE_F1,
        'denoise_grid': DENOISE_GRID,
        'results': all_results,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()