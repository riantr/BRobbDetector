"""LOOP-24: Per-stage instrumentation + DoubleML causal analysis of pm_v5_1 pipeline.

Pipeline stages:
  1. red_mask      — HSV threshold → binary mask
  2. denoise       — remove small components (< denoise_min_area)
  3. outer_contours — findContours RETR_CCOMP + area filter (≥ min_area)
  4. _base_quads   — hull/edgefit/mar per contour
  5. _pass         — admission filter (side + ortho + boundary)
  6. dedup         — IoU dedup

For each of 500 real images, record per-stage features + outcome (TP/FP/FN at D=5).
Then use DoubleML PLR to estimate the causal effect of each stage's key quality
feature on the final TP, controlling for all other stage features.

The stage with the largest |θ| is where optimization would yield the biggest gain.

Usage: python proto/cv_scope24.py
"""
import sys, os, json, time, argparse, warnings
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids, denoise, _side_ok
from cv_hybrid import outer_contours, _pass, dedup_keep_order, ROOT
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from quad_ortho import is_orthogonal_quad
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled

PANEL_PATH = os.path.join(ROOT, "benchmark", "cv_scope24_panel.json")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope24.json")
D = 5.0


def instrument_pipeline(img, p):
    """Run pm_v5_1 pipeline with per-stage instrumentation.

    Returns (final_quads, stage_features_dict).
    """
    H, W = img.shape[:2]
    total_pixels = H * W
    features = {}

    # ── Stage 1: red_mask ──
    mask_raw = red_mask(img)
    n_red = int(np.count_nonzero(mask_raw))
    n_comp_raw, _ = cv2.connectedComponents(mask_raw, connectivity=8)
    n_comp_raw -= 1  # subtract background
    features['s1_red_pixels'] = n_red
    features['s1_red_ratio'] = n_red / total_pixels
    features['s1_n_components'] = n_comp_raw

    # ── Stage 2: denoise ──
    if p['denoise_min_area'] > 0:
        n_before = n_comp_raw
        mask = denoise(mask_raw, p['denoise_min_area'])
        n_after_comp, _ = cv2.connectedComponents(mask, connectivity=8)
        n_after_comp -= 1
        n_after_pixels = int(np.count_nonzero(mask))
        features['s2_n_components_before'] = n_before
        features['s2_n_components_after'] = n_after_comp
        features['s2_removed_ratio'] = (n_before - n_after_comp) / max(1, n_before)
        features['s2_kept_area_ratio'] = n_after_pixels / max(1, n_red)
    else:
        mask = mask_raw
        features['s2_n_components_before'] = n_comp_raw
        features['s2_n_components_after'] = n_comp_raw
        features['s2_removed_ratio'] = 0.0
        features['s2_kept_area_ratio'] = 1.0

    # ── Stage 3: outer_contours ──
    cnts_all, hier_all = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    n_cnts_raw = 0
    n_cnts_outer_raw = 0
    if hier_all is not None:
        hier_all = hier_all[0]
        for i, c in enumerate(cnts_all):
            if hier_all[i][3] == -1:  # outer contour
                n_cnts_outer_raw += 1
                n_cnts_raw += 1
    cnts = outer_contours(mask, p)
    n_cnts = len(cnts)
    total_cnt_area = sum(cv2.contourArea(c) for c in cnts)
    mean_cnt_area = total_cnt_area / max(1, n_cnts)
    features['s3_n_contours_raw'] = n_cnts_outer_raw
    features['s3_n_contours'] = n_cnts
    features['s3_total_contour_area'] = total_cnt_area
    features['s3_mean_contour_area'] = mean_cnt_area
    features['s3_area_pass_ratio'] = n_cnts / max(1, n_cnts_outer_raw)

    # ── Stage 4: _base_quads ──
    n_hull = n_edgefit = n_mar = 0
    all_candidates = []
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = base.get(m)
            if q is not None and len(q) == 4:
                all_candidates.append(q)
                if m == 'hull':
                    n_hull += 1
                elif m == 'edgefit':
                    n_edgefit += 1
                elif m == 'mar':
                    n_mar += 1
    features['s4_n_hull'] = n_hull
    features['s4_n_edgefit'] = n_edgefit
    features['s4_n_mar'] = n_mar
    features['s4_n_total_candidates'] = n_hull + n_edgefit + n_mar

    # ── Stage 5: _pass (admission filter) ──
    n_passed = 0
    n_rejected_side = 0
    n_rejected_ortho = 0
    passed_quads = []
    for q in all_candidates:
        if q is None or not _side_ok(q, p):
            n_rejected_side += 1
            continue
        if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                                   boundary_band=p["boundary_band"]):
            n_rejected_ortho += 1
            continue
        n_passed += 1
        passed_quads.append(q.astype(np.int32))
    n_total_cand = len(all_candidates)
    features['s5_n_passed'] = n_passed
    features['s5_n_rejected_side'] = n_rejected_side
    features['s5_n_rejected_ortho'] = n_rejected_ortho
    features['s5_pass_rate'] = n_passed / max(1, n_total_cand)

    # ── Stage 6: dedup ──
    n_before_dedup = n_passed
    try:
        final = dedup_keep_order(passed_quads, p['dedup_iou'])
    except cv2.error:
        final = dedup_keep_order([q for q in passed_quads if q is not None and len(q) == 4],
                                  p['dedup_iou'])
    n_after_dedup = len(final)
    features['s6_n_before_dedup'] = n_before_dedup
    features['s6_n_after_dedup'] = n_after_dedup
    features['s6_dedup_rate'] = (n_before_dedup - n_after_dedup) / max(1, n_before_dedup)

    return final, features


def run_doubleml(panel_df):
    """Run DoubleML PLR for each stage's key feature.

    Returns dict of {stage: {theta, ci_lower, ci_upper, p_value}}.
    """
    import pandas as pd
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Define key feature per stage (the "treatment" variable)
    stage_treatments = {
        'stage1_red_mask':       's1_red_ratio',
        'stage2_denoise':        's2_kept_area_ratio',
        'stage3_outer_contours': 's3_n_contours',
        'stage4_base_quads':     's4_n_total_candidates',
        'stage5_admission':      's5_pass_rate',
        'stage6_dedup':          's6_dedup_rate',
    }

    # All features as potential controls
    all_features = [c for c in panel_df.columns
                    if c.startswith('s') and c not in stage_treatments.values()]

    Y = 'tp'  # outcome: image-level TP

    results = {}
    for stage, treatment in stage_treatments.items():
        controls = [c for c in all_features if c != treatment]
        cols = [Y, treatment] + controls
        df_sub = panel_df[cols].copy()

        # Drop rows with NaN/inf
        df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_sub) < 50:
            results[stage] = {'theta': None, 'error': 'insufficient data'}
            continue

        try:
            dml_data = DoubleMLData(df_sub, y_col=Y, d_cols=treatment, x_cols=controls)
            ml_l = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
            ml_m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
            dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
            dml_plr.fit()

            theta = float(dml_plr.coef[0])
            ci = dml_plr.confint()
            pval = float(dml_plr.pval[0])
            results[stage] = {
                'treatment_var': treatment,
                'theta': round(theta, 6),
                'ci_lower': round(float(ci.iloc[0, 0]), 6),
                'ci_upper': round(float(ci.iloc[0, 1]), 6),
                'p_value': round(pval, 6),
                'n_samples': len(df_sub),
            }
        except Exception as e:
            results[stage] = {'theta': None, 'error': str(e)[:200]}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-doubleml', action='store_true', help='skip DoubleML analysis')
    args = ap.parse_args()

    p = pm5_params()
    p['min_area'] = 1000  # pm_v5_1

    ids = get_ids()
    print(f"LOOP-24: Per-stage instrumentation + DoubleML on {len(ids)} real images (D={D})", flush=True)
    print(f"  pm_v5_1: min_area={p['min_area']:.0f} denoise={p['denoise_min_area']:.0f} "
          f"ortho_tol={p['ortho_tol']:.0f} bnd={p['boundary_band']:.0f}", flush=True)

    exp = load_expected()
    panel = []
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        final_quads, stage_feats = instrument_pipeline(img, p)

        # Evaluate outcome
        det = [q.astype(np.int32) for q in final_quads]
        tp, unmatched = match_boxes_point(det, gt, D)
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)
        true_fp = 0
        real_unl = 0
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1

        row = {
            'iid': iid,
            'n_gt': len(gt),
            'tp': tp,
            'fp': len(unmatched),
            'fn': len(gt) - tp,
            'true_fp': true_fp,
            'real_unlabeled': real_unl,
            'n_det': len(det),
            'recall': tp / max(1, len(gt)),
            'precision': tp / max(1, len(det)),
        }
        row.update(stage_feats)
        panel.append(row)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(ids)} images processed ({time.time()-t0:.1f}s)", flush=True)

    dt = time.time() - t0
    print(f"\nProcessed {len(panel)} images in {dt:.1f}s", flush=True)

    # Summary stats
    total_tp = sum(r['tp'] for r in panel)
    total_fp = sum(r['fp'] for r in panel)
    total_fn = sum(r['fn'] for r in panel)
    total_gt = sum(r['n_gt'] for r in panel)
    total_det = sum(r['n_det'] for r in panel)
    recall = total_tp / max(1, total_gt) * 100
    prec = total_tp / max(1, total_det) * 100
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    print(f"  Total: TP={total_tp} FP={total_fp} FN={total_fn} GT={total_gt} det={total_det}", flush=True)
    print(f"  recall={recall:.1f}% precision={prec:.1f}% f1={f1:.1f}%", flush=True)

    # Per-stage summary
    print(f"\nPer-stage feature summaries:", flush=True)
    stage_cols = {
        'Stage 1 (red_mask)':       ['s1_red_pixels', 's1_red_ratio', 's1_n_components'],
        'Stage 2 (denoise)':        ['s2_n_components_before', 's2_n_components_after', 's2_removed_ratio', 's2_kept_area_ratio'],
        'Stage 3 (outer_contours)': ['s3_n_contours_raw', 's3_n_contours', 's3_total_contour_area', 's3_area_pass_ratio'],
        'Stage 4 (base_quads)':     ['s4_n_hull', 's4_n_edgefit', 's4_n_mar', 's4_n_total_candidates'],
        'Stage 5 (admission)':      ['s5_n_passed', 's5_n_rejected_side', 's5_n_rejected_ortho', 's5_pass_rate'],
        'Stage 6 (dedup)':          ['s6_n_before_dedup', 's6_n_after_dedup', 's6_dedup_rate'],
    }
    for stage_name, cols in stage_cols.items():
        print(f"\n  {stage_name}:", flush=True)
        for c in cols:
            vals = [r[c] for r in panel]
            mean = np.mean(vals)
            std = np.std(vals)
            med = np.median(vals)
            print(f"    {c:30s}: mean={mean:10.2f}  std={std:10.2f}  median={med:10.2f}", flush=True)

    # Save panel
    with open(PANEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(panel, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved panel: {PANEL_PATH}", flush=True)

    # DoubleML analysis
    if not args.no_doubleml:
        print(f"\n{'='*78}", flush=True)
        print(f"DoubleML PLR Analysis (outcome=TP, per-stage treatment effects)", flush=True)
        print(f"{'='*78}", flush=True)

        import pandas as pd
        df = pd.DataFrame(panel)
        # Drop non-numeric columns
        df = df.select_dtypes(include=[np.number])
        # Drop constant columns
        df = df.loc[:, df.std() > 0]

        dml_results = run_doubleml(df)

        print(f"\n{'Stage':>30s}  {'Treatment':>25s}  {'θ':>10s}  {'CI':>20s}  {'p-value':>8s}  |θ|×std(D)", flush=True)
        print(f"{'-'*100}", flush=True)
        stage_effects = []
        for stage, res in dml_results.items():
            if res.get('theta') is None:
                print(f"{stage:>30s}  {'ERROR':>25s}  {res.get('error', '')}", flush=True)
                continue
            tv = res['treatment_var']
            theta = res['theta']
            ci_l = res['ci_lower']
            ci_u = res['ci_upper']
            pval = res['p_value']
            std_d = float(df[tv].std())
            effect_magnitude = abs(theta) * std_d  # effect of 1 std change
            stage_effects.append((stage, tv, theta, effect_magnitude, pval))
            print(f"{stage:>30s}  {tv:>25s}  {theta:>10.6f}  [{ci_l:>8.6f},{ci_u:>8.6f}]  "
                  f"{pval:>8.4f}  {effect_magnitude:>10.4f}", flush=True)

        print(f"{'-'*100}", flush=True)
        print(f"|θ|×std(D) = effect magnitude of 1 standard deviation improvement in treatment", flush=True)

        # Rank by effect magnitude
        if stage_effects:
            stage_effects.sort(key=lambda x: -x[3])
            print(f"\nRanking by effect magnitude (|θ|×std(D)):", flush=True)
            for rank, (stage, tv, theta, mag, pval) in enumerate(stage_effects, 1):
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"  {rank}. {stage:30s}  |θ|×std(D)={mag:.4f}  p={pval:.4f} {sig}", flush=True)
            print(f"\n  => Optimization of '{stage_effects[0][0]}' would yield the largest effect.", flush=True)

        all_results = {
            'n_images': len(panel),
            'dist_thr': D,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'recall': round(recall, 2),
            'f1': round(f1, 2),
            'doubleml': dml_results,
            'ranking': [
                {'rank': i+1, 'stage': s, 'treatment': tv,
                 'theta': t, 'effect_magnitude': m, 'p_value': p}
                for i, (s, tv, t, m, p) in enumerate(stage_effects)
            ],
        }
        with open(RESULT_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
        print(f"\nSaved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()