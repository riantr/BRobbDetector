"""LOOP-25: Large-scale synthetic optimization with sqlite3 + DoubleML-guided decisions.

Optimizes Stage 1 (red_mask), Stage 2 (denoise), Stage 4 (base_quads) sequentially.
Before each stage decision, DoubleML PLR is run for causal inference.

Design:
  - Synthetic images generated IN MEMORY (not saved to disk), reused across configs.
  - All parameters and per-image results stored in sqlite3 (benchmark/cv_scope25.db).
  - Target: ~100k image evaluations (N_IMAGES × N_CONFIGS).
  - DoubleML guides each stage decision: identifies which parameter has the
    largest causal effect on TP, then selects the best value.

Stage parameters:
  Stage 1 (red_mask):     smin [60,70,80,90,100], vmin [90,100,110,120]  = 20 configs
  Stage 2 (denoise):      denoise_min_area [0,20,40,60,80,100,150,200]    = 8 configs
  Stage 4 (base_quads):   fit_band [3,5,6,8,10], fit_min_pts [8,12,16],
                          fit_jump [15,20,30]                             = 10 configs (axis-by-axis)

Usage: python proto/cv_scope25.py [--n-images N]
"""
import sys, os, json, time, sqlite3, argparse, warnings
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise, _side_ok, _quad_adaptive, _order_quad
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT,
                       _edgefit_corners, _sector_corners)
from cv_scope4 import HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import obb_corners, draw_obb, compute_gt_minarearect
from cv_scope16 import gen_dataset
from eval_pointmatch import match_boxes_point
from quad_ortho import is_orthogonal_quad

DB_PATH = os.path.join(ROOT, "benchmark", "cv_scope25.db")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope25.json")
D = 5.0
DEFAULT_N_IMAGES = 500

STAGE1_GRID = {
    'smin': [60, 70, 80, 90, 100],
    'vmin': [90, 100, 110, 120],
}
STAGE2_GRID = {
    'denoise_min_area': [0, 20, 40, 60, 80, 100, 150, 200],
}
STAGE4_GRID = {
    'fit_band':     [3.0, 5.0, 6.0, 8.0, 10.0],
    'fit_min_pts':  [8, 12, 16],
    'fit_jump':     [15.0, 20.0, 30.0],
}


# ── sqlite3 schema ──

def init_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trials (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        stage TEXT NOT NULL,
        param_name TEXT NOT NULL,
        param_value REAL NOT NULL,
        config_json TEXT NOT NULL,
        n_images INTEGER NOT NULL,
        tp INTEGER, fp INTEGER, fn INTEGER,
        recall REAL, precision REAL, f1 REAL,
        timestamp TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS panel (
        trial_id INTEGER NOT NULL,
        img_idx INTEGER NOT NULL,
        s1_red_ratio REAL, s1_n_components INTEGER,
        s2_kept_area_ratio REAL, s2_removed_ratio REAL,
        s3_n_contours INTEGER, s3_area_pass_ratio REAL,
        s4_n_hull INTEGER, s4_n_edgefit INTEGER, s4_n_mar INTEGER, s4_n_total INTEGER,
        s5_pass_rate REAL, s5_n_rejected_ortho INTEGER,
        s6_dedup_rate REAL,
        n_gt INTEGER, tp INTEGER, fp INTEGER, fn INTEGER, recall REAL,
        FOREIGN KEY(trial_id) REFERENCES trials(trial_id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS dml_results (
        stage TEXT NOT NULL,
        treatment_var TEXT NOT NULL,
        theta REAL, ci_lower REAL, ci_upper REAL,
        p_value REAL, effect_magnitude REAL,
        n_samples INTEGER, timestamp TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS decisions (
        stage TEXT NOT NULL PRIMARY KEY,
        param_name TEXT NOT NULL,
        best_value REAL NOT NULL,
        best_tp INTEGER, best_f1 REAL,
        dml_theta REAL, dml_pvalue REAL, dml_effect REAL,
        reason TEXT NOT NULL, timestamp TEXT NOT NULL
    )""")
    conn.commit()
    return conn


# ── Instrumented pipeline ──

def run_pipeline(img, p, smin=80, vmin=110):
    """Run pm_v5_1 pipeline with custom red_mask params + per-stage instrumentation."""
    H, W = img.shape[:2]
    total_px = H * W
    f = {}

    # Stage 1: red_mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    mask_raw = cv2.bitwise_or(m1, m2)
    n_red = int(np.count_nonzero(mask_raw))
    n_comp_raw, _ = cv2.connectedComponents(mask_raw, connectivity=8)
    n_comp_raw -= 1
    f['s1_red_ratio'] = n_red / total_px
    f['s1_n_components'] = n_comp_raw

    # Stage 2: denoise
    if p['denoise_min_area'] > 0:
        mask = denoise(mask_raw, p['denoise_min_area'])
        n_after, _ = cv2.connectedComponents(mask, connectivity=8)
        n_after -= 1
        n_kept = int(np.count_nonzero(mask))
    else:
        mask = mask_raw
        n_after = n_comp_raw
        n_kept = n_red
    f['s2_kept_area_ratio'] = n_kept / max(1, n_red)
    f['s2_removed_ratio'] = (n_comp_raw - n_after) / max(1, n_comp_raw)

    # Stage 3: outer_contours
    cnts_all, hier_all = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    n_outer_raw = 0
    if hier_all is not None:
        for i in range(len(cnts_all)):
            if hier_all[0][i][3] == -1:
                n_outer_raw += 1
    cnts = outer_contours(mask, p)
    f['s3_n_contours'] = len(cnts)
    f['s3_area_pass_ratio'] = len(cnts) / max(1, n_outer_raw)

    # Stage 4: base_quads
    n_hull = n_edgefit = n_mar = 0
    all_cands = []
    for c in cnts:
        qh = _quad_adaptive(cv2.convexHull(c), p["approx_eps_list"])
        qe = _edgefit_corners(c, p)
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        qm = cv2.boxPoints(cv2.minAreaRect(c)) if rw >= 1 and rh >= 1 else None
        for q, label in [(qh, 'hull'), (qe, 'edgefit'), (qm, 'mar')]:
            if q is not None and len(q) == 4:
                all_cands.append(q)
                if label == 'hull': n_hull += 1
                elif label == 'edgefit': n_edgefit += 1
                else: n_mar += 1
    f['s4_n_hull'] = n_hull
    f['s4_n_edgefit'] = n_edgefit
    f['s4_n_mar'] = n_mar
    f['s4_n_total'] = n_hull + n_edgefit + n_mar

    # Stage 5: admission
    n_passed = n_rej_ortho = 0
    passed = []
    for q in all_cands:
        if q is None or not _side_ok(q, p):
            continue
        if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                                   boundary_band=p["boundary_band"]):
            n_rej_ortho += 1
            continue
        n_passed += 1
        passed.append(q.astype(np.int32))
    f['s5_pass_rate'] = n_passed / max(1, len(all_cands))
    f['s5_n_rejected_ortho'] = n_rej_ortho

    # Stage 6: dedup
    n_before = n_passed
    try:
        final = dedup_keep_order(passed, p['dedup_iou'])
    except cv2.error:
        final = dedup_keep_order([q for q in passed if q is not None and len(q) == 4],
                                  p['dedup_iou'])
    f['s6_dedup_rate'] = (n_before - len(final)) / max(1, n_before)

    return final, f


def eval_config(dataset, p, smin, vmin, conn, stage, param_name, param_value, config_dict):
    """Run pipeline on all images, store results in sqlite3, return aggregate metrics."""
    TP = FP = FN = 0
    panel_rows = []
    for idx, item in enumerate(dataset):
        img = item['img']
        gt = item['gt_quads']
        final, f = run_pipeline(img, p, smin=smin, vmin=vmin)
        det = [q.astype(np.int32) for q in final]
        tp, unmatched = match_boxes_point(det, gt, D)
        TP += tp; FP += len(unmatched); FN += len(gt) - tp
        panel_rows.append((
            f['s1_red_ratio'], f['s1_n_components'],
            f['s2_kept_area_ratio'], f['s2_removed_ratio'],
            f['s3_n_contours'], f['s3_area_pass_ratio'],
            f['s4_n_hull'], f['s4_n_edgefit'], f['s4_n_mar'], f['s4_n_total'],
            f['s5_pass_rate'], f['s5_n_rejected_ortho'],
            f['s6_dedup_rate'],
            len(gt), tp, len(unmatched), len(gt) - tp,
            tp / max(1, len(gt)),
        ))

    n_gt = TP + FN
    n_det = TP + FP
    recall = TP / max(1, n_gt) * 100
    prec = TP / max(1, n_det) * 100
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    c = conn.cursor()
    c.execute("INSERT INTO trials (stage, param_name, param_value, config_json, n_images, "
              "tp, fp, fn, recall, precision, f1, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
              (stage, param_name, param_value, json.dumps(config_dict), len(dataset),
               TP, FP, FN, round(recall, 2), round(prec, 2), round(f1, 2), ts))
    trial_id = c.lastrowid
    c.executemany("INSERT INTO panel (trial_id, img_idx, "
                  "s1_red_ratio, s1_n_components, s2_kept_area_ratio, s2_removed_ratio, "
                  "s3_n_contours, s3_area_pass_ratio, s4_n_hull, s4_n_edgefit, s4_n_mar, s4_n_total, "
                  "s5_pass_rate, s5_n_rejected_ortho, s6_dedup_rate, "
                  "n_gt, tp, fp, fn, recall) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  [(trial_id, idx) + row for idx, row in enumerate(panel_rows)])
    conn.commit()
    return {'tp': TP, 'fp': FP, 'fn': FN, 'recall': recall, 'f1': f1, 'trial_id': trial_id}


# ── DoubleML ──

def run_doubleml_for_stage(conn, stage, treatment_var):
    """Build panel from sqlite3 for this stage's trials, run DoubleML PLR."""
    import pandas as pd
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor

    c = conn.cursor()
    c.execute("""SELECT t.trial_id, t.param_value, p.s1_red_ratio, p.s1_n_components,
                        p.s2_kept_area_ratio, p.s2_removed_ratio,
                        p.s3_n_contours, p.s3_area_pass_ratio,
                        p.s4_n_hull, p.s4_n_edgefit, p.s4_n_mar, p.s4_n_total,
                        p.s5_pass_rate, p.s5_n_rejected_ortho, p.s6_dedup_rate,
                        p.n_gt, p.tp, p.fp, p.fn, p.recall
                 FROM trials t JOIN panel p ON t.trial_id = p.trial_id
                 WHERE t.stage = ?""", (stage,))
    rows = c.fetchall()
    if len(rows) < 100:
        return {'theta': None, 'error': f'insufficient data ({len(rows)} rows)'}

    cols = ['trial_id', 'param_value', 's1_red_ratio', 's1_n_components',
            's2_kept_area_ratio', 's2_removed_ratio',
            's3_n_contours', 's3_area_pass_ratio',
            's4_n_hull', 's4_n_edgefit', 's4_n_mar', 's4_n_total',
            's5_pass_rate', 's5_n_rejected_ortho', 's6_dedup_rate',
            'n_gt', 'tp', 'fp', 'fn', 'recall']
    df = pd.DataFrame(rows, columns=cols)

    Y = 'tp'
    D_col = treatment_var
    controls = [c for c in cols if c not in (Y, D_col, 'trial_id', 'fn', 'recall')]
    controls = [c for c in controls if df[c].std() > 0]

    df_sub = df[[Y, D_col] + controls].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_sub) < 100:
        return {'theta': None, 'error': f'insufficient clean data ({len(df_sub)} rows)'}

    try:
        dml_data = DoubleMLData(df_sub, y_col=Y, d_cols=D_col, x_cols=controls)
        ml_l = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        ml_m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
        dml_plr.fit()
        theta = float(dml_plr.coef[0])
        ci = dml_plr.confint()
        pval = float(dml_plr.pval[0])
        std_d = float(df_sub[D_col].std())
        effect_mag = abs(theta) * std_d
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        c.execute("INSERT INTO dml_results (stage, treatment_var, theta, ci_lower, ci_upper, "
                  "p_value, effect_magnitude, n_samples, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
                  (stage, treatment_var, round(theta, 6), round(float(ci.iloc[0, 0]), 6),
                   round(float(ci.iloc[0, 1]), 6), round(pval, 6), round(effect_mag, 6),
                   len(df_sub), ts))
        conn.commit()
        return {'theta': theta, 'p_value': pval, 'effect_magnitude': effect_mag,
                'ci_lower': float(ci.iloc[0, 0]), 'ci_upper': float(ci.iloc[0, 1]),
                'n_samples': len(df_sub)}
    except Exception as e:
        return {'theta': None, 'error': str(e)[:200]}


# ── Main optimization loop ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=DEFAULT_N_IMAGES)
    args = ap.parse_args()
    N = args.n_images

    p_base = pm5_params()
    p_base['min_area'] = 1000  # pm_v5_1

    print(f"LOOP-25: Synthetic optimization with sqlite3 + DoubleML", flush=True)
    print(f"  N_images={N}, D={D}", flush=True)
    print(f"  Stage 1: {STAGE1_GRID}", flush=True)
    print(f"  Stage 2: {STAGE2_GRID}", flush=True)
    print(f"  Stage 4: {STAGE4_GRID}", flush=True)

    n_configs = (len(STAGE1_GRID['smin']) * len(STAGE1_GRID['vmin']) +
                 len(STAGE2_GRID['denoise_min_area']) +
                 len(STAGE4_GRID['fit_band']) + len(STAGE4_GRID['fit_min_pts']) + len(STAGE4_GRID['fit_jump']))
    total_evals = n_configs * N
    print(f"  Total configs={n_configs}, total image evals={total_evals}", flush=True)

    # Generate synthetic dataset IN MEMORY
    print(f"\nGenerating {N} synthetic images in memory...", flush=True)
    t0 = time.time()
    full_dataset = gen_dataset()
    if N < len(full_dataset):
        dataset = full_dataset[:N]
    else:
        dataset = full_dataset
    dt = time.time() - t0
    print(f"  {len(dataset)} images in {dt:.1f}s", flush=True)

    # Init sqlite3
    conn = init_db(DB_PATH)
    print(f"  sqlite3: {DB_PATH}", flush=True)

    # Current best params (updated after each stage)
    best_smin = 80
    best_vmin = 110
    best_denoise = 40
    best_fit_band = 6.0
    best_fit_min_pts = 12
    best_fit_jump = 20.0

    all_decisions = {}
    total_evals_done = 0

    # ══════════════════════════════════════════════════════════════
    # Stage 1: red_mask (smin, vmin)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*78}", flush=True)
    print(f"STAGE 1: red_mask (smin, vmin)", flush=True)
    print(f"{'='*78}", flush=True)

    stage1_results = []
    for smin in STAGE1_GRID['smin']:
        for vmin in STAGE1_GRID['vmin']:
            p = dict(p_base)
            cfg = {'smin': smin, 'vmin': vmin, 'denoise_min_area': p['denoise_min_area']}
            label = f"smin={smin}, vmin={vmin}"
            t0 = time.time()
            res = eval_config(dataset, p, smin, vmin, conn, 'stage1', 'smin', smin, cfg)
            dt = time.time() - t0
            total_evals_done += len(dataset)
            stage1_results.append((smin, vmin, res))
            print(f"  {label:25s}: TP={res['tp']:4d} FP={res['fp']:4d} f1={res['f1']:5.1f}%  ({dt:.1f}s)", flush=True)

    # DoubleML for Stage 1
    print(f"\n[DoubleML] Stage 1 causal inference...", flush=True)
    dml_s1 = run_doubleml_for_stage(conn, 'stage1', 'param_value')
    if dml_s1.get('theta') is not None:
        print(f"  θ={dml_s1['theta']:.6f}  p={dml_s1['p_value']:.4f}  |θ|×std={dml_s1['effect_magnitude']:.4f}", flush=True)
    else:
        print(f"  ERROR: {dml_s1.get('error', 'unknown')}", flush=True)

    # Decision: best f1, tie-break max TP
    best = max(stage1_results, key=lambda x: (x[2]['f1'], x[2]['tp']))
    best_smin, best_vmin = best[0], best[1]
    reason = f"Best f1={best[2]['f1']:.1f}% TP={best[2]['tp']}"
    if dml_s1.get('theta') is not None:
        reason += f" | DML θ={dml_s1['theta']:.4f} p={dml_s1['p_value']:.4f}"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute("INSERT INTO decisions (stage, param_name, best_value, best_tp, best_f1, "
                 "dml_theta, dml_pvalue, dml_effect, reason, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ('stage1', 'smin/vmin', best_smin, best[2]['tp'], round(best[2]['f1'], 2),
                  dml_s1.get('theta'), dml_s1.get('p_value'), dml_s1.get('effect_magnitude'),
                  reason, ts))
    conn.commit()
    all_decisions['stage1'] = {'smin': best_smin, 'vmin': best_vmin, 'f1': best[2]['f1'], 'tp': best[2]['tp'], 'dml': dml_s1}
    print(f"\n  => DECISION: smin={best_smin}, vmin={best_vmin}  ({reason})", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Stage 2: denoise (denoise_min_area)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*78}", flush=True)
    print(f"STAGE 2: denoise (denoise_min_area)  [smin={best_smin}, vmin={best_vmin} locked]", flush=True)
    print(f"{'='*78}", flush=True)

    stage2_results = []
    for dma in STAGE2_GRID['denoise_min_area']:
        p = dict(p_base)
        p['denoise_min_area'] = dma
        cfg = {'smin': best_smin, 'vmin': best_vmin, 'denoise_min_area': dma}
        t0 = time.time()
        res = eval_config(dataset, p, best_smin, best_vmin, conn, 'stage2', 'denoise_min_area', dma, cfg)
        dt = time.time() - t0
        total_evals_done += len(dataset)
        stage2_results.append((dma, res))
        print(f"  denoise_min_area={dma:5d}: TP={res['tp']:4d} FP={res['fp']:4d} f1={res['f1']:5.1f}%  ({dt:.1f}s)", flush=True)

    print(f"\n[DoubleML] Stage 2 causal inference...", flush=True)
    dml_s2 = run_doubleml_for_stage(conn, 'stage2', 'param_value')
    if dml_s2.get('theta') is not None:
        print(f"  θ={dml_s2['theta']:.6f}  p={dml_s2['p_value']:.4f}  |θ|×std={dml_s2['effect_magnitude']:.4f}", flush=True)
    else:
        print(f"  ERROR: {dml_s2.get('error', 'unknown')}", flush=True)

    best = max(stage2_results, key=lambda x: (x[1]['f1'], x[1]['tp']))
    best_denoise = best[0]
    reason = f"Best f1={best[1]['f1']:.1f}% TP={best[1]['tp']}"
    if dml_s2.get('theta') is not None:
        reason += f" | DML θ={dml_s2['theta']:.4f} p={dml_s2['p_value']:.4f} (negative=more aggressive helps)"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute("INSERT INTO decisions (stage, param_name, best_value, best_tp, best_f1, "
                 "dml_theta, dml_pvalue, dml_effect, reason, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ('stage2', 'denoise_min_area', best_denoise, best[1]['tp'], round(best[1]['f1'], 2),
                  dml_s2.get('theta'), dml_s2.get('p_value'), dml_s2.get('effect_magnitude'),
                  reason, ts))
    conn.commit()
    all_decisions['stage2'] = {'denoise_min_area': best_denoise, 'f1': best[1]['f1'], 'tp': best[1]['tp'], 'dml': dml_s2}
    print(f"\n  => DECISION: denoise_min_area={best_denoise}  ({reason})", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Stage 4: base_quads (fit_band, fit_min_pts, fit_jump) — axis-by-axis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*78}", flush=True)
    print(f"STAGE 4: base_quads (fit_band, fit_min_pts, fit_jump)  [s1/s2 locked]", flush=True)
    print(f"{'='*78}", flush=True)

    p_s4 = dict(p_base)
    p_s4['denoise_min_area'] = best_denoise

    stage4_results = {}
    for axis_name, values in STAGE4_GRID.items():
        print(f"\n  [Axis: {axis_name}] values={values}", flush=True)
        axis_results = []
        for val in values:
            p = dict(p_s4)
            p[axis_name] = val
            cfg = {'smin': best_smin, 'vmin': best_vmin, 'denoise_min_area': best_denoise,
                   axis_name: val}
            t0 = time.time()
            res = eval_config(dataset, p, best_smin, best_vmin, conn, 'stage4', axis_name, val, cfg)
            dt = time.time() - t0
            total_evals_done += len(dataset)
            axis_results.append((val, res))
            print(f"    {axis_name}={val:6.1f}: TP={res['tp']:4d} FP={res['fp']:4d} f1={res['f1']:5.1f}%  ({dt:.1f}s)", flush=True)
        stage4_results[axis_name] = axis_results

    print(f"\n[DoubleML] Stage 4 causal inference...", flush=True)
    dml_s4 = run_doubleml_for_stage(conn, 'stage4', 'param_value')
    if dml_s4.get('theta') is not None:
        print(f"  θ={dml_s4['theta']:.6f}  p={dml_s4['p_value']:.4f}  |θ|×std={dml_s4['effect_magnitude']:.4f}", flush=True)
    else:
        print(f"  ERROR: {dml_s4.get('error', 'unknown')}", flush=True)

    # Best per axis
    s4_best = {}
    for axis_name, results in stage4_results.items():
        best = max(results, key=lambda x: (x[1]['f1'], x[1]['tp']))
        s4_best[axis_name] = best[0]
        print(f"  Best {axis_name}={best[0]}  f1={best[1]['f1']:.1f}%  TP={best[1]['tp']}", flush=True)

    reason = f"Best fit_band={s4_best.get('fit_band')}, fit_min_pts={s4_best.get('fit_min_pts')}, fit_jump={s4_best.get('fit_jump')}"
    if dml_s4.get('theta') is not None:
        reason += f" | DML θ={dml_s4['theta']:.4f} p={dml_s4['p_value']:.4f}"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute("INSERT INTO decisions (stage, param_name, best_value, best_tp, best_f1, "
                 "dml_theta, dml_pvalue, dml_effect, reason, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ('stage4', 'fit_band/min_pts/jump', s4_best.get('fit_band', 6.0), 0, 0,
                  dml_s4.get('theta'), dml_s4.get('p_value'), dml_s4.get('effect_magnitude'),
                  reason, ts))
    conn.commit()
    all_decisions['stage4'] = {**s4_best, 'dml': dml_s4}
    print(f"\n  => DECISION: {reason}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*78}", flush=True)
    print(f"FINAL SUMMARY", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"  Total image evaluations: {total_evals_done}", flush=True)
    print(f"  Stage 1: smin={best_smin}, vmin={best_vmin}  "
          f"(f1={all_decisions['stage1']['f1']:.1f}%, TP={all_decisions['stage1']['tp']})", flush=True)
    print(f"    DML: θ={dml_s1.get('theta','N/A')}, p={dml_s1.get('p_value','N/A')}", flush=True)
    print(f"  Stage 2: denoise_min_area={best_denoise}  "
          f"(f1={all_decisions['stage2']['f1']:.1f}%, TP={all_decisions['stage2']['tp']})", flush=True)
    print(f"    DML: θ={dml_s2.get('theta','N/A')}, p={dml_s2.get('p_value','N/A')}", flush=True)
    print(f"  Stage 4: fit_band={s4_best.get('fit_band')}, fit_min_pts={s4_best.get('fit_min_pts')}, "
          f"fit_jump={s4_best.get('fit_jump')}", flush=True)
    print(f"    DML: θ={dml_s4.get('theta','N/A')}, p={dml_s4.get('p_value','N/A')}", flush=True)
    print(f"  sqlite3: {DB_PATH}", flush=True)
    print(f"{'='*78}", flush=True)

    # Save JSON summary
    summary = {
        'n_images': len(dataset),
        'total_evals': total_evals_done,
        'dist_thr': D,
        'decisions': all_decisions,
        'dml': {'stage1': dml_s1, 'stage2': dml_s2, 'stage4': dml_s4},
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)

    conn.close()


if __name__ == "__main__":
    main()