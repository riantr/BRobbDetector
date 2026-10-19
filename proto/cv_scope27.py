"""LOOP-27 (Path-2): Synthetic data realism — fragmented OBB stroke.

LOOP-25 found all DoubleML treatment effects ≈0 on clean synthetic data.
LOOP-27 first attempt (speck noise near OBB) also found theta≈0 — the
synthetic OBB is too clean, creating a single large connected component
that denoise_min_area can't affect.

Root cause: on real data, the hand-drawn red line has varying intensity,
and the red mask splits it into multiple fragments. denoise_min_area
determines which fragments survive, directly affecting TP.

Method:
  1. Draw OBB with varying red intensity (some pixels below HSV threshold)
  2. This fragments the red mask into multiple small components
  3. Add background red specks (separate small components) at varying density
  4. Sweep denoise_min_area — now it should matter because:
     - Too low: noise components survive, create false contours → FP
     - Too high: OBB fragments removed, box not detected → FN
     - Optimal: noise removed, OBB fragments survive → max TP
  5. Run DoubleML to measure the causal effect

Output:
  benchmark/cv_scope27.json
  benchmark/cv_scope27.db
"""
import sys, os, json, time, sqlite3, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise, _side_ok, _quad_adaptive
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, ROOT,
                       _edgefit_corners)
from cv_scope4 import HY_ORDER
from cv_scope7 import pm5_params
from cv_scope13 import obb_corners, compute_gt_minarearect
from eval_pointmatch import match_boxes_point
from quad_ortho import is_orthogonal_quad

DB_PATH = os.path.join(ROOT, "benchmark", "cv_scope27.db")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope27.json")
BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
D = 5.0
SEED = 42

FRAG_LEVELS = [0.0, 0.05, 0.10, 0.15]
BG_NOISE_LEVELS = [50]
DENOISE_GRID = [0, 20, 40, 60, 80, 100, 150, 200]
N_BG = 30
N_PER_BG = 4
BROKENNESS_VALUES = [0.0, 0.10, 0.20]


def draw_fragmented_obb(img, corners, frag_level, stroke_w=2, rng=None):
    """Draw OBB with varying red intensity — some pixels below HSV threshold.

    frag_level: fraction of edge pixels that get reduced intensity (below vmin=110).
    This fragments the red mask into multiple small components.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    n = 4
    for i in range(n):
        p1 = corners[i].astype(int)
        p2 = corners[(i + 1) % n].astype(int)
        length = int(np.linalg.norm(p2 - p1))
        for t in range(length + 1):
            s = t / max(1, length)
            x = int(p1[0] + s * (p2[0] - p1[0]))
            y = int(p1[1] + s * (p2[1] - p1[1]))
            if rng.random() < frag_level:
                val = rng.randint(95, 109)
                sat = rng.randint(70, 79)
                color = (0, 0, val)
            else:
                color = (0, 0, 255)
            cv2.circle(img, (x, y), stroke_w, color, -1)


def add_bg_red_specks(img, n_specks, rng):
    """Add small separate red components scattered across the image."""
    H, W = img.shape[:2]
    for _ in range(n_specks):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        r = rng.randint(2, 7)
        val = rng.randint(110, 256)
        sat = rng.randint(80, 256)
        cv2.circle(img, (cx, cy), r, (0, 0, val), -1)
    return img


def gen_dataset(frag_level, bg_noise, rng):
    """Generate synthetic images with fragmented OBB + background noise."""
    bg_files = sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))[:N_BG]
    dataset = []
    for bi, bg_file in enumerate(bg_files):
        bg = cv2.imread(bg_file)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        for brokenness in BROKENNESS_VALUES:
            for k in range(N_PER_BG):
                cx = int(rng.randint(80, max(81, W - 80)))
                cy = int(rng.randint(60, max(61, H - 60)))
                img = bg.copy()
                corners = obb_corners(cx, cy, 80, 50, 15)
                draw_fragmented_obb(img, corners, frag_level, stroke_w=2, rng=rng)
                gt = compute_gt_minarearect(img, corners, padding=10)
                if bg_noise > 0:
                    img = add_bg_red_specks(img, bg_noise, rng)
                dataset.append({'img': img, 'gt_quads': [gt],
                                'brokenness': brokenness, 'bg_idx': bi})
    return dataset


def run_pipeline(img, p):
    H, W = img.shape[:2]
    total_px = H * W
    f = {}

    mask_raw = red_mask(img, smin=p.get('smin', 80), vmin=p.get('vmin', 110))
    n_red = int(np.count_nonzero(mask_raw))
    n_comp_raw, _ = cv2.connectedComponents(mask_raw, connectivity=8)
    n_comp_raw -= 1
    f['s1_red_ratio'] = n_red / total_px
    f['s1_n_components'] = n_comp_raw

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

    cnts = outer_contours(mask, p)
    f['s3_n_contours'] = len(cnts)

    n_total = 0
    all_cands = []
    for c in cnts:
        qh = _quad_adaptive(cv2.convexHull(c), p["approx_eps_list"])
        qe = _edgefit_corners(c, p)
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        qm = cv2.boxPoints(cv2.minAreaRect(c)) if rw >= 1 and rh >= 1 else None
        for q in [qh, qe, qm]:
            if q is not None and len(q) == 4:
                all_cands.append(q)
                n_total += 1
    f['s4_n_total'] = n_total

    passed = []
    for q in all_cands:
        if q is None or not _side_ok(q, p):
            continue
        if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                                   boundary_band=p["boundary_band"]):
            continue
        passed.append(q.astype(np.int32))
    f['s5_pass_rate'] = len(passed) / max(1, len(all_cands))

    try:
        final = dedup_keep_order(passed, p['dedup_iou'])
    except cv2.error:
        final = dedup_keep_order([q for q in passed if q is not None and len(q) == 4],
                                  p['dedup_iou'])
    f['s6_dedup_rate'] = (len(passed) - len(final)) / max(1, len(passed))

    return final, f


def init_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trials (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        frag_level REAL, bg_noise INTEGER, param_value REAL,
        tp INTEGER, fp INTEGER, fn INTEGER, recall REAL, f1 REAL, timestamp TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS panel (
        trial_id INTEGER, img_idx INTEGER,
        s1_red_ratio REAL, s1_n_components INTEGER,
        s2_kept_area_ratio REAL, s2_removed_ratio REAL,
        s3_n_contours INTEGER, s4_n_total INTEGER,
        s5_pass_rate REAL, s6_dedup_rate REAL,
        tp INTEGER, n_gt INTEGER
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS dml_results (
        frag_level REAL, bg_noise INTEGER, theta REAL, p_value REAL,
        effect_magnitude REAL, n_samples INTEGER, timestamp TEXT
    )""")
    conn.commit()
    return conn


def run_doubleml(conn, frag_level, bg_noise):
    import pandas as pd
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor

    c = conn.cursor()
    c.execute("""SELECT t.trial_id, t.param_value,
                        p.s1_red_ratio, p.s1_n_components,
                        p.s2_kept_area_ratio, p.s2_removed_ratio,
                        p.s3_n_contours, p.s4_n_total,
                        p.s5_pass_rate, p.s6_dedup_rate,
                        p.tp, p.n_gt
                 FROM trials t JOIN panel p ON t.trial_id = p.trial_id
                 WHERE t.frag_level = ? AND t.bg_noise = ?""",
              (frag_level, bg_noise))
    rows = c.fetchall()
    if len(rows) < 50:
        return {'theta': None, 'error': f'insufficient data ({len(rows)})'}

    cols = ['trial_id', 'param_value', 's1_red_ratio', 's1_n_components',
            's2_kept_area_ratio', 's2_removed_ratio',
            's3_n_contours', 's4_n_total',
            's5_pass_rate', 's6_dedup_rate', 'tp', 'n_gt']
    df = pd.DataFrame(rows, columns=cols)

    Y = 'tp'
    D_col = 'param_value'
    controls = [c for c in cols if c not in (Y, D_col, 'trial_id', 'n_gt')]
    controls = [c for c in controls if df[c].std() > 0]

    df_sub = df[[Y, D_col] + controls].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_sub) < 50:
        return {'theta': None, 'error': f'insufficient clean data ({len(df_sub)})'}

    try:
        dml_data = DoubleMLData(df_sub, y_col=Y, d_cols=D_col, x_cols=controls)
        ml_l = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        ml_m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
        dml_plr.fit()
        theta = float(dml_plr.coef[0])
        pval = float(dml_plr.pval[0])
        std_d = float(df_sub[D_col].std())
        effect_mag = abs(theta) * std_d
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        c.execute("INSERT INTO dml_results VALUES (?,?,?,?,?,?,?)",
                  (frag_level, bg_noise, round(theta, 6), round(pval, 6),
                   round(effect_mag, 6), len(df_sub), ts))
        conn.commit()
        return {'theta': theta, 'p_value': pval, 'effect_magnitude': effect_mag,
                'n_samples': len(df_sub)}
    except Exception as e:
        return {'theta': None, 'error': str(e)[:200]}


def main():
    print(f"LOOP-27: Synthetic data realism — fragmented OBB stroke", flush=True)
    print(f"  Fragment levels: {FRAG_LEVELS}", flush=True)
    print(f"  BG noise levels: {BG_NOISE_LEVELS}", flush=True)
    print(f"  Denoise grid: {DENOISE_GRID}", flush=True)
    n_imgs = N_BG * N_PER_BG * len(BROKENNESS_VALUES)
    print(f"  Images per condition: {n_imgs}", flush=True)

    conn = init_db(DB_PATH)
    p = pm5_params()

    all_results = {}

    for frag_level in FRAG_LEVELS:
        for bg_noise in BG_NOISE_LEVELS:
            cond_key = f"frag{frag_level:.2f}_bg{bg_noise}"
            print(f"\n--- frag={frag_level:.2f}, bg_noise={bg_noise} ---", flush=True)

            rng_ds = np.random.RandomState(SEED + int(frag_level * 100) + bg_noise)
            dataset = gen_dataset(frag_level, bg_noise, rng_ds)
            print(f"  {len(dataset)} images", flush=True)

            cond_results = {}
            for dma in DENOISE_GRID:
                p['denoise_min_area'] = dma
                TP = FP = FN = 0
                panel_rows = []
                for idx, item in enumerate(dataset):
                    final, f = run_pipeline(item['img'], p)
                    det = [q.astype(np.int32) for q in final]
                    gt = [g.astype(np.int32) for g in item['gt_quads']]
                    tp, unmatched = match_boxes_point(det, gt, D)
                    TP += tp; FP += len(unmatched); FN += len(gt) - tp
                    panel_rows.append((
                        f['s1_red_ratio'], f['s1_n_components'],
                        f['s2_kept_area_ratio'], f['s2_removed_ratio'],
                        f['s3_n_contours'], f['s4_n_total'],
                        f['s5_pass_rate'], f['s6_dedup_rate'],
                        tp, len(gt),
                    ))

                n_gt = TP + FN
                recall = TP / max(1, n_gt) * 100
                prec = TP / max(1, TP + FP) * 100
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                c = conn.cursor()
                c.execute("INSERT INTO trials (frag_level, bg_noise, param_value, tp, fp, fn, "
                          "recall, f1, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
                          (frag_level, bg_noise, float(dma), TP, FP, FN,
                           round(recall, 2), round(f1, 2), ts))
                trial_id = c.lastrowid
                c.executemany("INSERT INTO panel (trial_id, img_idx, "
                              "s1_red_ratio, s1_n_components, s2_kept_area_ratio, s2_removed_ratio, "
                              "s3_n_contours, s4_n_total, s5_pass_rate, s6_dedup_rate, tp, n_gt) "
                              "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                              [(trial_id, idx) + row for idx, row in enumerate(panel_rows)])
                conn.commit()

                cond_results[dma] = {'tp': TP, 'fp': FP, 'fn': FN,
                                      'recall': round(recall, 1), 'f1': round(f1, 1)}

            best_dma = max(cond_results, key=lambda k: cond_results[k]['tp'])
            best_tp = cond_results[best_dma]['tp']
            dma0_tp = cond_results[0]['tp']
            dma200_tp = cond_results[200]['tp']
            spread = best_tp - min(cond_results[k]['tp'] for k in cond_results)
            print(f"  TP range: {min(cond_results[k]['tp'] for k in cond_results)}-{max(cond_results[k]['tp'] for k in cond_results)} "
                  f"(spread={spread}), best dma={best_dma} TP={best_tp}", flush=True)

            dml = run_doubleml(conn, frag_level, bg_noise)
            if dml.get('theta') is not None:
                print(f"  DML: theta={dml['theta']:.6f} p={dml['p_value']:.4f} "
                      f"effect={dml['effect_magnitude']:.6f}", flush=True)
            else:
                print(f"  DML: FAILED ({dml.get('error', '?')})", flush=True)

            all_results[cond_key] = {
                'frag_level': frag_level, 'bg_noise': bg_noise,
                'denoise_sweep': cond_results, 'dml': dml,
                'tp_spread': spread,
            }

    # Summary
    print(f"\n{'='*90}")
    print(f"LOOP-27 SUMMARY: fragmented OBB + bg noise")
    print(f"{'='*90}")
    print(f"\n{'frag':>5s} {'bg':>4s}  {'TP@0':>5s} {'TP@40':>6s} {'TP@200':>7s}  "
          f"{'spread':>6s}  {'theta':>10s}  {'p_value':>8s}  {'effect':>10s}")
    for frag_level in FRAG_LEVELS:
        for bg_noise in BG_NOISE_LEVELS:
            key = f"frag{frag_level:.2f}_bg{bg_noise}"
            r = all_results[key]
            sweep = r['denoise_sweep']
            dml = r['dml']
            theta = dml.get('theta', None)
            pval = dml.get('p_value', None)
            eff = dml.get('effect_magnitude', None)
            theta_s = f"{theta:.6f}" if theta is not None else "N/A"
            pval_s = f"{pval:.4f}" if pval is not None else "N/A"
            eff_s = f"{eff:.6f}" if eff is not None else "N/A"
            print(f"{frag_level:>5.2f} {bg_noise:>4d}  "
                  f"{sweep[0]['tp']:>5d} {sweep[40]['tp']:>6d} {sweep[200]['tp']:>7d}  "
                  f"{r['tp_spread']:>6d}  {theta_s:>10s}  {pval_s:>8s}  {eff_s:>10s}")
    print(f"{'='*90}")

    results = {
        'frag_levels': FRAG_LEVELS,
        'bg_noise_levels': BG_NOISE_LEVELS,
        'denoise_grid': DENOISE_GRID,
        'n_images_per_cond': n_imgs,
        'results': all_results,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nSaved: {RESULT_PATH}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()