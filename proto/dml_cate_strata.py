"""Stage 2 (goal): stratified treatment effects (CATE) on the EXISTING 2^4 panel.

The earlier DoubleML verdicts were estimated on the whole 500-image panel, and
LOOP-24/28 showed that aggregate effects can hide stratum sign-cancellation
(p=0.005 per image, flat in aggregate).  This re-reads the SAME panel
(`benchmark/dml_panel_real.json`, 500 images x 16 cells, no new detection runs)
and re-estimates each of the four knobs INSIDE box-level strata:

  gt_min_side_mean  <= 30 px / <= 50 px   (the measured 47% / 23% miss bands)
  gt_aspect_mean    >= 6                  (the measured 38% miss band; derived
                                           as area / min_side^2 from the panel)
  n_gt              >= 3                   (the dense-cluster regime: 18 images
                                           with >=3 FN carry 92 of the 175 FN)
  n_holes           > median              (local red-complexity = density proxy)

Estimator: the exact within-image paired contrast (every image appears in all
16 cells, so the panel is a complete block) - the same estimator
`proto/dml_analyze.py::paired_effect` uses, restricted to the stratum.  SE is
the standard error of the per-image paired differences; p is the normal
approximation.  No model fitting, no new detection runs.

Usage: python proto/dml_cate_strata.py
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dml_analyze import load_panel, paired_effect, paired_conditional

PANEL = 'benchmark/dml_panel_real.json'
OUT = 'benchmark/dml_cate_strata.json'

FACTORS = [('t_minside', 'hole_min_side', '8 -> 14'),
           ('t_fitpass', 'hole_fit_pass', '0 -> 2'),
           ('t_refine', 'hole_refine', '0 -> 1'),
           ('t_sup60', 'min_red_support', '0.50 -> 0.60')]
OUTCOMES = ['tp5', 'fp5', 'tp3']


def pval(th, se):
    if not se or se <= 0 or not np.isfinite(th):
        return None
    z = abs(th / se)
    return float(math.erfc(z / math.sqrt(2.0)))


def main():
    data, df, _covs = load_panel(PANEL)

    # derived per-image aspect (panel has area_mean + min_side_mean, not aspect)
    df['gt_aspect_mean'] = df['gt_area_mean'] / np.maximum(df['gt_min_side_mean'] ** 2, 1e-6)

    med_holes = float(df.loc[df['cell'] == data['cells'][0]['name'], 'n_holes'].median())

    strata = {
        'size_le30':      (df['gt_min_side_mean'] <= 30).astype(int),
        'size_le50':      (df['gt_min_side_mean'] <= 50).astype(int),
        'thin_aspect_ge6': (df['gt_aspect_mean'] >= 6).astype(int),
        'thin_aspect_ge25': (df['gt_aspect_mean'] >= 2.5).astype(int),
        'dense_gt_ge3':   (df['n_gt'] >= 3).astype(int),
        'dense_holes_hi': (df['n_holes'] > med_holes).astype(int),
    }
    for k, v in strata.items():
        df['z_' + k] = v

    # stratum populations (one row per image x cell -> use one cell)
    first_cell = df[df['cell'] == data['cells'][0]['name']]
    pop = {}
    for k in strata:
        s = first_cell
        pop[k] = {
            'n_images_in': int((s['z_' + k] == 1).sum()),
            'n_images_out': int((s['z_' + k] == 0).sum()),
            'n_gt_in': int(s.loc[s['z_' + k] == 1, 'n_gt'].sum()),
            'n_gt_out': int(s.loc[s['z_' + k] == 0, 'n_gt'].sum()),
        }

    res = {'panel': PANEL, 'n_images': data['n_images'], 'n_gt': data['n_gt'],
           'outcomes': OUTCOMES, 'strata_population': pop,
           'median_n_holes': med_holes, 'factors': {}, 'aggregate': {}}

    print('median n_holes = %.1f' % med_holes)
    print('\n%-12s %-16s %-6s %8s %8s %5s %7s' %
          ('factor', 'stratum', 'inside', 'theta', 'se', 'n', 'p'))
    for tcol, key, lv in FACTORS:
        res['factors'][key] = {}
        agg = {}
        for y in OUTCOMES:
            th, se, n = paired_effect(df, y, tcol)
            agg[y] = {'theta': round(th, 3), 'se': round(se, 3), 'n': n,
                      'p': (round(pval(th, se), 4) if pval(th, se) is not None else None)}
        res['aggregate'][key] = agg
        print('%-12s %-16s %-6s %8.3f %8.3f %5d %7s' %
              (key, '(AGGREGATE)', '-', agg['tp5']['theta'], agg['tp5']['se'],
               agg['tp5']['n'], agg['tp5']['p']))
        for sk in strata:
            res['factors'][key][sk] = {}
            for bv in (1, 0):
                for y in OUTCOMES:
                    try:
                        th, se, n = paired_conditional(df, y, tcol, 'z_' + sk, bv)
                    except Exception as e:
                        res['factors'][key][sk]['%s|%d' % (y, bv)] = {'error': str(e)[:100]}
                        continue
                    res['factors'][key][sk]['%s|%d' % (y, bv)] = {
                        'theta': round(th, 3), 'se': round(se, 3), 'n': n,
                        'p': (round(pval(th, se), 4) if pval(th, se) is not None else None)}
                    if y == 'tp5':
                        print('%-12s %-16s %-6d %8.3f %8.3f %5d %7s' %
                              (key, sk, bv, th, se, n,
                               res['factors'][key][sk]['tp5|%d' % bv]['p']))
            print()

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nsaved %s' % OUT)

    # headline: which knob has an OPPOSITE-SIGN tp5 effect in a stratum?
    print('\n== sign flips on tp5 (stratum effect opposite to the aggregate) ==')
    flips = 0
    for tcol, key, lv in FACTORS:
        agg = res['aggregate'][key]['tp5']['theta']
        for sk in strata:
            for bv in (1, 0):
                e = res['factors'][key][sk].get('tp5|%d' % bv, {})
                if 'theta' not in e:
                    continue
                th = e['theta']
                if agg != 0 and th * agg < 0 and abs(th) >= 1.0:
                    flips += 1
                    print('  %-12s %-16s=%d  theta %+0.2f (aggregate %+0.2f) p=%s n=%d'
                          % (key, sk, bv, th, agg, e.get('p'), e.get('n')))
    if not flips:
        print('  none with |theta| >= 1.0')


if __name__ == '__main__':
    main()