"""Validate every benchmark/or2_panels/*.csv against the mandatory uniform schema.

Checks per panel (Done-when #2):
  1. required columns present: unit, block, cell, >=1 t_*, >=1 y_*, >=1 x_*
  2. no NaN anywhere in the required/treatment/outcome columns
  3. every treatment has >= 2 distinct levels AND every level has >= 10 units
  4. `block` is populated (no empty/null) and has > 1 distinct value
  5. no covariate is derived from an outcome: a covariate whose absolute
     correlation with any outcome is exactly 1.0 is flagged as leakage
  6. one and only one `unit` value per panel

Prints a PASS/FAIL table and exits non-zero if anything fails.

Usage: python proto/or2_validate_panels.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(ROOT, 'benchmark', 'or2_panels')


def check_panel(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return name, False, ['unreadable: %s' % e], {}
    problems = []

    req = ['unit', 'block', 'cell']
    missing_req = [c for c in req if c not in df.columns]
    if missing_req:
        problems.append('missing required columns %s' % missing_req)
    tcols = [c for c in df.columns if c.startswith('t_')]
    ycols = [c for c in df.columns if c.startswith('y_')]
    xcols = [c for c in df.columns if c.startswith('x_')]
    if not tcols:
        problems.append('no t_* treatment column')
    if not ycols:
        problems.append('no y_* outcome column')
    if not xcols:
        problems.append('no x_* covariate column')

    # 1/2 no NaN in the structural columns
    check_cols = [c for c in req + tcols + ycols if c in df.columns]
    nan_cols = [c for c in check_cols if df[c].isna().any()]
    if nan_cols:
        problems.append('NaN in %s' % nan_cols)

    # 3 treatment levels
    lvl_info = {}
    for t in tcols:
        vc = df[t].value_counts()
        lvl_info[t] = int(df[t].nunique())
        if df[t].nunique() < 2:
            problems.append('%s has < 2 levels' % t)
        small = vc[vc < 10]
        if len(small):
            problems.append('%s levels with <10 units: %s' %
                            (t, {str(k): int(v) for k, v in small.items()}))

    # 4 block populated
    if 'block' in df.columns:
        if df['block'].isna().any():
            problems.append('block has nulls')
        if df['block'].nunique() < 2:
            problems.append('block has < 2 distinct values')

    # 5 leakage: covariate perfectly correlated with an outcome
    for x in xcols:
        for y in ycols:
            try:
                a = pd.to_numeric(df[x], errors='coerce')
                b = pd.to_numeric(df[y], errors='coerce')
                if a.nunique() < 2 or b.nunique() < 2:
                    continue
                c = float(np.corrcoef(a, b)[0, 1])
                if abs(c) > 0.9999:
                    problems.append('covariate %s is a copy of outcome %s (r=%.4f)'
                                    % (x, y, c))
            except Exception:
                pass

    # 6 one unit value
    if 'unit' in df.columns:
        if df['unit'].nunique() != 1:
            problems.append('mixed unit values %s' % list(df['unit'].unique()))

    info = {'rows': int(len(df)), 'blocks': int(df['block'].nunique())
            if 'block' in df.columns else 0,
            'treatments': len(tcols), 'outcomes': len(ycols),
            'covariates': len(xcols),
            'unit': str(df['unit'].iloc[0]) if 'unit' in df.columns and len(df) else '?'}
    return name, len(problems) == 0, problems, info


def main():
    paths = sorted(glob.glob(os.path.join(PANEL_DIR, '*.csv')))
    if not paths:
        print('NO PANELS FOUND in %s' % PANEL_DIR)
        sys.exit(1)
    print('%-24s %-5s %6s %7s %3s %3s %3s %-6s %s'
          % ('panel', 'PASS', 'rows', 'blocks', 't', 'y', 'x', 'unit', 'problems'))
    bad = 0
    for p in paths:
        name, ok, problems, info = check_panel(p)
        if not ok:
            bad += 1
        print('%-24s %-5s %6d %7d %3d %3d %3d %-6s %s'
              % (name, 'PASS' if ok else 'FAIL', info.get('rows', 0),
                 info.get('blocks', 0), info.get('treatments', 0),
                 info.get('outcomes', 0), info.get('covariates', 0),
                 info.get('unit', '?'), '; '.join(problems) if problems else '-'))
    print('\n%d panels, %d PASS, %d FAIL' % (len(paths), len(paths) - bad, bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()