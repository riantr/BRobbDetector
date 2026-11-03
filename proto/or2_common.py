"""Shared plumbing for the or2 route panels (schema, CSV write, DoubleML estimate).

Every route writes the SAME schema so a DoWhy/DoubleML analysis can consume any of
them without special-casing:

  unit     "image" | "box"          the unit of analysis
  block    image id                 the cluster key (cluster-robust SE)
  cell     config / arm name        which cell the row came from
  t_*      treatment column(s)      manipulable, >=2 levels, >=10 units each
  y_*      outcome column(s)
  x_*      covariates               PRE-treatment only

`write_panel` emits both `benchmark/or2_<name>.json` (with the causal spec) and
`benchmark/or2_panels/<name>.csv` (the DoWhy/DoubleML ingest format).

`dml_estimate` runs DoubleML PLR with RandomForest nuisance. Cluster-robust SE is
requested via DoubleMLData(cluster_cols=[block]) and, if this DoubleML build does
not support it, the fallback that was actually used is recorded rather than
silently downgraded.

Note: `dowhy` is NOT installed in this environment and MUST NOT be installed —
this module only writes data + spec that `dowhy.CausalModel(...)` can consume.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(ROOT, 'benchmark', 'or2_panels')

REQUIRED = ('unit', 'block', 'cell')

# Cross-fitting seed. DoubleML draws its folds from the global numpy RNG when no
# random_state is supplied (DoubleMLResampling / DoubleMLClusterResampling both
# build KFold(shuffle=True, random_state=None)), so the fold assignment depends on
# process RNG state and on row order. `write_panel` pins the row order and
# `dml_estimate` seeds numpy immediately before constructing the DoubleML object;
# both steps are recorded in every estimate so the numbers are reproducible.
FOLD_SEED = 42
ROW_ORDER = 'sorted by (block, cell), stable'


def write_panel(name, rows, spec, meta=None):
    """rows: list of dicts following the schema. Returns the json path."""
    os.makedirs(PANEL_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in REQUIRED:
        if c not in df.columns:
            raise KeyError('panel %s missing required column %r' % (name, c))
    # deterministic row order: DoubleML's cross-fitting folds are positional
    if 'block' in df.columns and 'cell' in df.columns:
        df = df.sort_values(['block', 'cell'], kind='stable').reset_index(drop=True)
    csv_path = os.path.join(PANEL_DIR, name + '.csv')
    df.to_csv(csv_path, index=False)
    # normalize the causal spec so every panel carries the same mandatory fields
    spec = dict(spec)
    spec.setdefault('instruments', [])
    spec.setdefault('mediators', [])
    spec.setdefault('covariates_pre_treatment', True)
    spec.setdefault('covariate_note',
                    'all x_* are pre-treatment image/box properties measured before the '
                    'arm assignment; none is derived from a y_*')
    out = {
        'panel': name,
        'schema': {c: str(df[c].dtype) for c in df.columns},
        'n_rows': int(len(df)),
        'unit': str(df['unit'].iloc[0]) if len(df) else spec.get('unit'),
        'n_blocks': int(df['block'].nunique()),
        'treatments': [c for c in df.columns if c.startswith('t_')],
        'outcomes': [c for c in df.columns if c.startswith('y_')],
        'covariates': [c for c in df.columns if c.startswith('x_')],
        'causal_spec': spec,
        'meta': meta or {},
        'csv': os.path.relpath(csv_path, ROOT).replace('\\', '/'),
    }
    json_path = os.path.join(ROOT, 'benchmark', 'or2_%s.json' % name)
    json.dump(out, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return json_path, csv_path


def _make(clf_seed=42):
    """Cheap-but-deterministic nuisance. Kept small so a 48k-row factorial panel
    is estimable on CPU; the exact settings are recorded in every estimate."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=60, min_samples_leaf=10,
                                 max_features=0.5, random_state=clf_seed, n_jobs=1)


NUISANCE = 'RandomForestRegressor(60 trees, min_samples_leaf=10, max_features=0.5, seed=42)'


def dml_estimate(csv_path, treatment, outcome, covariates, n_folds=3):
    """DoubleML PLR, clustered on `block`. Returns a dict; never raises upward."""
    from doubleml import DoubleMLData, DoubleMLPLR
    res = {'treatment': treatment, 'outcome': outcome,
           'covariates': list(covariates), 'n_folds': n_folds,
           'nuisance': NUISANCE,
           'library': 'doubleml'}
    try:
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[treatment, outcome, 'block'] + list(covariates))
        if df[treatment].nunique() < 2:
            res['status'] = 'UNIDENTIFIABLE'
            res['reason'] = 'treatment has < 2 levels after dropna'
            return res
        res['n_units'] = int(len(df))
        res['n_blocks'] = int(df['block'].nunique())
        res['fold_seed'] = FOLD_SEED
        res['row_order'] = ROW_ORDER
        used_cluster = False
        try:
            dml_data = DoubleMLData(df, y_col=outcome, d_cols=treatment,
                                    x_cols=list(covariates),
                                    cluster_cols='block')
            used_cluster = True
        except TypeError:
            dml_data = DoubleMLData(df, y_col=outcome, d_cols=treatment,
                                    x_cols=list(covariates))
        # Pin the fold draw: DoubleML constructs its KFold(shuffle=True) inside
        # __init__, so the global numpy seed at this exact point determines the
        # partition. Nothing else consumes RNG between here and construction.
        np.random.seed(FOLD_SEED)
        dml = DoubleMLPLR(dml_data, _make(), _make(), n_folds=n_folds,
                          n_rep=1, score='partialling out')
        dml.fit()
        def _first(x):
            return float(np.asarray(x, dtype=float).ravel()[0])
        ci = np.asarray(dml.confint(), dtype=float)
        ci_flat = ci.ravel()
        th = _first(dml.coef)
        se = _first(dml.se)
        res.update({
            'status': 'ok',
            'theta': round(th, 5),
            'se': round(se, 5),
            'ci95_low': round(float(ci_flat[0]), 5),
            'ci95_high': round(float(ci_flat[1]), 5),
            'p_value': _first(dml.pval),
            'cluster_robust_se': used_cluster,
            'sample_splitting': ('DoubleML folds redrawn after np.random.seed(%d); '
                                 'rows %s' % (FOLD_SEED, ROW_ORDER)),
            'se_note': ('cluster-robust on block' if used_cluster else
                        'DoubleML build has no cluster_cols; plain PLR SE reported '
                        'and block count recorded'),
        })
    except Exception as e:                     # never let estimation break a route
        res['status'] = 'ESTIMATION_FAILED'
        res['reason'] = '%s: %s' % (type(e).__name__, str(e)[:200])
    return res


def record_unidentifiable(name, treatment, outcome, reason, evidence=None):
    """Record a NOT-ATTEMPTED / UNIDENTIFIABLE arm WITHOUT a theta.

    The arm must also be a non-treatment column (`ex_*`) so no downstream consumer
    can pick it up as a `t_*`; this function additionally removes any estimate
    accidentally stored under the same key.
    """
    json_path = os.path.join(ROOT, 'benchmark', 'or2_%s.json' % name)
    d = json.load(open(json_path, encoding='utf-8'))
    entry = {'arm': treatment, 'outcome': outcome,
             'status': 'UNIDENTIFIABLE', 'interpretation': 'NOT-ATTEMPTED',
             'reason': reason}
    if evidence:
        entry['evidence'] = evidence
    d.setdefault('not_attempted', []).append(entry)
    d.get('dml_estimates', {}).pop(treatment, None)
    json.dump(d, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('  [DML] %-14s %-10s UNIDENTIFIABLE (NOT-ATTEMPTED, ex_* arm): %s'
          % (name, treatment, reason), flush=True)
    return entry


def attach_estimates(name, treatment, outcome, covariates):
    """Estimate and write the result back into benchmark/or2_<name>.json.

    Accumulates into `dml_estimates[treatment]` so a route with several treatments
    keeps ALL of them; `dml` mirrors the first (primary) treatment.
    """
    csv_path = os.path.join(PANEL_DIR, name + '.csv')
    est = dml_estimate(csv_path, treatment, outcome, covariates)
    json_path = os.path.join(ROOT, 'benchmark', 'or2_%s.json' % name)
    d = json.load(open(json_path, encoding='utf-8'))
    # key by treatment, but disambiguate when the SAME treatment is estimated
    # against a second outcome (otherwise the later one silently overwrites)
    existing = d.setdefault('dml_estimates', {})
    key = treatment
    if treatment in existing and existing[treatment].get('outcome') != outcome:
        key = '%s::%s' % (treatment, outcome)
    existing[key] = est
    if 'dml' not in d or d.get('dml', {}).get('treatment') == treatment:
        d['dml'] = d['dml_estimates'][d['primary_treatment']] if 'primary_treatment' in d \
            else d['dml_estimates'][sorted(d['dml_estimates'])[0]]
    json.dump(d, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if est.get('status') == 'ok':
        print('  [DML] %-14s %-10s theta %+0.4f  se %.4f  95%%CI [%+0.4f, %+0.4f]  p=%.4g  n=%d  clusters=%d'
              % (name, treatment, est['theta'], est['se'], est['ci95_low'],
                 est['ci95_high'], est['p_value'], est['n_units'], est['n_blocks']),
              flush=True)
    else:
        print('  [DML] %-14s %-10s %s %s' % (name, treatment, est['status'],
                                            est.get('reason', '')), flush=True)
    return est