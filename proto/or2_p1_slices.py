"""P1 route slices (or2): give P1-a..d each their OWN panel JSON + CSV.

The goal mandates ONE complete-block factorial panel for the orthogonality routes
(`p1_ortho`, 96 cells x 500 images), but Done-when #3 also requires that every
route has "a panel JSON + a CSV + a harness". This derives, from the SAME measured
factorial, a one-factor-at-a-time slice per route:

  P1-a  ortho mode   {none, ortho_edge, ortho_blend}  at tol=10, band=0, joint=0
  P1-b  ortho tol    {6, 10, 15, 20}                  at edge,  band=0, joint=0
  P1-c  edge band    {0, 1, 2, 4}                     at edge,  tol=10, joint=0
  P1-d  joint fit    {0, 1}                           at edge,  tol=10, band=0

No detection is re-run: every slice is a row subset of `or2_panels/p1_ortho.csv`.
Constant treatment columns are dropped (a 1-level column is unidentifiable and
would fail the >=10-units-per-level rule).

Usage: python proto/or2_p1_slices.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or2_common as C

SRC = os.path.join(C.PANEL_DIR, 'p1_ortho.csv')
XCOLS = ['x_n_gt', 'x_img_w', 'x_img_h', 'x_red_ratio', 'x_n_holes',
         'x_gt_min_side_mean']

SLICES = {
    'p1_a': {'title': 'P1-a ortho mode', 'treat': ['t_edge', 't_blend'],
             'keep': lambda c: c['t_tol'] == 10 and c['t_band'] == 0 and c['t_joint'] == 0,
             'unit': 'image',
             'assumption': ('The enforcement MODE is assigned by the experimenter with '
                            'tolerance/band/joint held fixed, so its effect on TP is '
                            'identified conditionally on pre-treatment image properties.')},
    'p1_b': {'title': 'P1-b ortho tolerance', 'treat': ['t_tol'],
             'keep': lambda c: c['t_edge'] == 1 and c['t_band'] == 0 and c['t_joint'] == 0,
             'unit': 'image',
             'assumption': ('The tolerance is assigned by the experimenter with the '
                            'shipped mode/band/joint held fixed.')},
    'p1_c': {'title': 'P1-c edge band', 'treat': ['t_band'],
             'keep': lambda c: c['t_edge'] == 1 and c['t_tol'] == 10 and c['t_joint'] == 0,
             'unit': 'image',
             'assumption': ('The edge band is assigned by the experimenter with the '
                            'shipped mode/tolerance/joint held fixed. NOTE the GT is '
                            'filtered at band 0, so band>0 is a GT/detection rule '
                            'mismatch by construction.')},
    'p1_d': {'title': 'P1-d joint exactly-orthogonal fit', 'treat': ['t_joint'],
             'keep': lambda c: c['t_edge'] == 1 and c['t_tol'] == 10 and c['t_band'] == 0,
             'unit': 'image',
             'assumption': ('Joint-fit on/off is assigned by the experimenter with the '
                            'shipped mode/tolerance/band held fixed.')},
}


def parse_cell(name):
    parts = name.split('-')
    d = {}
    for p in parts:
        if p.startswith('tol'):
            d['t_tol'] = int(p[3:])
        elif p.startswith('b'):
            d['t_band'] = int(p[1:])
        elif p.startswith('j'):
            d['t_joint'] = int(p[1:])
        else:
            d['mode'] = p
    return d


def main(only=None):
    rows_all = list(csv.DictReader(open(SRC, encoding='utf-8')))
    ycols = [k for k in rows_all[0] if k.startswith('y_')] if rows_all else []
    src_meta = json.load(open(os.path.join(C.ROOT, 'benchmark', 'or2_p1_ortho.json'),
                              encoding='utf-8'))['meta']
    bt = src_meta.get('base_cell_totals', {})
    frozen_tp = {'3': 857, '5': 901, '8': 926, '12': 938}
    repro = {d: bt.get('tp%s' % d) == v for d, v in frozen_tp.items()}
    print('source rows:', len(rows_all), ' outcomes:', ycols)
    for name, spec in SLICES.items():
        if only and name != only:
            continue
        out = []
        for r in rows_all:
            cel = parse_cell(r['cell'])
            c = {'t_tol': cel.get('t_tol'), 't_band': cel.get('t_band'),
                 't_joint': cel.get('t_joint'),
                 't_edge': 1.0 if cel.get('mode') == 'ortho_edge' else 0.0,
                 't_blend': 1.0 if cel.get('mode') == 'ortho_blend' else 0.0}
            if not spec['keep'](c):
                continue
            row = {'unit': spec['unit'], 'block': r['block'], 'cell': r['cell']}
            for t in spec['treat']:
                row[t] = float(c[t])
            for y in ycols:
                row[y] = float(r[y])
            for x in XCOLS:
                row[x] = float(r[x])
            out.append(row)
        sp = {'unit': spec['unit'], 'outcome': 'y_tp5',
              'treatment': spec['treat'][0],
              'common_causes': [x for x in XCOLS] + [t for t in spec['treat'][1:]],
              'assumption': spec['assumption'],
              'dowhy_usage': ("dowhy.CausalModel(data=df, treatment='%s', outcome='y_tp5', "
                              "common_causes=%r)" % (spec['treat'][0],
                                                     [x for x in XCOLS] + spec['treat'][1:]))}
        j, cs = C.write_panel(name, out, sp,
                              meta={'derived_from': 'p1_ortho (row subset, no re-run)',
                                    'title': spec['title'],
                                    'base_cell': 'ortho_edge-tol10-b0-j0',
                                    'base_cell_totals': bt,
                                    'base_cell_corrected': src_meta.get('base_cell_corrected'),
                                    'base_cell_reproduces_frozen': repro,
                                    'base_cell_full_match': bool(src_meta.get('base_cell_match')),
                                    'base_cell_match': all(repro.values())
                                    and bool(src_meta.get('base_cell_match')),
                                    'outcomes': ycols})
        for t in spec['treat']:
            C.attach_estimates(name, t, 'y_tp5',
                               [x for x in XCOLS if x not in spec['treat']]
                               + [t2 for t2 in spec['treat'] if t2 != t])
        print('  %-5s %-42s rows %d  cells %d'
              % (name, spec['title'], len(out), len({r['cell'] for r in out})))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', default='', help='p1_a|p1_b|p1_c|p1_d (default: all)')
    a = ap.parse_args()
    main(a.route or None)