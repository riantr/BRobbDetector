"""P2-d (or2): cardinality-prior panel — estimate the trade, do not re-adopt it.

`common.load_expected()` gives a per-image expected annotation count that the
detector never uses. The HARD cap was already measured to cost 25 TP for 37 trueFP;
this route exists to put an interval on that trade and to test the never-tried SOFT
variant (rank by support, delete nothing).

Treatments: t_cap  = hard cap to the expected count (drop lowest red support)
            ex_soft = soft rank (reorder only; emits everything). This arm is a
                        LITERAL NO-OP, so it is a non-treatment `ex_*` column and is
                        recorded NOT-ATTEMPTED rather than estimated.
Outcome:    y_tp5 (GT-corrected), y_fp5 (raw unmatched)
Covariates: x_n_gt, x_csv_n, x_n_holes, x_red_ratio (all pre-treatment)

Usage: python proto/or2_p2_d_cardinality.py --workers 12
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support
from s1_inward_complete import score
import or2_common as C

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
KINDS = ['none', 'cap', 'soft']


def _work(task):
    iid, cap = task
    img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
    if img is None:
        return iid, {}
    m = red_mask(img, P['sat_min'], P['val_min'], P['hue_max'])
    if P['denoise_min_area'] > 0:
        m = denoise(m, P['denoise_min_area'])
    det = [np.asarray(q, np.int32) for q in
           detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)]
    sup = [_red_support(m, np.asarray(q, float)) for q in det]
    out = {'none': det}
    if cap > 0 and len(det) > cap:
        order = sorted(range(len(det)), key=lambda i: (-sup[i], i))
        out['cap'] = [det[i] for i in sorted(order[:cap])]
    else:
        out['cap'] = det
    out['soft'] = [det[i] for i in sorted(range(len(det)), key=lambda i: (-sup[i], i))]
    return iid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='p2d_cardinality')
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()

    dets = {k: {} for k in KINDS}
    with Pool(args.workers) as pool:
        for k, (iid, out) in enumerate(pool.imap_unordered(
                _work, [(i, int(exp.get(i, 0))) for i in ids], chunksize=4)):
            for kind in KINDS:
                dets[kind][iid] = out.get(kind, [])
            if (k + 1) % 200 == 0:
                print('  %d/%d' % (k + 1, len(ids)), flush=True)

    for kind in KINDS:
        s = score(dets[kind], ids, gt_map, exp)
        print('  %-5s D5 %4d TP / %4d trueFP  f%.1f  dets %4d'
              % (kind, s['5']['tp'], s['5']['true_fp'], s['5']['corr_f1'],
                 s['5']['n_det']), flush=True)

    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = gt_map[iid]
        rm = red_mask(img, P['sat_min'], P['val_min'], P['hue_max'])
        rd = float(rm.sum()) / max(H * W, 1) / 255.0
        from eval_pointmatch import match_boxes_point
        for kind in KINDS:
            det = dets[kind].get(iid, [])
            tp5, unm = match_boxes_point(det, gt, 5.0)
            rows.append({'unit': 'image', 'block': iid, 'cell': kind,
                         't_cap': 1.0 if kind == 'cap' else 0.0,
                         'ex_soft': 1.0 if kind == 'soft' else 0.0,
                         'y_tp5': float(tp5), 'y_fp5': float(len(unm)),
                         'x_n_gt': len(gt), 'x_csv_n': int(exp.get(iid, 0)),
                         'x_red_ratio': round(rd, 6), 'n_det': len(det)})

    # `soft` only reorders the detections; if it never changes the detection SET,
    # the treatment is a literal no-op and its PLR coefficient is unidentifiable.
    def _keyset(lst):
        return sorted(tuple(np.asarray(q).ravel().tolist()) for q in lst)
    soft_is_noop = all(
        _keyset(dets['none'].get(i, [])) == _keyset(dets['soft'].get(i, []))
        for i in ids)

    def _kind_tot(kind):
        rr = [r for r in rows if r['cell'] == kind]
        return {'tp5': int(sum(r['y_tp5'] for r in rr)),
                'fp5': int(sum(r['y_fp5'] for r in rr)),
                'dets': int(sum(r['n_det'] for r in rr))}
    soft_evidence = {'none': _kind_tot('none'), 'soft': _kind_tot('soft'),
                     'byte_identical_detection_sets': soft_is_noop,
                     'note': ('soft reorders the detections by red support and deletes '
                              'nothing, so both levels of ex_soft have the same outcome '
                              'on every image')}
    print('ex_soft degenerate (same detection sets as none): %s  %s'
          % (soft_is_noop, soft_evidence['none']), flush=True)

    spec = {
        'unit': 'image', 'outcome': 'y_tp5', 'treatment': 't_cap',
        'common_causes': ['x_n_gt', 'x_csv_n', 'x_red_ratio', 'ex_soft'],
        'assumption': ('The prior mode is assigned by the experimenter; the effect on '
                       'TP is identified conditionally on image density and redness. '
                       'ex_soft is a no-deletion reorder arm and a LITERAL NO-OP '
                       '(same detection sets), so it is retained only as a control and '
                       'is NOT estimated: its effect is UNIDENTIFIABLE.'),
        'dowhy_usage': ("dowhy.CausalModel(data=df, treatment='t_cap', outcome='y_tp5', "
                        "common_causes=['x_n_gt','x_csv_n','x_red_ratio','ex_soft'])"),
    }
    j, csvp = C.write_panel(args.out, rows, spec,
                            meta={'kinds': KINDS,
                                  'ex_soft_degenerate': soft_is_noop,
                                  'ex_soft_evidence': soft_evidence})
    C.attach_estimates(args.out, 't_cap', 'y_tp5', spec['common_causes'])
    if soft_is_noop:
        C.record_unidentifiable(
            args.out, 'ex_soft', 'y_tp5',
            reason=('NOT ATTEMPTED: ex_soft only reorders detections, never deletes, so '
                    'both levels emit the same detection set on every image; the outcome '
                    'has no variation with the treatment'),
            evidence=soft_evidence)
    else:
        C.attach_estimates(args.out, 'ex_soft', 'y_tp5',
                           [c for c in spec['common_causes'] if c != 'ex_soft'])
    print('saved %s and %s' % (j, csvp))


if __name__ == '__main__':
    main()