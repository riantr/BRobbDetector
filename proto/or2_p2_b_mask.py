"""P2-b (or2): stroke-mask source panel — including an ORACLE arm.

The classical failure is that a neighbouring stroke / arrow / text fragments the box
interior, and colour cannot tell "box side" from "arrow". A *learned* stroke mask
could, but training one is out of scope (no GPU). The ORACLE arm bounds what such a
mask could buy: render the GT boxes as strokes into a mask and run the SAME pipeline
on it. If the oracle adds little, no learned mask is worth training; if it adds a
lot, the ceiling is real and the work is justified.

Treatments: t_mask_wide   = widened-hue colour mask (hue<=10) vs the shipped hue<=4
            t_mask_oracle = GT-rendered stroke mask (upper bound, not deployable)
Outcome:    y_tp5 / y_tp3 per image (GT-corrected)
Covariates: x_n_gt, x_img_w, x_img_h, x_red_ratio (pre-treatment)

Usage: python proto/or2_p2_b_mask.py --workers 12
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
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect
from s1_inward_complete import score
import or2_common as C

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def gt_mask(quads, H, W, thick=3):
    m = np.zeros((H, W), np.uint8)
    for q in quads:
        cv2.polylines(m, [np.asarray(q, np.int32)], True, 255, thick)
    return m


def mask_for(kind, img, gt_quads):
    if kind == 'oracle':
        H, W = img.shape[:2]
        return gt_mask(gt_quads, H, W)
    hue = 10 if kind == 'wide' else P['hue_max']
    m = red_mask(img, P['sat_min'], P['val_min'], hue)
    if P['denoise_min_area'] > 0:
        m = denoise(m, P['denoise_min_area'])
    return m


def _work(task):
    iid, kinds = task
    img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
    if img is None:
        return iid, []
    H, W = img.shape[:2]
    gt = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
          if not quad_on_boundary(g, W, H, 0.0)]
    out = []
    for kind in kinds:
        m = mask_for(kind, img, gt)
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
        out.append([np.asarray(q, np.int32).tolist() for q in det])
    return iid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='p2b_mask')
    args = ap.parse_args()
    KINDS = ['shipped', 'wide', 'oracle']

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
                _work, [(i, KINDS) for i in ids], chunksize=4)):
            for ki, kind in enumerate(KINDS):
                dets[kind][iid] = [np.asarray(q, np.int32) for q in out[ki]]
            if (k + 1) % 200 == 0:
                print('  %d/%d' % (k + 1, len(ids)), flush=True)

    for kind in KINDS:
        s = score(dets[kind], ids, gt_map, exp)
        print('  %-8s D3 %4d/%4d f%.1f | D5 %4d/%4d f%.1f | dets %4d'
              % (kind, s['3']['tp'], s['3']['true_fp'], s['3']['corr_f1'],
                 s['5']['tp'], s['5']['true_fp'], s['5']['corr_f1'], s['5']['n_det']),
              flush=True)

    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = gt_map[iid]
        rm = red_mask(img, P['sat_min'], P['val_min'], P['hue_max'])
        rd = float(rm.sum()) / max(H * W, 1) / 255.0
        for kind in KINDS:
            det = dets[kind].get(iid, [])
            tp3 = tp5 = 0
            from eval_pointmatch import match_boxes_point
            tp3, _ = match_boxes_point(det, gt, 3.0)
            tp5, _ = match_boxes_point(det, gt, 5.0)
            rows.append({'unit': 'image', 'block': iid, 'cell': kind,
                         't_mask_wide': 1.0 if kind == 'wide' else 0.0,
                         't_mask_oracle': 1.0 if kind == 'oracle' else 0.0,
                         'y_tp3': float(tp3), 'y_tp5': float(tp5),
                         'x_n_gt': len(gt), 'x_img_w': W, 'x_img_h': H,
                         'x_red_ratio': round(rd, 6)})

    spec = {
        'unit': 'image', 'outcome': 'y_tp5', 'treatment': 't_mask_oracle',
        'common_causes': ['x_n_gt', 'x_img_w', 'x_img_h', 'x_red_ratio',
                          't_mask_wide'],
        'assumption': ('The mask source is assigned by the experimenter in a 3-arm '
                       'within-image design; each arm is identified ONLY against the '
                       'shipped reference, conditioning on the other arm (estimating an '
                       'arm against a pooled control is not a well-defined contrast). '
                       'The ORACLE arm uses the GT and is an UPPER BOUND, never a '
                       'deployable configuration.'),
        'dowhy_usage': ("dowhy.CausalModel(data=df, treatment='t_mask_oracle', "
                        "outcome='y_tp5', common_causes=['x_n_gt','x_img_w','x_img_h',"
                        "'x_red_ratio','t_mask_wide'])"),
    }
    j, csvp = C.write_panel(args.out, rows, spec,
                            meta={'kinds': KINDS,
                                  'contrast_definition': ('3-arm within-image design with shipped as '
                                      'the reference; each arm is estimated with the OTHER arm as a '
                                      'co-factor, so t_mask_wide = wide - shipped and t_mask_oracle = '
                                      'oracle - shipped (never vs a pooled control)'),
                                  'raw_arm_totals': {k: int(sum(r['y_tp5'] for r in rows
                                                                if r['cell'] == k)) for k in KINDS},
                                  'note': ('oracle = GT-rendered stroke mask: an upper '
                                           'bound on what a learned stroke mask could '
                                           'achieve; not a deployable arm')})
    C.attach_estimates(args.out, 't_mask_oracle', 'y_tp5', spec['common_causes'])
    C.attach_estimates(args.out, 't_mask_wide', 'y_tp5',
                       [c for c in spec['common_causes'] if c != 't_mask_wide']
                       + ['t_mask_oracle'])
    print('saved %s and %s' % (j, csvp))


if __name__ == '__main__':
    main()