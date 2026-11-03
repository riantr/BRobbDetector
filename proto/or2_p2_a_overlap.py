"""P2-a (or2): is the residual DEFINITIONAL? A box-level overlap panel.

The 18 images holding 92 of the 175 FN are the suspected cluster regime. If their
GT boxes are mutually overlapping, no hole-based method can separate them (two
overlapping drawn rectangles share a region), so the residual is a definitional
limit rather than an algorithmic one. This turns that question into an estimable
quantity: unit = GT box, treatment = how much it overlaps other GT boxes,
outcome = detected@D5 by the shipped detector.

Treatments: t_overlap_deg (max IoU with any other GT box in the same image),
            t_overlap_any (0/1, overlap_deg > 0)
Outcome:    y_det5 (1 = matched at D=5)
Covariates (PRE-treatment, all properties of the box/image, not of the overlap):
            x_min_side, x_area, x_aspect, x_edge_dist, x_n_gt_image

Usage: python proto/or2_p2_a_overlap.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from common import iou_obb
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect
import or2_common as C

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def max_overlap(q, others):
    best = 0.0
    for o in others:
        v = float(iou_obb(np.asarray(q, np.int32), np.asarray(o, np.int32)))
        if v > best:
            best = v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='p2a_overlap')
    ap.add_argument('--dist', type=int, default=5)
    args = ap.parse_args()

    from eval_pointmatch import match_boxes_point
    ids = get_ids()
    rows = []
    n_det_total = n_gt_total = tp_total = 0
    for k, iid in enumerate(ids, 1):
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
              if not quad_on_boundary(g, W, H, 0.0)]
        m = red_mask(img, P['sat_min'], P['val_min'], P['hue_max'])
        if P['denoise_min_area'] > 0:
            m = denoise(m, P['denoise_min_area'])
        det = [np.asarray(q, np.int32) for q in
               detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)]
        # base-arm accounting covers ALL 500 benchmark images (the frozen dets
        # count includes detections on the 20 no-GT images); the BOX panel rows
        # start only after the GT filter.
        n_det_total += len(det)
        if not gt:
            continue
        # per-GT-box detection status: a GT box is detected when some detection
        # matches it (all 4 corners within D) under the point-match rule
        _t, unmatched = match_boxes_point(det, gt, float(args.dist))
        n_gt_total += len(gt); tp_total += int(_t)
        used = set(range(len(det))) - set(unmatched)
        hit = []
        for gi, g in enumerate(gt):
            ok = False
            for di in used:
                t1, u1 = match_boxes_point([det[di]], [g], float(args.dist))
                if t1 == 1:
                    ok = True
                    break
            hit.append(1.0 if ok else 0.0)
        for gi, g in enumerate(gt):
            others = [o for j, o in enumerate(gt) if j != gi]
            ov = max_overlap(g, others)
            sides = [float(np.linalg.norm(g[(i + 1) % 4] - g[i])) for i in range(4)]
            area = 0.5 * abs(float(np.dot(g[1] - g[0], g[3] - g[0])) +
                             float(np.dot(g[2] - g[1], g[3] - g[1])))
            cen = g.mean(axis=0)
            edge = min(cen[0], W - 1 - cen[0], cen[1], H - 1 - cen[1])
            rows.append({
                'unit': 'box', 'block': iid, 'cell': 'shipped',
                # continuous exposure, kept for DoWhy use but NOT a treatment
                # (its 400+ near-unique levels cannot satisfy >=10 units/level)
                'ex_overlap_deg': round(ov, 4),
                # ordinal treatment: 0 = no overlap; 1/2/3 = terciles of the
                # overlapping boxes, so every level has >= 10 units
                't_overlap_bin': 0,      # filled in the post-pass below from the terciles
                't_overlap_any': 1.0 if ov > 0 else 0.0,
                'y_det5': hit[gi],
                'x_min_side': round(min(sides), 2), 'x_area': round(area, 1),
                'x_aspect': round(max(sides) / max(min(sides), 1e-6), 3),
                'x_edge_dist': round(edge, 1), 'x_n_gt_image': len(gt),
            })
        if k % 200 == 0:
            print('  %d/%d images' % (k, len(ids)), flush=True)

    n_ov = sum(1 for r in rows if r['t_overlap_any'] > 0)
    # ordinal overlap treatment: terciles of the OVERLAPPING boxes, so each of
    # levels 1..3 has ~n_ov/3 >= 10 units; level 0 is "no overlap"
    ov_vals = sorted(r['ex_overlap_deg'] for r in rows if r['ex_overlap_deg'] > 0)
    if n_ov >= 30:
        q1 = ov_vals[len(ov_vals) // 3]
        q2 = ov_vals[2 * len(ov_vals) // 3]
        for r in rows:
            v = r['ex_overlap_deg']
            r['t_overlap_bin'] = 0 if v <= 0 else (1 if v <= q1 else (2 if v <= q2 else 3))
    else:
        for r in rows:
            r['t_overlap_bin'] = 1 if r['ex_overlap_deg'] > 0 else 0
    print('boxes: %d  overlapping: %d (%.1f%%)  detected@D5: %.1f%%'
          % (len(rows), n_ov, 100.0 * n_ov / max(len(rows), 1),
             100.0 * np.mean([r['y_det5'] for r in rows])))
    det_rate_ov = np.mean([r['y_det5'] for r in rows if r['t_overlap_any'] > 0]) if n_ov else 0
    det_rate_no = np.mean([r['y_det5'] for r in rows if r['t_overlap_any'] == 0])
    print('  detection rate: overlapping %.1f%%  vs non-overlapping %.1f%%'
          % (100 * det_rate_ov, 100 * det_rate_no))
    print('  base arm (pm_v10): D5 %d TP / %d dets / %d GT -> %s'
          % (tp_total, n_det_total, n_gt_total,
             'REPRODUCES FROZEN' if (tp_total, n_det_total, n_gt_total) == (901, 1070, 1076)
             else 'MISMATCH'), flush=True)

    spec = {
        'unit': 'box', 'outcome': 'y_det5', 'treatment': 't_overlap_bin',
        'common_causes': ['x_min_side', 'x_area', 'x_aspect', 'x_edge_dist',
                          'x_n_gt_image'],
        'assumption': ('Whether a GT box overlaps another GT box is a property of the '
                       'hand annotation, not of the detector; conditional on box '
                       'geometry and density, the overlap-outcome relation identifies '
                       'the definitional-limit component.'),
        'dowhy_usage': ("dowhy.CausalModel(data=df, treatment='t_overlap_bin', "
                        "outcome='y_det5', common_causes=['x_min_side','x_area',"
                        "'x_aspect','x_edge_dist','x_n_gt_image'])"),
    }
    j, csvp = C.write_panel(args.out, rows, spec,
                            meta={'n_boxes': len(rows), 'n_overlapping': n_ov,
                                  'continuous_exposure': 'ex_overlap_deg (kept for DoWhy; 400+ near-unique levels, not a binned treatment)',
                                  'det_rate_overlapping': round(float(det_rate_ov), 4),
                                  'det_rate_non_overlapping': round(float(det_rate_no), 4),
                                  'base_arm_pm_v10': {
                                      'tp5': tp_total, 'dets': n_det_total, 'n_gt': n_gt_total,
                                      'reproduces_frozen': (tp_total, n_det_total, n_gt_total) == (901, 1070, 1076),
                                      'note': ('shipped pm_v10 detector summed over the same 500-image arm; '
                                               'these are the per-GT-box totals of the frozen D5 table')},
                                  'dist': args.dist})
    C.attach_estimates(args.out, 't_overlap_bin', 'y_det5', spec['common_causes'])
    C.attach_estimates(args.out, 't_overlap_any', 'y_det5', spec['common_causes'])
    print('saved %s and %s' % (j, csvp))


if __name__ == '__main__':
    main()