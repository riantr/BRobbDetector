"""P2-e (or2): agreement / triage panel, box-level.

Unit = a SHIPPED detection. Treatment = whether the trained OBB detector agrees with
it at D=3 (a deployable, label-free signal). Outcome = is that detection a TP@D5.
If agreement predicts correctness, the pair of detectors can triage recall without
a human: a missed box in the shipped output is worth flagging when the second
detector disagrees with everything nearby.

Reads the OBB detections cached by P2-c (`tmp_or2_dl_dets.json`, arm 1 = 2,000 imgs).

Covariates (PRE-treatment): x_sup (perimeter red support), x_n_det_image,
x_det_area, x_det_min_side.

Usage: python proto/or2_p2_e_agreement.py
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support
import or2_common as C

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
CACHE = 'tmp_loop2_pseudo/dl_dets.json'


def qdist(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dm = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    used, worst = set(), 0.0
    for i in range(4):
        for j in np.argsort(dm[i]):
            if int(j) not in used:
                used.add(int(j))
                worst = max(worst, float(dm[i, j]))
                break
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='p2e_agreement')
    ap.add_argument('--tau', type=float, default=3.0)
    args = ap.parse_args()

    if not os.path.exists(CACHE):
        print('NOT-ATTEMPTED: %s missing (run proto/or2_p2_c_scale.py first)' % CACHE)
        return
    dl = {k: v for k, v in json.load(open(CACHE, encoding='utf-8')).items()}
    dl = dl.get('1', dl.get(1, {}))

    from eval_pointmatch import match_boxes_point, is_real_unlabeled
    ids = get_ids()
    exp = load_expected()
    rows = []
    n_gt_total = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
              if not quad_on_boundary(g, W, H, 0.0)]
        n_gt_total += len(gt)
        m = red_mask(img, P['sat_min'], P['val_min'], P['hue_max'])
        if P['denoise_min_area'] > 0:
            m = denoise(m, P['denoise_min_area'])
        det = [np.asarray(q, np.int32) for q in
               detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)]
        if not det:
            continue
        dlq = [np.asarray(q, np.int32) for q in dl.get(iid, [])]
        tp, unm = match_boxes_point(det, gt, 5.0)
        unmatched = set(unm)
        for di, q in enumerate(det):
            agree = 0.0
            for q2 in dlq:
                if qdist(q, q2) <= args.tau:
                    agree = 1.0
                    break
            sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
            rows.append({'unit': 'box', 'block': iid,
                         'cell': 'agree' if agree else 'alone',
                         't_agree': agree,
                         'y_is_tp5': 0.0 if di in unmatched else 1.0,
                         'x_sup': round(float(_red_support(m, np.asarray(q, float))), 4),
                         'x_n_det_image': len(det),
                         'x_det_min_side': round(min(sides), 2),
                         'x_det_area': round(float(cv2.contourArea(np.asarray(q, np.float32))), 1)})

    n = len(rows)
    tp_sum = int(sum(r['y_is_tp5'] for r in rows))
    ag = [r for r in rows if r['t_agree'] == 1.0]
    al = [r for r in rows if r['t_agree'] == 0.0]
    print('detections: %d  agree@D3 %.1f%%  TP rate: agree %.1f%% / alone %.1f%%'
          % (n, 100.0 * len(ag) / max(n, 1),
             100.0 * np.mean([r['y_is_tp5'] for r in ag]) if ag else 0.0,
             100.0 * np.mean([r['y_is_tp5'] for r in al]) if al else 0.0))
    print('  base arm (pm_v10): %d dets / %d TP / %d GT -> %s'
          % (n, tp_sum, n_gt_total,
             'REPRODUCES FROZEN' if (n, tp_sum, n_gt_total) == (1070, 901, 1076)
             else 'MISMATCH'), flush=True)

    spec = {
        'unit': 'box', 'outcome': 'y_is_tp5', 'treatment': 't_agree',
        'common_causes': ['x_sup', 'x_n_det_image', 'x_det_min_side', 'x_det_area'],
        'assumption': ('Agreement with a second, independently-trained detector is a '
                       'label-free signal; conditional on the detection\'s own support '
                       'and geometry it identifies the probability that the detection is '
                       'a true positive.'),
        'dowhy_usage': ("dowhy.CausalModel(data=df, treatment='t_agree', "
                        "outcome='y_is_tp5', common_causes=['x_sup','x_n_det_image',"
                        "'x_det_min_side','x_det_area'])"),
    }
    j, csvp = C.write_panel(args.out, rows, spec,
                            meta={'n_dets': n, 'frac_agree': round(len(ag) / max(n, 1), 4),
                                  'tp_rate_agree': round(float(np.mean(
                                      [r['y_is_tp5'] for r in ag])) if ag else 0.0, 4),
                                  'tp_rate_alone': round(float(np.mean(
                                      [r['y_is_tp5'] for r in al])) if al else 0.0, 4),
                                  'base_arm_pm_v10': {
                                      'n_dets': n, 'tp5': tp_sum, 'n_gt': n_gt_total,
                                      'reproduces_frozen': (n, tp_sum, n_gt_total) == (1070, 901, 1076),
                                      'note': ('shipped pm_v10 detections/TP over the same 500-image arm; '
                                               'the panel unit is the detection, so the base arm is the '
                                               'detection count and TP that the frozen table scores')},
                                  'tau': args.tau,
                                  'dl_cache': CACHE + ' (gitignored via tmp_*/)',
                                  'arm1_checkpoint': {
                                      'path': 'tmp_loop2_scale/train/weights/best.pt',
                                      'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
                                              os.path.getmtime('tmp_loop2_scale/train/weights/best.pt')))},
                                  'cache_mtime': time.strftime('%Y-%m-%d %H:%M:%S',
                                      time.localtime(os.path.getmtime(CACHE)))})
    C.attach_estimates(args.out, 't_agree', 'y_is_tp5', spec['common_causes'])
    print('saved %s and %s' % (j, csvp))


if __name__ == '__main__':
    main()