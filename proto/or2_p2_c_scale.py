"""P2-c (or2): corpus-scaling panel, per image, inference only.

Re-scores the two checkpoints already on disk (no retraining):
  arm 0 = 300 training images   tmp_loop2_scale_a/train/weights/best.pt
  arm 1 = 2,000 training images tmp_loop2_scale/train/weights/best.pt
and emits a per-image 2-arm panel so the scaling effect is estimable with a
cluster-robust SE on the image.

Caches the OBB detections to a gitignored file so P2-e can reuse them.

Usage: python proto/or2_p2_c_scale.py
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
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary
import or2_common as C

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
ARMS = {0: 'tmp_loop2_scale_a/train/weights/best.pt',
        1: 'tmp_loop2_scale/train/weights/best.pt'}
CACHE = 'tmp_loop2_pseudo/dl_dets.json'


def predict(model_path, ids, conf=0.55):
    from ultralytics import YOLO
    m = YOLO(model_path, task='obb')
    out = {}
    for k, iid in enumerate(ids, 1):
        r = m.predict(os.path.join(IMG, iid + '.jpg'), conf=conf, verbose=False)[0]
        qs = []
        if r.obb is not None and len(r.obb) > 0:
            for cx, cy, w, h, th in r.obb.xywhr.cpu().numpy():
                c, s = np.cos(float(th)), np.sin(float(th))
                R = np.array([[c, -s], [s, c]])
                pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                                [w / 2, h / 2], [-w / 2, h / 2]])
                qs.append((pts @ R.T + np.array([cx, cy])).astype(np.int32).tolist())
        out[iid] = qs
        if k % 100 == 0:
            print('    %d/%d' % (k, len(ids)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='p2c_scale')
    ap.add_argument('--conf', type=float, default=0.55)
    ap.add_argument('--use-cache', action='store_true',
                    help='rebuild the panel from the cached DL detections (no inference)')
    args = ap.parse_args()
    t0 = time.time()
    inferred = []

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]

    # pm_v10 BASE arm: the shipped pipeline on the same 500 images. Variance
    # discipline requires every route's harness to carry a base arm that
    # reproduces the frozen table; this supplies it for P2-c.
    from scope45_holes import detect as _ship
    from common import red_mask as _rm
    from canny_quad import denoise as _dn
    from scope45_holes import P as _P, HOLE_DEFAULT as _HC, PIPE_DEFAULT as _HP
    base = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        if img is None:
            base[iid] = []
            continue
        m = _rm(img, _P['sat_min'], _P['val_min'], _P['hue_max'])
        if _P['denoise_min_area'] > 0:
            m = _dn(m, _P['denoise_min_area'])
        base[iid] = [np.asarray(q, np.int32).tolist() for q in _ship(img, _P, _HC, _HP, mask=m)]
    print('base arm (pm_v10) computed on %d images' % len(base))

    dets = {}                 # 0 = DL 300 imgs, 1 = DL 2000 imgs (t_scale)
    if args.use_cache and os.path.exists(CACHE):
        cached = json.load(open(CACHE, encoding='utf-8'))
        print('using cached DL detections from %s' % CACHE)
        for k, v in cached.items():
            dets[int(k)] = v
    for arm, path in ARMS.items():
        if not os.path.exists(path):
            print('MISSING checkpoint for arm %d: %s -> NOT-ATTEMPTED' % (arm, path))
            return
        if arm in dets:
            continue
        print('arm %d: inferring %s' % (arm, path), flush=True)
        dets[arm] = predict(path, ids, args.conf)
        inferred.append(arm)
    json.dump({str(k): v for k, v in dets.items()},
              open(CACHE, 'w', encoding='utf-8'))

    from eval_pointmatch import match_boxes_point
    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = gt_map[iid]
        rm = red_mask(img, 80, 110, 10)
        rd = float(rm.sum()) / max(H * W, 1) / 255.0
        for arm in (0, 1):
            det = [np.asarray(q, np.int32) for q in dets[arm].get(iid, [])]
            tp5, unm = match_boxes_point(det, gt, 5.0)
            tp12, _ = match_boxes_point(det, gt, 12.0)
            rows.append({'unit': 'image', 'block': iid, 'cell': 'arm%d' % arm,
                         't_scale': float(arm),
                         'y_tp5': float(tp5), 'y_tp12': float(tp12),
                         'y_fp5': float(len(unm)),
                         'x_n_gt': len(gt), 'x_img_w': W, 'x_img_h': H,
                         'x_red_ratio': round(rd, 6)})

    def agg(arm):
        t5 = sum(r['y_tp5'] for r in rows if r['cell'] == 'arm%d' % arm)
        t12 = sum(r['y_tp12'] for r in rows if r['cell'] == 'arm%d' % arm)
        return int(t5), int(t12)
    a5, a12 = agg(0)
    b5, b12 = agg(1)
    ngt = sum(len(v) for v in gt_map.values())
    from eval_pointmatch import match_boxes_point as _mbp
    bt5 = sum(_mbp([np.asarray(q, np.int32) for q in base[i]], gt_map[i], 5.0)[0] for i in ids)
    bt12 = sum(_mbp([np.asarray(q, np.int32) for q in base[i]], gt_map[i], 12.0)[0] for i in ids)
    print('base arm (pm_v10, same 500 images): D5 %d TP / D12 %d  -> %s'
          % (bt5, bt12, 'REPRODUCES FROZEN (901/938)' if (bt5, bt12) == (901, 938) else 'MISMATCH'))
    print('arm0 (DL 300 imgs) : D5 %d TP' % a5)
    print('arm1 (DL 2000 imgs): D5 %d TP / %d GT   D12 %d' % (b5, ngt, b12))

    spec = {
        'unit': 'image', 'outcome': 'y_tp5', 'treatment': 't_scale',
        'common_causes': ['x_n_gt', 'x_img_w', 'x_img_h', 'x_red_ratio'],
        'assumption': ('Training-corpus size is assigned by the experimenter; the two '
                       'checkpoints share model/epochs/imgsz/conf and the same split, '
                       'so the per-image TP difference identifies the scaling effect. '
                       'This panel is the per-image view of '
                       'benchmark/loop2_scaling_probe.json.'),
        'dowhy_usage': ("dowhy.CausalModel(data=df, treatment='t_scale', outcome='y_tp5', "
                        "common_causes=['x_n_gt','x_img_w','x_img_h','x_red_ratio'])"),
    }
    j, csvp = C.write_panel(args.out, rows, spec,
                            meta={'arm0_train_imgs': 300, 'arm1_train_imgs': 2000,
                                  'arm0_tp5': a5, 'arm1_tp5': b5,
                                  'arm0_tp12': a12, 'arm1_tp12': b12, 'n_gt': ngt,
                                  'det_cache': CACHE + ' (gitignored via tmp_*/)',
                                  'wall_clock_s': round(time.time() - t0, 2),
                                  'inference_mode': 'cache' if not inferred else 'fresh',
                                  'arms_inferred_this_run': inferred,
                                  'arm_checkpoints': {
                                      str(a): {'path': path,
                                               'mtime': time.strftime('%Y-%m-%d %H:%M:%S',
                                                   time.localtime(os.path.getmtime(path)))
                                               } for a, path in ARMS.items()},
                                  'base_arm_pm_v10': {'tp5': bt5, 'tp12': bt12,
                                      'reproduces_frozen': (bt5, bt12) == (901, 938),
                                      'note': 'pm_v10 computed by this harness on the same 500 images, '
                                              'verified against the frozen table and deliberately kept '
                                              'OUT of t_scale so the 300->2000 estimand stays clean'}})
    C.attach_estimates(args.out, 't_scale', 'y_tp5', spec['common_causes'])
    C.attach_estimates(args.out, 't_scale', 'y_tp12', spec['common_causes'])
    print('saved %s and %s' % (j, csvp))


if __name__ == '__main__':
    main()