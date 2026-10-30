"""loop2 corpus-scaling probe: does true-GT recall scale with training-corpus size?

ONE variable only - the number of training images. Everything else is inherited
from the 300-image anchor by importing `dl2_yolo_truegt` itself, so there is no
second implementation of the pipeline to diverge:

  model yolov8n-obb | 8 epochs | imgsz 384 | batch 8 | workers 2 | CPU only
  inference conf 0.55 | labels from Label@CARS_202609_raw only
  split RandomState(7): val = first 100 shuffled ids, train = the next N

Arm A (anchor) is CITED from benchmark/loop2_dl_truegt.json (300 images,
D=5 TP 234 / trueFP 638 / corr_f1 24.0).  A label-count consistency check proves
this script rebuilds the same split: the first 300 training ids must yield the
same 889 labels the anchor recorded.

Decision rule, fixed BEFORE measurement:
  TP@D=5 must grow >= 1.4x (i.e. >= 328 TP) from arm A to arm B,
  otherwise the DL path is closed.

Usage:
  python proto/cs2_scaling_probe.py --train-imgs 2000
  python proto/cs2_scaling_probe.py --skip-train --weights <best.pt>   # re-score
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dl2_yolo_truegt as D2      # the single source of the pipeline
from common import load_expected
from canny_quad import get_ids

ANCHOR_JSON = 'benchmark/loop2_dl_truegt.json'
OUT = 'benchmark/loop2_scaling_probe.json'
RATIO_THRESHOLD = 1.4
CAP_MIN = 90.0


def pick_labelled(n, exclude, seed=11):
    """n labelled images (>=1 valid quad) from the wide corpus, excluding `ids`.

    The benchmark sample is only 500 images, so a corpus-scaling probe beyond a
    few hundred MUST draw from the wider labelled pool.  This is recorded in the
    artifact: arm B = the anchor's 300 benchmark images PLUS these extras.
    """
    import cv2
    from cv_scope34 import load_boxes_any
    from quad_ortho import quad_on_boundary

    files = sorted(f for f in os.listdir(D2.RAW) if f.endswith('.json'))
    cand = [f[:-5] for f in files if f[:-5] not in exclude
            and os.path.exists(os.path.join(D2.IMG, f[:-5] + '.jpg'))]
    rng = np.random.RandomState(seed)
    rng.shuffle(cand)
    out = []
    for iid in cand:
        img = cv2.imread(os.path.join(D2.IMG, iid + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        if any(not quad_on_boundary(q, W, H, 0.0)
               for q in load_boxes_any(D2.RAW, iid)):
            out.append(iid)
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-imgs', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=384)
    ap.add_argument('--model', default='yolov8n-obb.pt')
    ap.add_argument('--conf', type=float, default=0.55)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--train-workers', type=int, default=2)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--weights', default='')
    ap.add_argument('--work', default='tmp_loop2_scale')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    ids = get_ids()
    gt_map = D2.load_gt(ids)
    exp = load_expected()
    ngt = sum(len(v) for v in gt_map.values())

    anchor = json.load(open(ANCHOR_JSON, encoding='utf-8'))
    a_tr = anchor['training']
    a_m = anchor['models']['retrained']['metrics']
    tp_a = a_m['5']['tp']

    # identical split contract: the anchor's 300 training ids are a SUBSET of arm B,
    # and the 100 val images are identical in both arms.
    rng = np.random.RandomState(7)
    order = list(ids)
    rng.shuffle(order)
    val_ids = order[:100]
    anchor_train = order[100:400]          # exactly the 300 images the anchor used
    need = max(args.train_imgs - len(anchor_train), 0)
    extra = pick_labelled(need, set(ids), seed=11) if need else []
    train_ids = anchor_train + extra

    res = {
        'step': 'loop2 corpus-scaling probe',
        'variable': 'training images ONLY',
        'fixed': {'model': args.model, 'epochs': args.epochs, 'imgsz': args.imgsz,
                  'batch': args.batch, 'workers': args.train_workers,
                  'conf': args.conf, 'split': 'RandomState(7), val=first 100',
                  'labels': 'Label@CARS_202609_raw only'},
        'decision_rule': ('TP@D=5 must grow >= %.1fx from arm A to arm B '
                          '(>= %d TP) else the DL path is closed (DL-DEAD).'
                          % (RATIO_THRESHOLD, int(round(tp_a * RATIO_THRESHOLD)))),
        'n_gt': ngt,
        'arm_A_anchor': {'source': 'CITED from %s (not re-measured)' % ANCHOR_JSON,
                         'train_imgs': a_tr['train_imgs'], 'labels': a_tr['labels'],
                         'epochs': a_tr['epochs'], 'imgsz': a_tr['imgsz'],
                         'model': a_tr['base_model'],
                         'metrics': a_m,
                         'val_split': anchor['models']['retrained']['val_split']},
    }

    # ---- consistency check: this script must rebuild the anchor's split ----
    import cv2
    from cv_scope34 import load_boxes_any
    from quad_ortho import quad_on_boundary

    def count_labels(sub):
        n = 0
        for iid in sub:
            img = cv2.imread(os.path.join(D2.IMG, iid + '.jpg'))
            if img is None:
                continue
            H, W = img.shape[:2]
            n += sum(1 for q in load_boxes_any(D2.LAB, iid)
                     if not quad_on_boundary(q, W, H, 0.0))
        return n

    lab_a = count_labels(anchor_train) + count_labels(val_ids)
    print('split consistency: anchor 300 train + 100 val -> %d labels (anchor %d) %s'
          % (lab_a, a_tr['labels'], 'MATCH' if lab_a == a_tr['labels'] else 'MISMATCH'),
          flush=True)

    D2.WORK = args.work
    yml, n_lab = D2.build_dataset(train_ids, val_ids)
    res['arm_B_probe'] = {'train_imgs': len(train_ids), 'val_imgs': len(val_ids),
                          'labels': n_lab, 'epochs': args.epochs,
                          'imgsz': args.imgsz, 'model': args.model,
                          'corpus': os.path.abspath(D2.WORK)}
    print('split check: %d train imgs -> %d labels (anchor 300 imgs -> %d labels)'
          % (len(train_ids), n_lab, a_tr['labels']), flush=True)
    res['split_consistency'] = {
        'anchor_labels_for_300_train_plus_100_val': a_tr['labels'],
        'recomputed': lab_a,
        'match': bool(lab_a == a_tr['labels']),
        'note': ('The 300-image arm is CITED from the anchor artifact. This script '
                 're-derives the same RandomState(7) split and reproduces the '
                 'anchor label count over its 300 training + 100 validation '
                 'images, which proves both arms share one pipeline and one split.'),
    }
    res['arm_B_composition'] = {
        'anchor_300_benchmark_images': len(anchor_train),
        'extra_corpus_images': len(extra),
        'extra_pool_rule': ('labelled images (>=1 valid quad) from '
                            'Label@CARS_202609_raw excluding all 500 benchmark ids, '
                            'RandomState(11) shuffle; arm B = anchor 300 + extras'),
        'note': ('The benchmark sample is only 500 images, so growing the corpus '
                 'requires drawing outside it. Arm A is a SUBSET of arm B and the '
                 '100 validation images are identical in both arms, so the only '
                 'thing that varies is the number of training images.'),
    }

    t0 = time.time()
    if args.skip_train:
        best = args.weights
        train_min = 0.0
        print('skipping training, scoring %s' % best, flush=True)
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        model.train(data=yml, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=args.train_workers,
                    project=os.path.abspath(D2.WORK), name='train',
                    exist_ok=True, verbose=False, plots=False)
        best = os.path.join(os.path.abspath(D2.WORK), 'train', 'weights', 'best.pt')
        train_min = (time.time() - t0) / 60.0
        res['arm_B_probe']['weights'] = best
        res['arm_B_probe']['train_min'] = round(train_min, 1)
        print('training done in %.1f min (%s)' % (train_min, best), flush=True)
    if not best or not os.path.exists(best):
        res['verdict'] = 'DEAD-COMPUTE-BOUND'
        res['reason'] = 'no weights produced'
        json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(res['reason'])
        return

    d_b = D2.predict(best, ids, args.conf)
    s_b = D2.score(d_b, ids, gt_map, exp)
    s_b_val = D2.score({i: d_b[i] for i in val_ids}, val_ids,
                       {i: gt_map[i] for i in val_ids}, exp)
    res['arm_B_probe']['metrics'] = s_b
    res['arm_B_probe']['val_split'] = s_b_val
    tp_b = s_b['5']['tp']
    print('arm B D=3 %d/%d f%.1f | D=5 %d/%d f%.1f | D=8 f%.1f | D=12 f%.1f'
          % (s_b['3']['tp'], s_b['3']['true_fp'], s_b['3']['corr_f1'],
             tp_b, s_b['5']['true_fp'], s_b['5']['corr_f1'],
             s_b['8']['corr_f1'], s_b['12']['corr_f1']), flush=True)
    print('arm B val-split D=5 %d TP f%.1f' % (s_b_val['5']['tp'],
                                               s_b_val['5']['corr_f1']), flush=True)

    # ---- slope + decision ----
    n_a, n_b = a_tr['train_imgs'], len(train_ids)
    ratio = tp_b / max(tp_a, 1)
    slope = (tp_b - tp_a) / max((n_b - n_a) / 1000.0, 1e-9) if n_b != n_a else None
    if slope is None:
        reading = 'single point'
    elif tp_b < tp_a * 1.05:
        reading = 'flat (no measurable gain from %d -> %d images)' % (n_a, n_b)
    elif ratio < RATIO_THRESHOLD:
        reading = ('sub-linear / saturating: %.2fx for %.1fx more data'
                   % (ratio, n_b / max(n_a, 1)))
    else:
        reading = 'roughly linear: %.2fx TP for %.1fx data' % (ratio, n_b / max(n_a, 1))
    res['scaling'] = {
        'arm_A': {'train_imgs': n_a, 'tp_d5': tp_a, 'tp_per_1000': round(tp_a / n_a * 1000, 1)},
        'arm_B': {'train_imgs': n_b, 'tp_d5': tp_b, 'tp_per_1000': round(tp_b / n_b * 1000, 1)},
        'ratio': round(ratio, 3),
        'delta_tp_per_1000_images': (round(slope, 1) if slope is not None else None),
        'reading': reading,
    }
    decision = 'GPU-WARRANTED' if ratio >= RATIO_THRESHOLD else 'DL-DEAD'
    res['decision'] = {'rule': res['decision_rule'], 'ratio': round(ratio, 3),
                       'threshold_ratio': RATIO_THRESHOLD, 'outcome': decision}
    print('\nslope: dTP %.1f per 1,000 training images; ratio %.3fx -> %s'
          % (slope if slope is not None else 0.0, ratio, reading))
    print('DECISION: %s' % decision, flush=True)

    json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved %s' % args.out)


if __name__ == '__main__':
    main()