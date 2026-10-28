"""STEP 1 (loop2): re-baseline the trained detector against the TRUE GT.

Premise (verified in benchmark/loop2_baseline_and_premise.json):
  * BRobbDetector/Assets/yolov8s_obb.onnx   mtime 2026-10-12 17:14
  * proto/common.py::LAB re-pointed to Label@CARS_202609_raw in d3b6896 (2026-10-21)
  => the shipped checkpoint was trained AND scored against Compare-V1, so every
     published YOLO number is not comparable to pm_v10's 85.4.

This script
  1. scores the SHIPPED checkpoint on the rule-consistent 1076-box GT,
  2. retrains from the CURRENT labels (true GT only - no detector-derived
     additions, so the evaluation stays clean),
  3. scores the retrained model the same way,
  4. applies a pre-stated decision rule for steps 4-5.

Scoring uses the same GT-corrected point-match metric as the frozen baseline.

Usage: python proto/dl2_yolo_truegt.py [--epochs 20] [--conf 0.55] [--train]
       (without --train it only scores the shipped checkpoint)
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, LAB, load_expected
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary
from s1_inward_complete import score

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
SHIPPED = 'BRobbDetector/Assets/yolov8s_obb.onnx'
WORK = 'tmp_loop2_yolo'                     # gitignored (tmp_*/)
PM_V10 = {'3': 81.2, '5': 85.4, '8': 87.6, '12': 88.6}
PM_V10_TP5 = 901


def load_gt(ids):
    gt = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt[iid] = [np.asarray(g, np.int32) for g in load_boxes_any(LAB, iid)
                   if not quad_on_boundary(g, W, H, 0.0)]
    return gt


def predict(model_path, ids, conf, task='obb'):
    from ultralytics import YOLO
    m = YOLO(model_path, task=task)
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
                qs.append((pts @ R.T + np.array([cx, cy])).astype(np.int32))
        out[iid] = qs
        if k % 100 == 0:
            print('    predicted %d/%d' % (k, len(ids)), flush=True)
    return out


def build_dataset(train_ids, val_ids):
    root = os.path.abspath(WORK)
    if os.path.isdir(root):
        shutil.rmtree(root)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(root, 'labels', split), exist_ok=True)
    n_lab = 0
    for split, ids in (('train', train_ids), ('val', val_ids)):
        for iid in ids:
            src = os.path.join(IMG, iid + '.jpg')
            img = cv2.imread(src)
            if img is None:
                continue
            H, W = img.shape[:2]
            shutil.copy2(src, os.path.join(root, 'images', split, iid + '.jpg'))
            lines = []
            for q in load_boxes_any(LAB, iid):
                q = np.asarray(q, float)
                if quad_on_boundary(q, W, H, 0.0):
                    continue
                coords = ' '.join('%.6f %.6f' % (x / W, y / H) for x, y in q)
                lines.append('0 ' + coords)
                n_lab += 1
            with open(os.path.join(root, 'labels', split, iid + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
    yml = os.path.join(root, 'data.yaml')
    with open(yml, 'w') as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\n"
                "names:\n  0: box\n" % root.replace('\\', '/'))
    return yml, n_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--conf', type=float, default=0.55)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--model', default='yolov8s-obb.pt')
    ap.add_argument('--train-imgs', type=int, default=400)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--train-workers', type=int, default=2)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--out', default='benchmark/loop2_dl_truegt.json')
    args = ap.parse_args()

    ids = get_ids()
    gt_map = load_gt(ids)
    exp = load_expected()
    ngt = sum(len(v) for v in gt_map.values())

    # decision rule, stated BEFORE any measurement
    rule = ("If the retrained model's recall@D=5 is below pm_v10's 83.7%% "
            "(TP@D=5 < 901), then steps 4 and 5 are recorded DEAD-BY-EVIDENCE "
            "and are NOT attempted.")

    res = {'premise': {'onnx_mtime': '2026-10-12 17:14',
                       'lab_repointed_commit': 'd3b6896 (2026-10-21)',
                       'lab_now': LAB},
           'decision_rule': rule, 'conf': args.conf, 'n_gt': ngt,
           'pm_v10': {'corr_f1': PM_V10, 'tp5': PM_V10_TP5},
           'models': {}}
    print('%d images, %d GT boxes, conf %.2f' % (len(ids), ngt, args.conf), flush=True)

    print('\n== shipped checkpoint (trained 2026-10-12, i.e. pre-GT-repointing) ==')
    prev = None
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out, encoding='utf-8'))
        except Exception:
            prev = None
    if args.train and prev and 'shipped_onnx' in prev.get('models', {}):
        s_ship = prev['models']['shipped_onnx']['metrics']
        print('  (reusing the already-measured shipped score: D=5 %d TP, f%.1f)'
              % (s_ship['5']['tp'], s_ship['5']['corr_f1']), flush=True)
    else:
        d_ship = predict(SHIPPED, ids, args.conf)
        s_ship = score(d_ship, ids, gt_map, exp)
        print('  shipped  D=3 %d/%d f%.1f | D=5 %d/%d f%.1f | D=8 f%.1f | D=12 f%.1f'
              % (s_ship['3']['tp'], s_ship['3']['true_fp'], s_ship['3']['corr_f1'],
                 s_ship['5']['tp'], s_ship['5']['true_fp'], s_ship['5']['corr_f1'],
                 s_ship['8']['corr_f1'], s_ship['12']['corr_f1']), flush=True)
    res['models']['shipped_onnx'] = {'metrics': s_ship}

    best = None
    if args.train:
        rng = np.random.RandomState(7)
        order = list(ids)
        rng.shuffle(order)
        val_ids, train_ids = order[:100], order[100:100 + args.train_imgs]
        yml, n_lab = build_dataset(train_ids, val_ids)
        print('\n== retrain on the TRUE GT: %d train / %d val images, %d labels, '
              '%d epochs, imgsz %d ==' % (len(train_ids), len(val_ids), n_lab,
                                          args.epochs, args.imgsz), flush=True)
        from ultralytics import YOLO
        model = YOLO(args.model)
        model.train(data=yml, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=args.train_workers,
                    project=os.path.abspath(WORK), name='train', exist_ok=True,
                    verbose=False, plots=False)
        best = os.path.join(os.path.abspath(WORK), 'train', 'weights', 'best.pt')
        res['training'] = {'train_imgs': len(train_ids), 'val_imgs': len(val_ids),
                           'labels': n_lab, 'epochs': args.epochs,
                           'imgsz': args.imgsz, 'base_model': args.model,
                           'batch': args.batch, 'weights': best,
                           'trained_on': 'Label@CARS_202609_raw only '
                                         '(no detector-derived additions)',
                           'cap_note': ('CPU-only: the 20-epoch 640px run was '
                                        'projected at ~8 h and was killed at the '
                                        'goal 90-min cap; parameters shrunk.')}

        print('\n== retrained (%d epochs) ==' % args.epochs, flush=True)
        d_new = predict(best, ids, args.conf)
        s_new = score(d_new, ids, gt_map, exp)
        # held-out part of the split, reported separately
        s_val = score({i: d_new[i] for i in val_ids},
                      val_ids, {i: gt_map[i] for i in val_ids}, exp)
        res['models']['retrained'] = {'metrics': s_new, 'val_split': s_val}
        print('  retrained D=3 %d/%d f%.1f | D=5 %d/%d f%.1f | D=8 f%.1f | D=12 f%.1f'
              % (s_new['3']['tp'], s_new['3']['true_fp'], s_new['3']['corr_f1'],
                 s_new['5']['tp'], s_new['5']['true_fp'], s_new['5']['corr_f1'],
                 s_new['8']['corr_f1'], s_new['12']['corr_f1']), flush=True)
        print('  (held-out val split of 100 imgs: D=5 %d TP, f%.1f)'
              % (s_val['5']['tp'], s_val['5']['corr_f1']), flush=True)

        tp5 = s_new['5']['tp']
        res['decision'] = {
            'rule_applied': rule,
            'retrained_tp5': tp5,
            'pm_v10_tp5': PM_V10_TP5,
            'steps_4_5': 'PROCEED' if tp5 >= PM_V10_TP5 else 'DEAD-BY-EVIDENCE',
        }
        print('\n  DECISION: retrained TP@D5 %d vs pm_v10 %d -> steps 4-5 %s'
              % (tp5, PM_V10_TP5, res['decision']['steps_4_5']), flush=True)
    else:
        res['decision'] = {'rule_applied': rule,
                           'steps_4_5': 'PENDING (run with --train)'}

    json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nsaved %s' % args.out)


if __name__ == '__main__':
    main()