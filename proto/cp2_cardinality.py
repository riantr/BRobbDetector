"""STEP 2 (loop2): cardinality prior from the CSVs.

`common.load_expected()` gives a per-image expected annotation count that the
detector never uses (it only feeds the `is_real_unlabeled` heuristic).  This
step (a) quantifies how often that count disagrees with the true GT, then
(b) tests it as a SELECTION constraint: when an image emits more quads than
expected, keep the `expected` quads with the highest red support; when it emits
fewer, record under-detection.  It can only remove detections, so it is a
precision lever - the true positives it deletes are reported explicitly.

Stop-if (goal): if the CSV count disagrees with the true GT on more than 20% of
images, the prior is unsound -> record DEAD with that number.

Usage: python proto/cp2_cardinality.py --workers 12
"""
import argparse
import json
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

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
SOUND_MAX_DISAGREE = 0.20


def _work(task):
    iid, mode, cap = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, mode, [], 0
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    kept = [np.asarray(q, np.int32) for q in det]
    removed = 0
    if mode == 'cap' and cap > 0 and len(kept) > cap:
        sup = [_red_support(m, np.asarray(q, float)) for q in det]
        order = sorted(range(len(kept)), key=lambda i: (-sup[i], i))
        keep_idx = sorted(order[:cap])
        removed = len(kept) - len(keep_idx)
        kept = [kept[i] for i in keep_idx]
    return iid, mode, kept, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='benchmark/loop2_cardinality.json')
    ap.add_argument('--modes', default='base,cap')
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()

    # ---------- (a) how sound is the prior? ----------
    agree = exact = under = over = zero_csv = 0
    diffs = []
    for iid in ids:
        n_gt = len(gt_map[iid])
        n_csv = int(exp.get(iid, 0))
        diffs.append(n_csv - n_gt)
        if n_csv == n_gt:
            exact += 1
        else:
            if n_csv < n_gt:
                under += 1
            else:
                over += 1
        if n_csv == 0 and n_gt == 0:
            zero_csv += 1
    n = len(ids)
    disagree_rate = 1.0 - exact / max(n, 1)
    sound = disagree_rate <= SOUND_MAX_DISAGREE
    print('prior soundness: exact %d/%d (%.1f%%), csv<n_gt %d, csv>n_gt %d, '
          'disagreement %.1f%% -> %s'
          % (exact, n, 100.0 * exact / n, under, over, 100.0 * disagree_rate,
             'SOUND' if sound else 'UNSOUND (>20%)'), flush=True)

    out = {
        'prior': {'n_images': n, 'exact_agree': exact,
                  'disagree_rate': round(disagree_rate, 4),
                  'csv_lt_gt': under, 'csv_gt_gt': over,
                  'threshold': SOUND_MAX_DISAGREE, 'sound': bool(sound),
                  'diff_hist': {str(k): diffs.count(k) for k in sorted(set(diffs))}},
        'results': {},
    }
    if not sound:
        out['verdict'] = 'DEAD'
        out['reason'] = ('CSV expected count disagrees with the true GT on %.1f%% '
                         'of images (> %.0f%%), so the cardinality prior is '
                         'unsound.' % (100 * disagree_rate, 100 * SOUND_MAX_DISAGREE))
        print(out['reason'], flush=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('saved %s' % args.out)
        return

    # ---------- (b) the prior as a selection constraint ----------
    modes = args.modes.split(',')
    dets = {}
    removed_total = 0
    for mode in modes:
        det_map = {}
        rem = 0
        with Pool(args.workers) as pool:
            for iid, _m, det, r in pool.imap_unordered(
                    _work, [(i, mode, int(exp.get(i, 0))) for i in ids], chunksize=3):
                det_map[iid] = det
                rem += r
        dets[mode] = det_map
        removed_total = rem
        r = score(det_map, ids, gt_map, exp)
        out['results'][mode] = {'metrics': r, 'removed': rem}
        print('  %-5s D=3 %4d/%4d f%.1f | D=5 %4d/%4d f%.1f | D=8 %4d/%4d f%.1f '
              '| D=12 %4d/%4d f%.1f | dets=%4d removed=%d'
              % (mode, r['3']['tp'], r['3']['true_fp'], r['3']['corr_f1'],
                 r['5']['tp'], r['5']['true_fp'], r['5']['corr_f1'],
                 r['8']['tp'], r['8']['true_fp'], r['8']['corr_f1'],
                 r['12']['tp'], r['12']['true_fp'], r['12']['corr_f1'],
                 r['5']['n_det'], rem), flush=True)

    if 'base' in out['results'] and 'cap' in out['results']:
        b, c = out['results']['base']['metrics'], out['results']['cap']['metrics']
        # true positives deleted by the cap
        del_tp = b['5']['tp'] - c['5']['tp']
        out['delta'] = {D: {'dTP': c[D]['tp'] - b[D]['tp'],
                            'dTrueFP': c[D]['true_fp'] - b[D]['true_fp'],
                            'dcorr_f1': round(c[D]['corr_f1'] - b[D]['corr_f1'], 1)}
                        for D in ('3', '5', '8', '12')}
        out['deleted_true_positives_at_D5'] = del_tp
        print('\n  cap deleted %d true positives at D=5' % del_tp)
        print('  delta D=5: dTP %+d dTrueFP %+d dcorr_f1 %+.1f'
              % (out['delta']['5']['dTP'], out['delta']['5']['dTrueFP'],
                 out['delta']['5']['dcorr_f1']))
        win = (c['5']['tp'] > b['5']['tp'] and
               c['5']['corr_f1'] >= b['5']['corr_f1'])
        out['verdict'] = 'WIN' if win else 'DEAD'
        out['reason'] = ('dTP@D5 %+d, dTrueFP@D5 %+d, dcorr_f1 %+.1f, deleted %d TPs'
                         % (out['delta']['5']['dTP'], out['delta']['5']['dTrueFP'],
                            out['delta']['5']['dcorr_f1'], del_tp))
        print('  verdict: %s' % out['verdict'])

    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nsaved %s' % args.out)


if __name__ == '__main__':
    main()