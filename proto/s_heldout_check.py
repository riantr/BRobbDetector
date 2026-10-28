"""Held-out re-check for S1-S5 (Done-when #3: seed 1 and seed 2 results per row).

No item passed the D=5 TP gate on the tuning set (seed 0), so this is not an
adoption check - it is the generalisation confirmation that the DEAD verdicts
are not artefacts of the tuning sample.  Each item is evaluated with its single
representative variant, on the SAME held-out samples the repo uses for
`proto/sweep_real.py --seed N` (different 500-image draws from the 56832-image
pool; seed 0 is the tuning set).

Usage: python proto/s_heldout_check.py --workers 12 --seeds 1,2
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, load_expected
from cv_scope34 import load_boxes_any
from quad_ortho import quad_on_boundary
from sweep_real import sample_ids
import s1_inward_complete as S1
import s2_small_specialist as S2
import s3_support_dedup as S3
import s4_run_select as S4
import s5_yolo_refine as S5

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"

ITEMS = [('S1', 'in060', S1), ('S2', 'union_40', S2), ('S3', 'sup', S3),
         ('S4', 'runs3', S4), ('S5', 'yunion', S5)]


def build(seed):
    ids = sample_ids(seed, 500)
    gt_map, exp = {}, load_expected()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    return ids, gt_map, exp


def run(ids, gt_map, exp, mod, mode, workers, initargs=None):
    det_map = {}
    init = S5._init if initargs is not None else None
    with Pool(workers, initializer=init, initargs=(initargs,)) if init else \
            Pool(workers) as pool:
        for rec in pool.imap_unordered(
                mod._work, [(i, mode) for i in ids], chunksize=3):
            det_map[rec[0]] = rec[2]
    return S1.score(det_map, ids, gt_map, exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--out", default="benchmark/s_heldout.json")
    args = ap.parse_args()

    out = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        ids, gt_map, exp = build(seed)
        ngt = sum(len(v) for v in gt_map.values())
        print('\n### seed %d: %d images, %d GT boxes' % (seed, len(ids), ngt), flush=True)
        base = run(ids, gt_map, exp, S1, 'base', args.workers)
        print('  base   D=5 %4d/%4d corr_f1=%.1f' %
              (base['5']['tp'], base['5']['true_fp'], base['5']['corr_f1']), flush=True)
        seed_res = {'n_images': len(ids), 'n_gt': ngt, 'base': base, 'items': {}}
        for name, mode, mod in ITEMS:
            props = None
            if name == 'S5':
                S5.CACHE = 'benchmark/_tmp_s5_yolo_seed%d.json' % seed
                props = S5.extract_proposals(ids)
            r = run(ids, gt_map, exp, mod, mode, args.workers, props)
            d = {'variant': mode, 'metrics': r,
                 'dTP5': r['5']['tp'] - base['5']['tp'],
                 'dTrueFP5': r['5']['true_fp'] - base['5']['true_fp'],
                 'dcorr_f1_5': round(r['5']['corr_f1'] - base['5']['corr_f1'], 1),
                 'dTP3': r['3']['tp'] - base['3']['tp']}
            seed_res['items'][name] = d
            print('  %-4s %-9s D=5 %4d/%4d corr_f1=%.1f   dTP5 %+4d dTrueFP5 %+4d '
                  'dcorr_f1 %+5.1f  dTP3 %+4d'
                  % (name, mode, r['5']['tp'], r['5']['true_fp'], r['5']['corr_f1'],
                     d['dTP5'], d['dTrueFP5'], d['dcorr_f1_5'], d['dTP3']), flush=True)
        out[str(seed)] = seed_res

    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nsaved %s' % args.out)


if __name__ == '__main__':
    main()