"""STEP 3 (loop2): labels at 100x scale from the ink.

Licence: the GT arbitration established that the hand annotation is burned into
the input image (raw red support 0.979 mean, 97.6% of GT boxes >= 0.8), so a red
rectangle the pipeline validates IS an annotation.  This runs pm_v10 over
unlabelled pool images (image + expected count, no coordinate label), emits a
label manifest, and reports the self-consistency statistics that can be computed
WITHOUT labels: the emitted-count vs CSV-expected-count agreement and the red
support distribution.

HONEST LIMIT, recorded in the artifact: the corpus inherits pm_v10's 83.7%
recall / 87.1% precision.  Boxes pm_v10 cannot see never enter the labels, so the
corpus can buy robustness and precision but CANNOT recover the 175 FN.

Usage: python proto/pl2_pseudo_labels.py --workers 12 --n 5000 --seed 11
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
MANIFEST_DIR = 'tmp_loop2_pseudo'
OUT = 'benchmark/loop2_pseudo_labels.json'


def build_pool():
    """Images with a jpg and an expected count but NO coordinate label."""
    exp = load_expected()
    bench = set(get_ids())
    pool = []
    for iid, v in exp.items():
        if iid in bench or v < 1:
            continue
        if not os.path.exists(os.path.join(IMG, iid + '.jpg')):
            continue
        if len(load_boxes_any(RAW, iid)) > 0:
            continue
        pool.append(iid)
    pool.sort()
    return pool, exp


def _work(iid):
    img = cv2.imread(os.path.join(IMG, iid + '.jpg'))
    if img is None:
        return iid, [], []
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    quads, sups = [], []
    for q in det:
        qf = np.asarray(q, float)
        sups.append(round(float(_red_support(m, qf)), 3))
        quads.append(np.asarray(q, np.int32).tolist())
    return iid, quads, sups


def corpus_inventory(sample=500, seed=5):
    """Measure the REAL labelled corpus, which is far larger than the benchmark.

    Used to test step 3's own premise: if coordinate labels already exist at
    scale, pseudo-labelling is unnecessary.
    """
    import json as _json
    import random
    files = sorted(f for f in os.listdir(RAW) if f.endswith('.json'))
    random.seed(seed)
    samp = random.sample(files, min(sample, len(files)))
    has_img = nonempty = boxes = 0
    for f in samp:
        iid = f[:-5]
        if not os.path.exists(os.path.join(IMG, iid + '.jpg')):
            continue
        has_img += 1
        try:
            d = _json.load(open(os.path.join(RAW, f), encoding='utf-8-sig'))
        except Exception:
            continue
        infos = d.get('infos', [])
        if infos:
            nonempty += 1
        boxes += len(infos)
    per_img = boxes / max(has_img, 1)
    return {
        'label_dir': RAW,
        'label_files_json': len(files),
        'sample': len(samp),
        'sample_with_jpg': has_img,
        'sample_nonempty_rate': round(nonempty / max(has_img, 1), 4),
        'sample_boxes': boxes,
        'boxes_per_image': round(per_img, 3),
        'est_total_boxes': int(per_img * len(files)),
        'note': ('The benchmark uses only the 500 images from canny_quad.get_ids(); '
                 'the label directory already covers ~56k images.  The DL retrain '
                 'in STEP 1 was therefore starved by the 300-image compute cap, '
                 'NOT by label availability.'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()

    pool, exp = build_pool()
    rng = np.random.RandomState(args.seed)
    take = list(pool)
    rng.shuffle(take)
    ids = sorted(take[:args.n])
    print('unlabelled pool: %d images -> sample %d (RandomState(%d), sorted)'
          % (len(pool), len(ids), args.seed), flush=True)

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    t0 = time.time()
    records = []
    done = 0
    with Pool(args.workers) as mp:
        for iid, quads, sups in mp.imap_unordered(_work, ids, chunksize=8):
            records.append({'id': iid, 'quads': quads, 'support': sups})
            done += 1
            if done % 500 == 0:
                print('  %d/%d  (%.1f min)' % (done, len(ids),
                                               (time.time() - t0) / 60), flush=True)
    wall = time.time() - t0
    records.sort(key=lambda r: r['id'])

    n_img = len(records)
    n_quad = sum(len(r['quads']) for r in records)
    with1 = sum(1 for r in records if r['quads'])
    sups = [s for r in records for s in r['support']]
    # self-consistency: emitted count vs the CSV expected count (no labels needed)
    exact = sum(1 for r in records if len(r['quads']) == int(exp.get(r['id'], 0)))
    diffs = [len(r['quads']) - int(exp.get(r['id'], 0)) for r in records]
    hist = {}
    for d in diffs:
        hist[str(d)] = hist.get(str(d), 0) + 1

    manifest = os.path.join(MANIFEST_DIR, 'labels.json')
    json.dump(records, open(manifest, 'w', encoding='utf-8'))

    res = {
        'step': 'STEP 3 - labels at scale from the ink',
        'sampling_rule': ('pool = images with a jpg + expected count >= 1 + no '
                          'coordinate label in Label@CARS_202609_raw, excluding '
                          'the 500 benchmark images; RandomState(%d) shuffle, '
                          'then sorted, take %d' % (args.seed, args.n)),
        'pool_size': len(pool),
        'images_processed': n_img,
        'quads_emitted': n_quad,
        'mean_quads_per_image': round(n_quad / max(n_img, 1), 3),
        'images_with_at_least_one_quad': with1,
        'frac_images_with_quads': round(with1 / max(n_img, 1), 4),
        'quad_support': {'p10': round(float(np.percentile(sups, 10)), 3),
                         'p50': round(float(np.percentile(sups, 50)), 3),
                         'p90': round(float(np.percentile(sups, 90)), 3)}
                        if sups else None,
        'self_consistency': {
            'vs_csv_expected_count_exact': exact,
            'vs_csv_expected_count_exact_rate': round(exact / max(n_img, 1), 4),
            'mean_abs_count_diff': round(float(np.mean(np.abs(diffs))), 3),
            'count_diff_hist': hist,
        },
        'wall_clock_min': round(wall / 60, 2),
        'workers': args.workers,
        'manifest': manifest,
        'inherits_teacher_ceiling': {
            'statement': ('This corpus is generated by pm_v10, whose measured '
                          'recall is 83.7% and precision 87.1% at D=5 on the '
                          '1076-box GT. Boxes pm_v10 does not see never enter '
                          'the labels, so the corpus CANNOT recover the 175 FN '
                          'and must never be used as an acceptance metric.'),
            'teacher_recall_d5': 0.837,
            'teacher_precision_d5': 0.871,
        },
        'verdict': 'DELIVERED (corpus) - not a detector change',
    }
    try:
        res['corpus_inventory'] = corpus_inventory()
    except Exception as e:
        res['corpus_inventory'] = {'error': str(e)[:120]}
    print('\nlabelled corpus: %d json label files, %.1f%% non-empty, ~%d boxes (est)'
          % (res['corpus_inventory'].get('label_files_json', 0),
             100 * res['corpus_inventory'].get('sample_nonempty_rate', 0),
             res['corpus_inventory'].get('est_total_boxes', 0)))
    print('\nimages %d  quads %d  mean %.2f/img  with>=1 %.1f%%  wall %.1f min'
          % (n_img, n_quad, res['mean_quads_per_image'],
             100 * res['frac_images_with_quads'], wall / 60))
    print('self-consistency vs CSV count: exact %.1f%%  mean|diff| %.2f'
          % (100 * res['self_consistency']['vs_csv_expected_count_exact_rate'],
             res['self_consistency']['mean_abs_count_diff']))
    print('quad support p50 %.3f  p90 %.3f'
          % (res['quad_support']['p50'], res['quad_support']['p90']))
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved %s' % OUT)


if __name__ == '__main__':
    main()