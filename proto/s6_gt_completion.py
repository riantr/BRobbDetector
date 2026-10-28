"""S6: GT completion - a SEPARATE extended-GT artifact, never a detector change.

The shipped detector emits 35 detections at D=5 that match no GT box and pass the
`is_real_unlabeled` heuristic (a red-supported rectangle around an annotation
that the hand label set does not contain).  LOOP-20/26 measured that adding such
boxes to the GT raises recall 67.3% -> 76.3% with ZERO detection change.

This emits the proposal as its own artifact (`benchmark/s6_extended_gt.json`) with
a tiered rule, plus a re-scored table.  It does NOT touch the 1076-box GT used to
judge S1-S5, and it does NOT modify any file under the GT directory.  The
re-scored numbers are an UPPER BOUND and are contingent on human confirmation -
they must never be published as a detector improvement.

Tiers
  A  red support >= 0.80 and all 4 sides >= 0.60   (strong: a drawn box)
  B  red support >= 0.60 and all 4 sides >= 0.40   (probable)
  C  is_real_unlabeled but below B                  (weak: heuristic only)

Usage: python proto/s6_gt_completion.py --workers 12
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
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def side_support(mask, q):
    from scope45_holes import _red_support
    return float(_red_support(mask, np.asarray(q, float)))


def _work(task):
    iid, = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, [], None
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    eval_mask = cv2.dilate(red_mask(img, 80, 110, 10),
                           np.ones((3, 3), np.uint8), iterations=3) > 0
    sup = [side_support(m, q) for q in det]
    return iid, [(np.asarray(q, np.int32).tolist(), s) for q, s in zip(det, sup)], \
        eval_mask.astype(np.uint8).tolist()


def tier_of(sup):
    if sup >= 0.80:
        return 'A'
    if sup >= 0.60:
        return 'B'
    return 'C'


def score_against(det_map, gt_map, ids, D=5.0):
    tp = fp = ndet = 0
    for iid in ids:
        det = det_map.get(iid, [])
        gt = gt_map.get(iid, [])
        ndet += len(det)
        t, unmatched = match_boxes_point(det, gt, float(D))
        tp += t
        fp += len(unmatched)
    ngt = sum(len(gt_map.get(i, [])) for i in ids)
    rec = tp / max(ngt, 1) * 100
    prec = tp / max(tp + fp, 1) * 100
    return {"tp": tp, "fp": fp, "n_det": ndet, "ngt": ngt,
            "recall": round(rec, 1), "precision": round(prec, 1),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/s6_extended_gt.json")
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()
    ngt0 = sum(len(v) for v in gt_map.values())
    print('%d images, %d GT boxes' % (len(ids), ngt0), flush=True)

    det_map, additions = {}, []
    with Pool(args.workers) as pool:
        for k, (iid, det, _em) in enumerate(pool.imap_unordered(
                _work, [(i,) for i in ids], chunksize=3)):
            det_map[iid] = [np.asarray(q, np.int32) for q, _s in det]
            gt = gt_map[iid]
            _t, unmatched = match_boxes_point(det_map[iid], gt, 5.0)
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            em = cv2.dilate(red_mask(img, 80, 110, 10),
                            np.ones((3, 3), np.uint8), iterations=3) > 0
            for di in unmatched:
                q, s = det[di]
                qa = np.asarray(q, np.int32)
                if not is_real_unlabeled(qa, exp.get(iid, 0), len(gt), em):
                    continue
                additions.append({"id": iid, "quad": qa.tolist(),
                                   "support": round(float(s), 3),
                                   "tier": tier_of(float(s))})
            if (k + 1) % 100 == 0:
                print('  %d/%d' % (k + 1, len(ids)), flush=True)

    from collections import Counter
    tc = Counter(a['tier'] for a in additions)
    print('\nreal_unlabeled detections promoted as GT candidates: %d  %s'
          % (len(additions), dict(tc)))

    base_scores = score_against(det_map, gt_map, ids)
    print('\nbaseline against the 1076 GT: %s' % json.dumps(base_scores))

    rescored = {}
    for name, tiers in (('A', {'A'}), ('A+B', {'A', 'B'}),
                        ('A+B+C', {'A', 'B', 'C'})):
        ext = {k: list(v) for k, v in gt_map.items()}
        for a in additions:
            if a['tier'] in tiers:
                ext[a['id']].append(np.asarray(a['quad'], np.int32))
        s = score_against(det_map, ext, ids)
        rescored[name] = s
        print('  extended GT %-6s ngt=%4d  TP %4d  recall %5.1f  prec %5.1f  f1 %5.1f'
              % (name, s['ngt'], s['tp'], s['recall'], s['precision'], s['f1']))

    out = {
        'synthetic_gt_proposal': True,
        'warning': ('UPPER BOUND, contingent on human confirmation. The promoted '
                    'boxes come from the detector\'s own is_real_unlabeled '
                    'heuristic, so the re-scored numbers are circular until a '
                    'human confirms each box. The 1076-box GT is untouched.'),
        'n_gt': ngt0, 'n_candidates': len(additions),
        'tiers': dict(tc), 'baseline': base_scores, 'rescored': rescored,
        'additions': additions,
    }
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nsaved %s' % args.out)


if __name__ == '__main__':
    main()