"""Parity harness for the OpenCvSharp4 -> OpenCvSharp5 upgrade.

Sub-commands:
  prep   <indir>            copy the 500 benchmark images into <indir>
  compare <dirA> <dirB>    box-level diff between two detection dirs
  eval   <dir>             point-match evaluation of <dir> against benchmark GT

The point-match metric is the project objective: a detection is TP only if all
4 corners are within D px of the assigned GT corners (optimal 4-to-4 perm).
"""
import os, sys, json, shutil, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

from common import IMG, LAB, load_expected
from eval_pointmatch import (load_boxes, match_boxes_point, is_real_unlabeled,
                             red_mask)

DIST_VALUES = [3, 5, 8, 12]


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    return ids[:500]


def cmd_prep(args):
    ids = get_ids()
    os.makedirs(args.indir, exist_ok=True)
    n = 0
    for iid in ids:
        src = os.path.join(IMG, iid + ".jpg")
        dst = os.path.join(args.indir, iid + ".jpg")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        n += 1
    print(f"prepared {n} images -> {args.indir}")


def cmd_compare(args):
    ids = get_ids()
    a_dir, b_dir = args.dir_a, args.dir_b
    tot_a = tot_b = 0
    same = 0
    reorder_only = 0
    diff_imgs = []
    deltas = []
    boxcount_diff = []
    for iid in ids:
        a = load_boxes(a_dir, iid)
        b = load_boxes(b_dir, iid)
        tot_a += len(a)
        tot_b += len(b)
        if len(a) != len(b):
            boxcount_diff.append((iid, len(a), len(b)))
        img_max_delta = 0.0
        unmatched = []
        for ba in a:
            best = None
            for bb in b:
                d = float(np.abs(ba.astype(float) - bb.astype(float)).max())
                # order-invariant: min over rotations/reflections of the 4 corners
                rots = [np.roll(bb.astype(float), k, axis=0) for k in range(4)]
                rots += [np.roll(bb.astype(float)[::-1], k, axis=0) for k in range(4)]
                dm = min(float(np.abs(ba.astype(float) - r).max()) for r in rots)
                if best is None or dm < best:
                    best = dm
            if best > 0:
                unmatched.append((iid, best))
            img_max_delta = max(img_max_delta, best or 0.0)
        if img_max_delta == 0:
            same += 1
        elif img_max_delta <= 1.5 and len(a) == len(b):
            reorder_only += 1
        else:
            diff_imgs.append((iid, len(a), len(b), img_max_delta))
        deltas.append(img_max_delta)
    deltas = np.array(deltas)
    print(f"images compared    : {len(ids)}")
    print(f"pixel-identical   : {same}")
    print(f"<=1.5px rounding  : {reorder_only}")
    print(f"materially differ : {len(diff_imgs)}")
    print(f"boxes A / B       : {tot_a} / {tot_b}")
    print(f"box-count differs : {len(boxcount_diff)}")
    if len(deltas):
        print(f"per-image max corner delta: mean={deltas.mean():.2f} "
              f"p50={np.median(deltas):.2f} p90={np.percentile(deltas, 90):.2f} "
              f"max={deltas.max():.2f}")
    for row in diff_imgs[:15]:
        print("   ", row)
    if len(diff_imgs) > 15:
        print(f"    ... and {len(diff_imgs) - 15} more")
    for row in boxcount_diff[:10]:
        print("   boxcount:", row)


def cmd_eval(args):
    ids = get_ids()
    exp = load_expected()
    for D in DIST_VALUES:
        TP = FN = raw_fp = true_fp = real_unl = 0
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            gt = load_boxes(LAB, iid)
            det = [d.astype(np.int32) for d in load_boxes(args.dir, iid)]
            mask = red_mask(img)
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
            csv_n = exp.get(iid, 0)
            tp, unmatched = match_boxes_point(det, gt, D)
            TP += tp
            FN += len(gt) - tp
            raw_fp += len(unmatched)
            for di in unmatched:
                if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                    real_unl += 1
                else:
                    true_fp += 1
        total_gt = TP + FN
        recall = TP / total_gt * 100 if total_gt else 0
        prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
        f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0
        print(f"  D={D:2d}: TP={TP:4d}  FN={FN:4d}  raw_fp={raw_fp:4d}  "
              f"true_fp={true_fp:4d}  real_unl={real_unl:4d}  "
              f"recall={recall:5.1f}%  corr_f1={f1_c:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("indir")
    p.set_defaults(func=cmd_prep)
    p = sub.add_parser("compare")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("eval")
    p.add_argument("dir")
    p.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()