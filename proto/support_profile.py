"""Red-support profile of the detections: real data vs emulation.

`min_red_support` is the fraction of the emitted quad's perimeter that must be
red.  It is the only gate that behaves differently on the two datasets, so it
is worth knowing WHY.

For every detection the script measures
  * the red support of the emitted quad,
  * whether the detection is a TP at D=5,
and reports the support distribution of the TPs and of the FPs plus the
"threshold curve" (how many TPs are lost and how many FPs are killed).

Usage:
    python proto/support_profile.py --dataset real
    python proto/support_profile.py --dataset emul --n 1500
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
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import match_boxes_point
from quad_ortho import quad_on_boundary
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _red_support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
DIST_VALUES = [3, 5, 8, 12]

_CFG = {}
_PIPE = {}


def _init(cfg, pipe):
    global _CFG, _PIPE
    _CFG = cfg
    _PIPE = pipe


def _prod_mask(img):
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    return m


def _work_real(iid):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, []
    m = _prod_mask(img)
    det = detect(img, P, _CFG, _PIPE, mask=m)
    return iid, [(np.asarray(q, np.int32), _red_support(m, q)) for q in det]


def _work_emul(task):
    stem, img_dir = task
    from eval_emulation import color_mask
    img = cv2.imread(os.path.join(img_dir, stem + ".jpg"))
    if img is None:
        return stem, []
    m = color_mask(img, "red", P, None)
    m = denoise(m, P["denoise_min_area"])
    det = detect(img, P, _CFG, _PIPE, mask=m)
    return stem, [(np.asarray(q, np.int32), _red_support(m, q)) for q in det]


def pct(v, ps=(1, 5, 10, 25, 50, 75, 90)):
    if not v:
        return {}
    a = np.asarray(v, float)
    return {("p%d" % p): round(float(np.percentile(a, p)), 3) for p in ps}


def curve(sups_tp, sups_fp, ths=np.arange(0.0, 1.001, 0.05)):
    """For each threshold: TPs kept, FPs killed."""
    out = []
    for t in ths:
        out.append({"thr": round(float(t), 2),
                    "tp_kept": int(sum(1 for s in sups_tp if s >= t)),
                    "tp_lost": int(sum(1 for s in sups_tp if s < t)),
                    "fp_kept": int(sum(1 for s in sups_fp if s >= t)),
                    "fp_killed": int(sum(1 for s in sups_fp if s < t))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["real", "emul"], default="real")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--img-dir", default=os.path.join(ROOT, "data", "Image@emul4"))
    ap.add_argument("--lab-dir", default=os.path.join(ROOT, "data", "Label@emul4"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = dict(HOLE_DEFAULT)
    for kv in filter(None, args.cfg.split(",")):
        k, v = kv.split("=")
        try:
            cfg[k] = float(v)
        except ValueError:
            cfg[k] = v
    pipe = dict(PIPE_DEFAULT)

    gt_map = {}
    if args.dataset == "real":
        ids = get_ids()
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                gt_map[iid] = []
                continue
            H, W = img.shape[:2]
            gt_map[iid] = [g for g in load_boxes_any(RAW, iid)
                           if not quad_on_boundary(g, W, H, 0.0)]
        if args.n:
            ids = ids[:args.n]
        work = _work_real
        tasks = ids
    else:
        from eval_emulation import load_gt
        ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in os.listdir(args.lab_dir) if p.endswith(".json"))
        if args.n:
            ids = ids[:args.n]
        for iid in ids:
            gt_map[iid] = [g for g, t in load_gt(iid, args.lab_dir) if t == "红"]
        work = _work_emul
        tasks = [(iid, args.img_dir) for iid in ids]

    print("%s: %d images, %d GT boxes" % (args.dataset, len(ids),
                                          sum(len(gt_map[i]) for i in ids)),
          flush=True)
    det_map = {}
    with Pool(args.workers, initializer=_init, initargs=(cfg, pipe)) as pool:
        for k, (iid, det) in enumerate(pool.imap_unordered(work, tasks, chunksize=4)):
            det_map[iid] = det
            if (k + 1) % 500 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)

    sups_tp = []
    sups_fp = []
    for iid in ids:
        det = [d for d, s in det_map.get(iid, [])]
        sups = [s for d, s in det_map.get(iid, [])]
        gt = gt_map[iid]
        tp, unmatched = match_boxes_point(det, gt, 5.0)
        # match_boxes_point returns the unmatched DETECTION indices
        uset = set(unmatched)
        # recompute which detections are matched
        matched = [i for i in range(len(det)) if i not in uset]
        for i in matched:
            sups_tp.append(sups[i])
        for i in uset:
            sups_fp.append(sups[i])

    print("\nTP detections: n=%d  support %s" % (len(sups_tp), pct(sups_tp)))
    print("FP detections: n=%d  support %s" % (len(sups_fp), pct(sups_fp)))
    print("\n  thr    TP kept  TP lost   FP kept  FP killed")
    for r in curve(sups_tp, sups_fp):
        print("  %.2f   %6d  %6d   %6d  %8d" % (r["thr"], r["tp_kept"], r["tp_lost"],
                                                r["fp_kept"], r["fp_killed"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "n_tp": len(sups_tp),
                       "n_fp": len(sups_fp), "tp_support": pct(sups_tp),
                       "fp_support": pct(sups_fp),
                       "curve": curve(sups_tp, sups_fp)}, f,
                      ensure_ascii=False, indent=1)
        print("saved %s" % args.out)


if __name__ == "__main__":
    main()