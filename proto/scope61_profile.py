"""LOOP-61: measure the SYSTEMATIC bias of the hole side fit.

74 of the 198 remaining FNs have a perfectly clean single hole inside the GT
box - the box IS found, only its quad is more than 5px off.  If the four
side lines are systematically shifted inwards (or outwards) the CORNERS move
by ~1.41x the shift, which alone explains a 5-10px corner error.

This script aggregates, over all true positives of the hole pipeline, the red
fraction around every fitted side line as a function of the perpendicular
offset s.  The peak of that profile is the true stroke centre; the current
fit is at s=0.

Usage: python proto/scope61_profile.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import match_boxes_point, corner_match_distance
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of, _hole_quad,
                           _hole_side_lines, _side_geom, detect_holes, detect_edgefit)
from cv_hybrid import dedup_keep_order
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)
PIPE = dict(PIPE_DEFAULT)
OFFS = np.arange(-8.0, 8.5, 1.0)


def side_profile(mask, q, nsamp=None):
    """Red fraction around each side of a quad -> (4, len(OFFS))."""
    H, W = mask.shape
    out = np.zeros((4, len(OFFS)))
    for i in range(4):
        a = np.asarray(q[i], float)
        b = np.asarray(q[(i + 1) % 4], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        ns = nsamp or max(int(L), 8)
        ts = np.linspace(0.0, 1.0, ns)
        base = a[None, :] + ts[:, None] * d[None, :]
        for k, s in enumerate(OFFS):
            p = base + s * n[None, :]
            xs = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
            out[i, k] = float((mask[ys, xs] > 0).mean())
    return out


def main():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    acc = np.zeros((4, len(OFFS)))
    n = 0
    hist = collections.Counter()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map[iid]
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        dets = detect_holes(img, P, CFG, PIPE) + detect_edgefit(img, P, CFG, PIPE)
        dets = dedup_keep_order(dets, PIPE["dedup_iou"])
        tp, unmatched = match_boxes_point(dets, gt, 5.0)
        for di in range(len(dets)):
            g = None
            for gj, gg in enumerate(gt):
                if corner_match_distance(dets[di], gg) <= 5.0:
                    g = gg
                    break
            if g is None:
                continue
            # only the pure hole detections (not the edgefit fallback)
            acc += side_profile(mask, dets[di].astype(float))
            n += 1
    print(f"TP sides averaged: {n}")
    prof = acc / max(n, 1)
    print("\noffset:  " + " ".join(f"{s:6.1f}" for s in OFFS))
    for i in range(4):
        print(f"side {i}:  " + " ".join(f"{v:6.2f}" for v in prof[i]))
    mean = prof.mean(axis=0)
    print("mean :  " + " ".join(f"{v:6.2f}" for v in mean))
    k = int(np.argmax(mean))
    print(f"\npeak at offset {OFFS[k]:+.1f} (value {mean[k]:.3f}); "
          f"value at 0 = {mean[len(OFFS)//2]:.3f}")
    out = os.path.join(ROOT, "benchmark", "scope61_profile.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "offs": OFFS.tolist(),
                   "per_side": prof.tolist(), "mean": mean.tolist()},
                  f, ensure_ascii=False, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()