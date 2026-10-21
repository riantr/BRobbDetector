"""LOOP-40: re-tune the detector on the TRUE ground truth.

Every parameter sweep before 2026-10-20 was measured against
`Compare-V1@CARS_202609_annotated` - a machine artifact, not the hand-drawn
annotation (see proto/gt_final.py).  Worse, the sweep was run with a red mask
that was ~75% X-ray object texture.  After the hue-window fix the mask is
finally clean, so the whole parameter space has to be swept again - this time
against `Label@CARS_202609_raw` (the true GT).

One axis at a time from the production (C#) configuration, exactly like
LOOP-31, and judged with the same GATE: strict improvement in corrected_f1
at D=5 wins, a TIE is an exact match, MARGINAL is within 5 TP and 1.0
corrected_f1.

Caching: the contour list depends only on the mask parameters
(hue_max / sat_min / val_min / denoise_min_area / min_area) and the raw
edgefit quad only on the fit parameters, so both are memoised across the
sweep.  The evaluation mask (hue_max=10, dilated 3x) is packed with
np.packbits so it can be restored without re-reading the image.

Usage:
  python proto/scope40_retune.py                # full sweep
  python proto/scope40_retune.py --axes hue_max ortho_tol
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids, denoise, _side_ok
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, outer_contours,
                       _edgefit_corners, dedup_keep_order)
from quad_ortho import is_orthogonal_quad
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
DIST_VALUES = [3, 5, 8, 12]

# Production configuration (BRobbDetector/Services/CvHybridDetector.cs).
BASE = dict(CQ_DEFAULT)
BASE.update(FIT_DEFAULT)
BASE.update(MASK_DEFAULT)
BASE.update({
    "hue_max": 5, "sat_min": 80, "val_min": 110,
    "denoise_min_area": 40, "min_area": 900.0,
    "min_side": 15.0, "max_side": 800.0, "max_aspect": 12.0,
    "ortho_tol": 10.0, "boundary_band": 6.0, "dedup_iou": 0.30,
    "fit_band": 6.0, "fit_min_pts": 12, "fit_jump": 20.0,
})

# One-axis-at-a-time sweep.  Values are the candidates around BASE.
AXES = {
    "hue_max":        [3, 4, 5, 6, 7, 8, 10],
    "sat_min":        [60, 70, 80, 90, 100],
    "val_min":        [80, 95, 110, 125, 140],
    "denoise_min_area": [0, 20, 40, 60, 100],
    "min_area":       [300, 500, 700, 900, 1200, 1500],
    "min_side":       [8, 12, 15, 20],
    "max_side":       [600, 800, 1200],
    "max_aspect":     [4, 8, 12, 20],
    "ortho_tol":      [5, 8, 10, 12, 15, 20, 30],
    "boundary_band":  [0, 3, 6, 12],
    "dedup_iou":      [0.15, 0.30, 0.45, 0.60],
    "fit_band":       [3, 4, 6, 8, 10],
    "fit_min_pts":    [6, 8, 12, 16],
    "fit_jump":       [10, 15, 20, 30],
}

MASK_KEYS = ("hue_max", "sat_min", "val_min", "denoise_min_area", "min_area")
FIT_KEYS = ("fit_band", "fit_min_pts", "fit_jump")


class Cache:
    """Memoised contours / raw quads / packed evaluation masks."""

    def __init__(self):
        self.contours = {}    # maskkey -> {iid: [contour]}
        self.quads = {}       # (maskkey, fitkey) -> {iid: [quad|None]}
        self.mask = {}        # iid -> (shape, packed bytes)
        self.gt = {}          # iid -> [quad]

    # ---- evaluation mask (hue_max=10, dilated 3x) ----
    def eval_mask(self, iid, img):
        if iid not in self.mask:
            m = cv2.dilate(red_mask(img, 80, 110, 10),
                           np.ones((3, 3), np.uint8), iterations=3)
            self.mask[iid] = (m.shape, np.packbits(m > 0))
        shape, packed = self.mask[iid]
        return np.unpackbits(packed, count=shape[0] * shape[1]) \
            .reshape(shape).astype(bool)

    # ---- contours ----
    def get_contours(self, iid, img, p, maskkey):
        if maskkey not in self.contours:
            self.contours[maskkey] = {}
        store = self.contours[maskkey]
        if iid not in store:
            mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
            if p["denoise_min_area"] > 0:
                mask = denoise(mask, p["denoise_min_area"])
            store[iid] = outer_contours(mask, p)
        return store[iid]

    # ---- raw edgefit quads (before the admission gate) ----
    def get_quads(self, iid, cnts, p, key):
        if key not in self.quads:
            self.quads[key] = {}
        store = self.quads[key]
        if iid not in store:
            store[iid] = [_edgefit_corners(c, p) for c in cnts]
        return store[iid]


def gate_and_dedup(raw, W, H, p):
    kept = []
    for q in raw:
        if q is None:
            continue
        if not _side_ok(q, p):
            continue
        if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                                  boundary_band=p["boundary_band"]):
            continue
        kept.append(q.astype(np.int32))
    return dedup_keep_order(kept, p["dedup_iou"])


def run_config(ids, p, cache, exp, gt_dir=RAW):
    """Detect + evaluate one configuration at every D threshold."""
    maskkey = tuple(p[k] for k in MASK_KEYS)
    fitkey = tuple(p[k] for k in FIT_KEYS)
    acc = {d: {"tp": 0, "fn": 0, "raw_fp": 0, "true_fp": 0, "real_unl": 0}
           for d in DIST_VALUES}
    ndet = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        cnts = cache.get_contours(iid, img, p, maskkey)
        raw = cache.get_quads(iid, cnts, p, (maskkey, fitkey))
        det = gate_and_dedup(raw, W, H, p)
        ndet += len(det)

        if iid not in cache.gt:
            cache.gt[iid] = load_boxes_any(gt_dir, iid)
        gt = cache.gt[iid]
        md = cache.eval_mask(iid, img)
        csv_n = exp.get(iid, 0)
        for d in DIST_VALUES:
            a = acc[d]
            tp, unmatched = match_boxes_point(det, gt, float(d))
            a["tp"] += tp
            a["fn"] += len(gt) - tp
            a["raw_fp"] += len(unmatched)
            for di in unmatched:
                if is_real_unlabeled(det[di], csv_n, len(gt), md):
                    a["real_unl"] += 1
                else:
                    a["true_fp"] += 1

    out = {}
    for d in DIST_VALUES:
        a = acc[d]
        tot = a["tp"] + a["fn"]
        rec = a["tp"] / tot * 100 if tot else 0
        prec = a["tp"] / (a["tp"] + a["true_fp"]) * 100 if (a["tp"] + a["true_fp"]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        out[str(d)] = {
            "tp": a["tp"], "fn": a["fn"], "raw_fp": a["raw_fp"],
            "true_fp": a["true_fp"], "real_unlabeled": a["real_unl"],
            "recall": round(rec, 1), "precision": round(prec, 1),
            "corrected_f1": round(f1, 1),
        }
    return ndet, out


def fmt(name, ndet, per_D):
    return (f"  {name:22s} det={ndet:4d}  " + "  ".join(
        f"D{d}: {per_D[str(d)]['tp']:4d}/{per_D[str(d)]['corrected_f1']:5.1f}"
        for d in DIST_VALUES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="*", default=None,
                    help="subset of axes to sweep (default: all)")
    ap.add_argument("--base", nargs="*", default=[],
                    help="override the base config, e.g. --base sat_min=100")
    ap.add_argument("--out", default="scope40_retune.json")
    args = ap.parse_args()

    ids = get_ids()
    exp = load_expected()
    cache = Cache()
    axes = args.axes if args.axes else list(AXES.keys())
    results = {}

    for kv in args.base:
        k, v = kv.split("=", 1)
        BASE[k] = float(v) if "." in v else int(v)

    print(f"LOOP-40 re-tune on the TRUE GT ({RAW}), {len(ids)} images")
    print(f"base: {json.dumps({k: BASE[k] for k in MASK_KEYS + FIT_KEYS}, default=str)}")

    t0 = time.time()
    ndet, per_D = run_config(ids, BASE, cache, exp)
    results["base"] = {"params": dict(BASE), "det": ndet, "per_D": per_D}
    print(fmt("BASE", ndet, per_D) + f"  ({time.time()-t0:.0f}s)")

    for axis in axes:
        if axis not in AXES:
            print(f"  !! unknown axis {axis}")
            continue
        print(f"\n[{axis}]")
        axis_rows = {}
        for v in AXES[axis]:
            p = dict(BASE)
            p[axis] = v
            t1 = time.time()
            ndet, per_D = run_config(ids, p, cache, exp)
            key = f"{axis}={v}"
            axis_rows[key] = {"params": {axis: v}, "det": ndet, "per_D": per_D}
            results[f"{axis}={v}"] = axis_rows[key]
            print(fmt(key, ndet, per_D) + f"  ({time.time()-t1:.0f}s)")

    path = os.path.join(ROOT, "benchmark", args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "gt": RAW, "dist_values": DIST_VALUES,
                   "base_params": {k: BASE[k] for k in MASK_KEYS + FIT_KEYS},
                   "axes": {a: AXES[a] for a in axes},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()