"""LOOP-41: re-do the small-target / min_area analysis on the TRUE ground truth.

LOOP-32 (2026-10-19) concluded that pm_v5_2's `min_area=1500` drops small
targets, using a case study on 20260905_062030_276 (an image that is NOT in
the 500-image benchmark).  That conclusion was drawn against the fake GT
(Compare-V1) and with the flooded hue<=10 mask, so it has to be re-measured
against `Label@CARS_202609_raw` with the clean hue<=5 mask.

For every GT box that the production config does NOT match at D=5 the script
records WHY:

  no_red        the GT box is not drawn in the input image at all
                (red support < 0.3) - the GT and the image disagree
  tiny          the GT polygon area is below min_area - a genuine small
                target that the area filter removes
  merged        the GT box is swallowed by a larger red component (the
                text inside the box merges with its outline)
  imprecise     a detection exists but its corners are 5-30 px away
  missed        nothing close was detected at all

It also reports the size distribution of the GT boxes (how many are small
targets at all) and the effect of min_area on the real benchmark.

Usage: python proto/scope41_small_targets.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import (match_boxes_point, is_real_unlabeled, red_support,
                             corner_match_distance)
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids, denoise
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, outer_contours,
                       _edgefit_corners, dedup_keep_order)
from quad_ortho import is_orthogonal_quad
from canny_quad import _side_ok
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
HAND = os.path.join(ROOT, "data", "Step3@test")
DIST = 5.0

BASE = dict(CQ_DEFAULT)
BASE.update(FIT_DEFAULT)
BASE.update(MASK_DEFAULT)
BASE.update({
    "hue_max": 4, "sat_min": 100, "val_min": 140,
    "denoise_min_area": 40, "min_area": 300.0,
    "min_side": 12.0, "max_side": 800.0, "max_aspect": 12.0,
    "ortho_tol": 10.0, "boundary_band": 6.0, "dedup_iou": 0.30,
    "fit_band": 8.0, "fit_min_pts": 12, "fit_jump": 20.0,
})


def detect(img, p):
    H, W = img.shape[:2]
    mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    out = []
    for c in outer_contours(mask, p):
        q = _edgefit_corners(c, p)
        if q is None or not _side_ok(q, p):
            continue
        if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                                  boundary_band=p["boundary_band"]):
            continue
        out.append(q.astype(np.int32))
    return dedup_keep_order(out, p["dedup_iou"])


def box_stats(b):
    sides = [float(np.linalg.norm(b[(i + 1) % 4] - b[i])) for i in range(4)]
    return {
        "area": float(cv2.contourArea(b.astype(np.float32))),
        "min_side": min(sides), "max_side": max(sides),
        "aspect": max(sides) / max(min(sides), 1e-6),
    }


def classify(gt, det, contours, mask_d, mask_d5):
    """Why is this GT box not matched?"""
    st = box_stats(gt)
    sup10 = red_support(gt, mask_d)
    sup5 = red_support(gt, mask_d5)
    best = min((corner_match_distance(d, gt) for d in det), default=1e9)

    # which contour (area 0 filter) covers the GT centre?
    cxx, cyy = int(round(gt[:, 0].mean())), int(round(gt[:, 1].mean()))
    cover_area = 0.0
    for c in contours:
        if cv2.pointPolygonTest(c, (float(cxx), float(cyy)), False) >= 0:
            cover_area = max(cover_area, cv2.contourArea(c))
            break

    if sup5 < 0.3:
        reason = "no_red"
    elif st["area"] < BASE["min_area"]:
        reason = "tiny"
    elif cover_area > 3 * st["area"]:
        reason = "merged"
    elif best <= 30.0:
        reason = "imprecise"
    else:
        reason = "missed"
    st.update({"support_hue10": round(sup10, 3), "support_hue5": round(sup5, 3),
               "best_det_dist": (None if best > 1e8 else round(best, 1)),
               "cover_contour_area": round(cover_area, 1),
               "reason": reason})
    return st


def run(ids, p, label):
    exp = load_expected()
    reasons = {}
    fn_rows = []
    sizes = []
    tp = fn = 0
    ndet = 0
    t0 = time.time()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes_any(RAW, iid)
        det = detect(img, p)
        ndet += len(det)

        mask_d = cv2.dilate(red_mask(img, 80, 110, 10),
                            np.ones((3, 3), np.uint8), iterations=3)
        mask_d5 = cv2.dilate(red_mask(img, 80, 110, 5),
                             np.ones((3, 3), np.uint8), iterations=3)
        # contours with NO area filter, to see what the GT box really merged into
        raw_mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
        p0 = dict(p); p0["min_area"] = 0.0
        contours = outer_contours(raw_mask, p0)

        t, _ = match_boxes_point(det, gt, DIST)
        tp += t
        fn += len(gt) - t
        matched = set()
        cands = []
        for di in range(len(det)):
            for gj in range(len(gt)):
                d = corner_match_distance(det[di], gt[gj])
                if d <= DIST:
                    cands.append((d, di, gj))
        cands.sort()
        used_d, used_g = set(), set()
        for d, di, gj in cands:
            if di in used_d or gj in used_g:
                continue
            used_d.add(di); used_g.add(gj); matched.add(gj)

        for gj, g in enumerate(gt):
            st = box_stats(g)
            sizes.append(st["area"])
            if gj in matched:
                continue
            info = classify(g, det, contours, mask_d, mask_d5)
            info["id"] = iid
            reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
            fn_rows.append(info)

    sizes = np.array(sizes)
    out = {
        "label": label, "n_images": len(ids), "det": ndet,
        "tp": tp, "fn": fn,
        "gt_box_area": {
            "min": float(sizes.min()), "p10": float(np.percentile(sizes, 10)),
            "median": float(np.median(sizes)), "p90": float(np.percentile(sizes, 90)),
            "max": float(sizes.max()),
            "below_300": int((sizes < 300).sum()),
            "below_900": int((sizes < 900).sum()),
            "below_1500": int((sizes < 1500).sum()),
        },
        "fn_reasons": reasons,
        "fn_detail": fn_rows,
    }
    print(f"\n=== {label} ===  {len(ids)} imgs, det={ndet}, TP={tp}, FN={fn} "
          f"({time.time()-t0:.0f}s)")
    print(f"  GT box area: min={sizes.min():.0f} p10={np.percentile(sizes,10):.0f} "
          f"med={np.median(sizes):.0f} p90={np.percentile(sizes,90):.0f} max={sizes.max():.0f}")
    print(f"  GT boxes below area 300/900/1500: "
          f"{out['gt_box_area']['below_300']}/{out['gt_box_area']['below_900']}"
          f"/{out['gt_box_area']['below_1500']}")
    print(f"  FN reasons: {reasons}")
    return out


def main():
    ids = get_ids()
    results = {"base": run(ids, BASE, "BASE (LOOP-40 optimum)")}

    # min_area sensitivity on the real benchmark
    print("\n=== min_area sensitivity (true GT) ===")
    sweep = {}
    for ma in [150, 300, 500, 700, 900, 1200, 1500]:
        p = dict(BASE); p["min_area"] = float(ma)
        t0 = time.time()
        tp = fn = ndet = 0
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            det = detect(img, p)
            gt = load_boxes_any(RAW, iid)
            t, _ = match_boxes_point(det, gt, DIST)
            tp += t; fn += len(gt) - t; ndet += len(det)
        sweep[ma] = {"det": ndet, "tp": tp, "fn": fn}
        print(f"  min_area={ma:5d}  det={ndet:4d}  TP={tp:4d}  FN={fn:4d} "
              f"({time.time()-t0:.0f}s)")
    results["min_area_sweep"] = sweep

    # the 40 hand-labelled images that are NOT in the 500-image benchmark
    hand_ids = sorted(f[:-5] for f in os.listdir(HAND) if f.endswith(".jpg"))
    present = [i for i in hand_ids if os.path.exists(os.path.join(IMG, i + ".jpg"))]
    if present:
        res = run(present, BASE, "HAND-LABEL set (data/Step3@test)")
        results["hand"] = res

    path = os.path.join(ROOT, "benchmark", "scope41_small_targets.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gt": RAW, "base_params": {k: BASE[k] for k in
                   ("hue_max", "sat_min", "val_min", "denoise_min_area", "min_area",
                    "min_side", "max_side", "max_aspect", "ortho_tol",
                    "boundary_band", "dedup_iou", "fit_band", "fit_min_pts",
                    "fit_jump")}, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()