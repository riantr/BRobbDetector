"""LOOP-30: Investigate what `mar` (minAreaRect) uniquely recovers.

The prior claim was that mar "recovers rounded/curved contours that can't
reduce to 4 vertices (~95 GT rings)". The user challenges this: annotation
targets are hand-drawn rectangles, NOT rounded shapes.

This script:
  1. Runs the pm_v5_1 pipeline per-method (hull/edgefit/mar) on 500 real imgs.
  2. For each contour, records which methods pass the admission filter.
  3. Finds contours where ONLY mar succeeds (mar-unique).
  4. Characterizes each mar-unique contour:
       - approxPolyDP vertex count at each epsilon in approx_eps_list
       - convexHull vertex count
       - is contour convex?
       - circularity = 4*pi*area / perimeter^2  (1.0=circle, 0.785=square)
       - mar-box vs hull-box IoU (how different is mar's answer?)
       - contour area, perimeter, n_points
  5. Checks if the mar-unique detection is a TP (matches GT at D=5).
  6. Visualizes up to N sample mar-unique contours (contour + mar box + GT).
  7. Also reports per-method TP and the TP that mar uniquely contributes to
     the hybrid (i.e., TP in hybrid that would be lost if mar were removed).

Usage:
  python proto/cv_scope30.py
  python proto/cv_scope30.py --viz 20
"""
import sys, os, json, argparse, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, iou_obb, load_expected
from canny_quad import get_ids, to_json, denoise, _quad_adaptive, DEFAULT as CQ_DEFAULT
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                       FIT_DEFAULT, ROOT, DIST_VALUES, _edgefit_corners)
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from eval_pointmatch import load_boxes, match_boxes_point

D_EVAL = 5.0
MIN_AREA_1000 = 1000.0
VIZ_DIR = os.path.join(ROOT, "tmp_sc30_viz")


def pm51_params():
    p = pm5_params()
    p["min_area"] = MIN_AREA_1000
    return p


def contour_circularity(c):
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if perim < 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (perim * perim)


def approx_vertex_counts(c, eps_list):
    """Number of vertices from approxPolyDP at each epsilon fraction of perimeter."""
    perim = cv2.arcLength(c, True)
    out = []
    for eps in eps_list:
        approx = cv2.approxPolyDP(c, eps * perim, True)
        out.append(len(approx))
    return out


def hull_vertex_counts(c, eps_list):
    h = cv2.convexHull(c)
    perim = cv2.arcLength(h, True)
    out = []
    for eps in eps_list:
        approx = cv2.approxPolyDP(h, eps * perim, True)
        out.append(len(approx))
    return out


def is_convex(c):
    return cv2.isContourConvex(c)


def analyze_image(iid, p, gt):
    """Return per-contour analysis records for one image."""
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return [], None
    H, W = img.shape[:2]
    mask = _prep(img, p)
    cnts = outer_contours(mask, p)
    records = []
    for ci, c in enumerate(cnts):
        base = _base_quads(c, p)
        passed = {}
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            passed[m] = q is not None
        rec = {
            "iid": iid, "cidx": ci,
            "n_pts": len(c),
            "area": float(cv2.contourArea(c)),
            "perim": float(cv2.arcLength(c, True)),
            "circularity": contour_circularity(c),
            "convex": is_convex(c),
            "approx_v": approx_vertex_counts(c, p["approx_eps_list"]),
            "hull_v": hull_vertex_counts(c, p["approx_eps_list"]),
            "passed": passed,
            "mar_only": passed["mar"] and not passed["hull"] and not passed["edgefit"],
            "mar_box": base["mar"],
            "hull_box": base["hull"],
        }
        if base["mar"] is not None and base["hull"] is not None:
            rec["mar_hull_iou"] = float(iou_obb(
                base["mar"].astype(np.int32), base["hull"].astype(np.int32)))
        else:
            rec["mar_hull_iou"] = None
        records.append(rec)
    return records, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", type=int, default=0,
                    help="visualize up to N mar-unique TP contours")
    args = ap.parse_args()

    p = pm51_params()
    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"LOOP-30: mar-unique contour analysis on {len(ids)} imgs, "
          f"{total_gt} GT, min_area={p['min_area']}, D={D_EVAL}")

    all_records = []
    img_cache = {}

    for n, iid in enumerate(ids):
        recs, img = analyze_image(iid, p, gt_map[iid])
        if img is not None:
            img_cache[iid] = img
        all_records.extend(recs)
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] analyzed")

    # build per-method det maps using the standard pipeline
    per_method = {m: {iid: [] for iid in ids} for m in HY_ORDER}
    for n, iid in enumerate(ids):
        img = img_cache.get(iid)
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = _prep(img, p)
        cnts = outer_contours(mask, p)
        for c in cnts:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base[m], W, H, p)
                if q is not None:
                    per_method[m][iid].append(q)
        for m in HY_ORDER:
            per_method[m][iid] = dedup_keep_order(per_method[m][iid], p["dedup_iou"])

    # hybrid with and without mar
    hy_full = {iid: dedup_keep_order(
        [q for m in HY_ORDER for q in per_method[m].get(iid, [])], p["dedup_iou"])
        for iid in ids}
    hy_no_mar = {iid: dedup_keep_order(
        [q for m in ("hull", "edgefit") for q in per_method[m].get(iid, [])], p["dedup_iou"])
        for iid in ids}

    # evaluate TP
    def eval_tp(dm):
        tp = 0
        for iid in ids:
            det = [d.astype(np.int32) for d in dm.get(iid, [])]
            t, _ = match_boxes_point(det, gt_map[iid], D_EVAL)
            tp += t
        return tp

    tp_hull = eval_tp(per_method["hull"])
    tp_edgefit = eval_tp(per_method["edgefit"])
    tp_mar = eval_tp(per_method["mar"])
    tp_full = eval_tp(hy_full)
    tp_no_mar = eval_tp(hy_no_mar)
    n_hull = sum(len(v) for v in per_method["hull"].values())
    n_edgefit = sum(len(v) for v in per_method["edgefit"].values())
    n_mar = sum(len(v) for v in per_method["mar"].values())
    n_full = sum(len(v) for v in hy_full.values())
    n_no_mar = sum(len(v) for v in hy_no_mar.values())

    print(f"\n{'='*72}")
    print(f"PER-METHOD RESULTS (D={D_EVAL:.0f}, min_area={p['min_area']:.0f})")
    print(f"{'='*72}")
    print(f"{'method':10s} {'det':>6s} {'TP':>6s}")
    print(f"{'hull':10s} {n_hull:6d} {tp_hull:6d}")
    print(f"{'edgefit':10s} {n_edgefit:6d} {tp_edgefit:6d}")
    print(f"{'mar':10s} {n_mar:6d} {tp_mar:6d}")
    print(f"{'hy(h+e+m)':10s} {n_full:6d} {tp_full:6d}")
    print(f"{'hy(h+e)':10s} {n_no_mar:6d} {tp_no_mar:6d}")
    print(f"\nmar contribution to hybrid TP: {tp_full - tp_no_mar:+d}")
    print(f"mar det count: {n_mar}, mar-only TP (standalone): {tp_mar}")

    # mar-unique contour characterization
    mar_only_recs = [r for r in all_records if r["mar_only"]]
    print(f"\n{'='*72}")
    print(f"MAR-UNIQUE CONTOURS (mar passes, hull+edgefit both fail)")
    print(f"{'='*72}")
    print(f"total contours: {len(all_records)}")
    print(f"mar-unique contours: {len(mar_only_recs)}")

    if not mar_only_recs:
        print("\n*** NO mar-unique contours found. mar never succeeds alone. ***")
        print("mar is redundant — everything it passes, hull or edgefit also passes.")
    else:
        # aggregate stats
        circ = [r["circularity"] for r in mar_only_recs]
        convex = sum(1 for r in mar_only_recs if r["convex"])
        areas = [r["area"] for r in mar_only_recs]
        # min approx vertices across eps list (best-case reduction)
        min_approx = [min(r["approx_v"]) for r in mar_only_recs]
        min_hull_v = [min(r["hull_v"]) for r in mar_only_recs]
        n_4approx = sum(1 for v in min_approx if v == 4)
        n_ne4 = sum(1 for v in min_approx if v != 4)
        n_4hull = sum(1 for v in min_hull_v if v == 4)

        print(f"\nshape stats over {len(mar_only_recs)} mar-unique contours:")
        print(f"  circularity: min={min(circ):.3f} med={np.median(circ):.3f} "
              f"max={max(circ):.3f}  (square=0.785, circle=1.0)")
        print(f"  convex: {convex}/{len(mar_only_recs)}")
        print(f"  area: min={min(areas):.0f} med={np.median(areas):.0f} max={max(areas):.0f}")
        print(f"  approxPolyDP min-vertices == 4: {n_4approx}/{len(mar_only_recs)} "
              f"(CAN reduce to quad but fails _pass)")
        print(f"  approxPolyDP min-vertices != 4: {n_ne4}/{len(mar_only_recs)} "
              f"(cannot reduce to 4 vertices)")
        print(f"  convexHull approxPolyDP == 4:   {n_4hull}/{len(mar_only_recs)}")

        # distribution of min approx vertices
        from collections import Counter
        vdist = Counter(min_approx)
        print(f"  min approxPolyDP vertex distribution: {dict(sorted(vdist.items()))}")
        vdist_h = Counter(min_hull_v)
        print(f"  min hull approxPolyDP vertex dist:    {dict(sorted(vdist_h.items()))}")

        # how different is mar's box from hull's box?
        ious = [r["mar_hull_iou"] for r in mar_only_recs
                if r["mar_hull_iou"] is not None]
        if ious:
            print(f"  mar-vs-hull IoU (where hull box exists): "
                  f"min={min(ious):.3f} med={np.median(ious):.3f} max={max(ious):.3f}")
        else:
            print(f"  mar-vs-hull IoU: no hull box available for any mar-unique contour")

        # check how many mar-unique detections are TP
        # build a det map of ONLY mar-unique boxes
        mar_unique_det = {iid: [] for iid in ids}
        for r in mar_only_recs:
            q = _pass(r["mar_box"].astype(np.float32), 99999, 99999, p)
            if q is not None:
                mar_unique_det[r["iid"]].append(q)
        for iid in ids:
            mar_unique_det[iid] = dedup_keep_order(mar_unique_det[iid], p["dedup_iou"])
        tp_unique = eval_tp(mar_unique_det)
        n_unique = sum(len(v) for v in mar_unique_det.values())
        print(f"\n  mar-unique boxes (post-dedup): {n_unique}, TP@D={D_EVAL:.0f}: {tp_unique}")

        # CATEGORIZE why hull/edgefit failed
        # Reason A: approxPolyDP never reaches 4 vertices (too many/few verts)
        # Reason B: reaches 4 verts but fails orthogonality/side check (_pass)
        # Reason C: contour is non-convex (hull changes shape)
        cat_A = sum(1 for r in mar_only_recs if min(r["approx_v"]) != 4)
        cat_B = sum(1 for r in mar_only_recs if min(r["approx_v"]) == 4)
        cat_C = sum(1 for r in mar_only_recs if not r["convex"])
        print(f"\n  failure categorization (why hull/edgefit failed):")
        print(f"    A: approxPolyDP != 4 verts at any eps: {cat_A}")
        print(f"    B: approxPolyDP == 4 verts but _pass rejected: {cat_B}")
        print(f"    C: contour non-convex: {cat_C}")

        # visualize
        if args.viz > 0:
            os.makedirs(VIZ_DIR, exist_ok=True)
            # pick TP mar-unique first
            tp_recs = []
            for r in mar_only_recs:
                q = _pass(r["mar_box"].astype(np.float32), 99999, 99999, p)
                if q is None:
                    continue
                det = [q.astype(np.int32)]
                t, _ = match_boxes_point(det, gt_map[r["iid"]], D_EVAL)
                if t > 0:
                    tp_recs.append(r)
            viz_list = tp_recs[:args.viz] if tp_recs else mar_only_recs[:args.viz]
            print(f"\n  visualizing {len(viz_list)} contours -> {VIZ_DIR}")
            for r in viz_list:
                img = img_cache.get(r["iid"])
                if img is None:
                    continue
                vis = img.copy()
                # redraw contour
                mask = _prep(img, p)
                cnts = outer_contours(mask, p)
                c = cnts[r["cidx"]]
                cv2.drawContours(vis, [c], -1, (0, 255, 255), 2)  # yellow = contour
                qm = r["mar_box"].astype(np.int32)
                cv2.polylines(vis, [qm], True, (0, 255, 0), 3)    # green = mar box
                # GT boxes
                for gt_box in gt_map[r["iid"]]:
                    cv2.polylines(vis, [gt_box.astype(np.int32)], True, (255, 0, 0), 2)
                # hull box if exists
                if r["hull_box"] is not None:
                    qh = r["hull_box"].astype(np.int32)
                    cv2.polylines(vis, [qh], True, (0, 0, 255), 2)  # red = hull box
                fname = (f"{r['iid']}_c{r['cidx']}_circ{r['circularity']:.3f}_"
                         f"approx{min(r['approx_v'])}.jpg")
                cv2.imwrite(os.path.join(VIZ_DIR, fname), vis)

    # save
    out = os.path.join(ROOT, "benchmark", "cv_scope30.json")
    summary = {
        "n_images": len(ids), "total_gt": total_gt,
        "min_area": p["min_area"], "D_eval": D_EVAL,
        "per_method": {
            "hull": {"det": n_hull, "tp": tp_hull},
            "edgefit": {"det": n_edgefit, "tp": tp_edgefit},
            "mar": {"det": n_mar, "tp": tp_mar},
            "hy_full": {"det": n_full, "tp": tp_full},
            "hy_no_mar": {"det": n_no_mar, "tp": tp_no_mar},
        },
        "mar_contribution_to_hybrid_tp": tp_full - tp_no_mar,
        "n_total_contours": len(all_records),
        "n_mar_unique_contours": len(mar_only_recs),
    }
    if mar_only_recs:
        summary["mar_unique_stats"] = {
            "circularity": {"min": min(circ), "med": float(np.median(circ)),
                            "max": max(circ)},
            "convex_count": convex,
            "approx_never_4": cat_A,
            "approx_4_but_pass_fail": cat_B,
            "non_convex": cat_C,
            "min_approx_v_dist": dict(sorted(vdist.items())),
            "n_mar_unique_boxes": n_unique,
            "tp_mar_unique": tp_unique,
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


def _contour_for_rec(iid, rec, p):
    """Unused helper (kept for refactor safety)."""
    return None


if __name__ == "__main__":
    main()