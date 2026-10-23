"""LOOP-43: forensics on the 520 false negatives of pm_v7 (true GT, D=5).

pm_v7 matches 602 of 1121 GT boxes at D=5, so 519-520 GT boxes are missed.
LOOP-41 bucketed them coarsely (merged 257 / missed 253 / imprecise 9 /
no_red 1) which is not actionable enough: "merged" and "missed" are symptoms,
not mechanisms.  This script re-derives the mechanism for EVERY missed box by
tracing the exact production pipeline stage that dropped it:

  no_red          the box is not drawn in the input image at all
                  (perimeter red < 0.25 even under loose thresholds)
  weak_red        the box IS drawn but its strokes fail the production
                  colour thresholds (sat>=100 / val>=140 / hue<=4)
  broken          the strokes are red enough but no single connected
                  component covers the perimeter -> the outline has gaps
  filter_edgefit  a component covers the box but edgefit could not fit 4
                  lines (too few points on a side, or a corner jumped
                  more than fit_jump)
  filter_side     the fitted quad violates min_side / max_side / max_aspect
  filter_ortho    the fitted quad is not orthogonal within ortho_tol
  filter_dedup    a valid quad existed but the keep-order IoU dedup threw
                  it away because an earlier quad overlapped it
  merged_text     the box outline is fused with the TEXT inside it -> the
                  component is much bigger than the box and the fitted quad
                  spans the whole blob
  merged_outside  the box outline is fused with red stuff OUTSIDE it (a
                  neighbouring box / object) -> same effect
  merged_mixed    both
  imprecise       a detection exists but its corners are 5-30 px away
  missed          nothing else explains it

Usage:
  python proto/scope43_fn_forensics.py            # full run
  python proto/scope43_fn_forensics.py --viz 8     # dump 8 samples per reason
"""
import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, corner_match_distance
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids, denoise, _side_ok
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, PROD_DEFAULT,
                       outer_contours, _edgefit_corners, dedup_keep_order)
from quad_ortho import is_orthogonal_quad
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
DIST = 5.0

PROD = dict(CQ_DEFAULT)
PROD.update(FIT_DEFAULT)
PROD.update(MASK_DEFAULT)
PROD.update(PROD_DEFAULT)

# Loose colour window used only to decide whether the box is drawn AT ALL.
LOOSE = {"sat_min": 60, "val_min": 80, "hue_max": 8}


def perim_points(box, step=2):
    """Dense sample of the box perimeter."""
    segs = []
    for i in range(4):
        p = box[i].astype(float)
        q = box[(i + 1) % 4].astype(float)
        L = float(np.linalg.norm(q - p))
        n = max(int(L / step), 2)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        segs.append(p[None, :] + ts[:, None] * (q - p)[None, :])
    return np.concatenate(segs, axis=0)


def coverage(pts, mask):
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return float(np.count_nonzero(mask[ys, xs])) / len(pts)


def quad_in(quad, dets, tol=1.0):
    for d in dets:
        if (np.abs(np.asarray(d, float) - np.asarray(quad, float)).max()) <= tol:
            return True
    return False


def ortho_dev(q):
    """Max deviation (deg) of the 4 adjacent angles from 90 deg."""
    q = np.asarray(q, float)
    devs = []
    for i in range(4):
        e1 = q[(i + 1) % 4] - q[i]
        e2 = q[(i + 2) % 4] - q[(i + 1) % 4]
        ang = np.degrees(np.arctan2(e1[0] * e2[1] - e1[1] * e2[0],
                                    e1[0] * e2[0] + e1[1] * e2[1]))
        devs.append(abs(abs(ang) - 90.0))
    return max(devs)


def trace_chain(contour, dets, p, W, H):
    """Which production filter (if any) rejected this contour's quad."""
    q = _edgefit_corners(contour, p)
    if q is None:
        return "filter_edgefit", None
    if not _side_ok(q, p):
        sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        return "filter_side", q
    if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                              boundary_band=p["boundary_band"]):
        return "filter_ortho", q
    if not quad_in(np.asarray(q).astype(np.int32), dets):
        return "filter_dedup", q
    return "ok", q


def detect_prod(img, p):
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


def analyse_image(img, gt, dets):
    """Per-GT forensics for one image. Returns a list of dicts (one per GT)."""
    H, W = img.shape[:2]
    p = PROD
    mask_prod = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
    mask_loose = red_mask(img, LOOSE["sat_min"], LOOSE["val_min"], LOOSE["hue_max"])
    den = denoise(mask_prod, p["denoise_min_area"]) if p["denoise_min_area"] > 0 else mask_prod
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(den, 8)

    # all top-level contours of the denoised mask (NO area filter) so we can
    # see components that the min_area filter removed
    p0 = dict(p); p0["min_area"] = 0.0
    contours = outer_contours(den, p0)
    comp_of_contour = []
    for c in contours:
        x, y = int(c[0][0][0]), int(c[0][0][1])
        comp_of_contour.append(int(lab[y, x]) if 0 <= y < H and 0 <= x < W else 0)

    rows = []
    for gi, g in enumerate(gt):
        pts = perim_points(g)
        prod_cov = coverage(pts, mask_prod)
        loose_cov = coverage(pts, mask_loose)
        gt_area = float(cv2.contourArea(g.astype(np.float32)))

        # which connected component covers the perimeter best?
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        vals = lab[ys, xs]
        vals = vals[vals > 0]
        comp_cov = 0.0
        best_comp = 0
        if len(vals):
            counts = collections.Counter(vals.tolist())
            best_comp, n = counts.most_common(1)[0]
            comp_cov = n / len(pts)

        # split the component's pixels into inside / outside the GT polygon
        inside_frac = None
        comp_area = None
        if best_comp:
            comp_area = float(stats[best_comp, 4])
            poly = np.zeros((H, W), np.uint8)
            cv2.fillPoly(poly, [g.astype(np.int32)], 255)
            comp_mask = (lab == best_comp)
            inside_frac = float(np.count_nonzero(comp_mask & (poly > 0))) / max(comp_area, 1)

        # contour that belongs to that component
        cidx = -1
        for i, cc in enumerate(comp_of_contour):
            if cc == best_comp:
                cidx = i
                break

        best_det = min((corner_match_distance(d, g) for d in dets), default=1e9)

        row = {
            "prod_cov": round(prod_cov, 3),
            "loose_cov": round(loose_cov, 3),
            "comp_cov": round(comp_cov, 3),
            "comp_area": None if comp_area is None else round(comp_area, 1),
            "gt_area": round(gt_area, 1),
            "area_ratio": None if not comp_area else round(comp_area / max(gt_area, 1), 2),
            "inside_frac": None if inside_frac is None else round(inside_frac, 3),
            "best_det_dist": None if best_det > 1e8 else round(best_det, 1),
        }

        if loose_cov < 0.25:
            row["reason"] = "no_red"
        elif prod_cov < 0.25:
            row["reason"] = "weak_red"
        elif comp_cov < 0.25:
            row["reason"] = "broken"
        else:
            if cidx < 0:
                row["reason"] = "merged_unknown"
            else:
                status, q = trace_chain(contours[cidx], dets, p, W, H)
                if status == "ok":
                    # a quad exists and survived - the component is simply
                    # bigger than the box
                    if inside_frac is None:
                        sub = "merged_unknown"
                    elif inside_frac >= 0.5:
                        sub = "merged_text"
                    elif inside_frac <= 0.2:
                        sub = "merged_outside"
                    else:
                        sub = "merged_mixed"
                    row["reason"] = sub
                    row["filter"] = status
                elif status == "filter_dedup":
                    row["reason"] = "filter_dedup"
                    row["filter"] = status
                elif status == "filter_side":
                    row["reason"] = "filter_side"
                    row["filter"] = status
                    if q is not None:
                        sides = [float(np.linalg.norm(q[(i + 1) % 4] - q[i]))
                                 for i in range(4)]
                        row["min_side"] = round(min(sides), 1)
                        row["max_side"] = round(max(sides), 1)
                        row["aspect"] = round(max(sides) / max(min(sides), 1e-6), 2)
                elif status == "filter_ortho":
                    row["reason"] = "filter_ortho"
                    row["filter"] = status
                    if q is not None:
                        row["ortho_dev"] = round(ortho_dev(q), 1)
                        row["q_dist"] = round(corner_match_distance(
                            np.asarray(q).astype(np.int32), g), 1)
                elif status == "filter_edgefit":
                    row["reason"] = "filter_edgefit"
                    row["filter"] = status
                else:
                    row["reason"] = "unknown"
                # did the component's own area pass min_area?
                if comp_area is not None and comp_area < p["min_area"]:
                    row["comp_below_min_area"] = True
        rows.append(row)
    return rows


def viz_samples(rows_by_id, ids, gt_by_id, out_dir, per_reason=8):
    os.makedirs(out_dir, exist_ok=True)
    buckets = collections.defaultdict(list)
    for iid, rows in rows_by_id.items():
        for gi, r in enumerate(rows):
            buckets[r["reason"]].append((iid, gi))
    for reason, items in buckets.items():
        for iid, gi in items[:per_reason]:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            vis = img.copy()
            g = gt_by_id[iid][gi]
            cv2.polylines(vis, [g.astype(np.int32)], True, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, f"{reason}__{iid}__{gi}.jpg"), vis)
    return {k: len(v) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--out", default="scope43_fn_forensics.json")
    args = ap.parse_args()

    ids = get_ids()
    exp = load_expected()
    print(f"LOOP-43 FN forensics on the TRUE GT, {len(ids)} images")

    reasons = collections.Counter()
    reason_dets = collections.Counter()
    fn_rows = []
    gt_by_id = {}
    rows_by_id = {}
    tp = 0
    ndet = 0
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        gt_by_id[iid] = gt
        dets = detect_prod(img, PROD)
        ndet += len(dets)
        t, _ = match_boxes_point(dets, gt, DIST)
        tp += t

        rows = analyse_image(img, gt, dets)
        rows_by_id[iid] = rows

        # mark which GT boxes were matched
        matched = set()
        cands = []
        for di in range(len(dets)):
            for gj in range(len(gt)):
                d = corner_match_distance(dets[di], gt[gj])
                if d <= DIST:
                    cands.append((d, di, gj))
        cands.sort()
        ud, ug = set(), set()
        for d, di, gj in cands:
            if di in ud or gj in ug:
                continue
            ud.add(di); ug.add(gj); matched.add(gj)

        for gi, r in enumerate(rows):
            if gi in matched:
                continue
            r["id"] = iid
            r["gi"] = gi
            reasons[r["reason"]] += 1
            if "det" in r:
                reason_dets[r["reason"]] += r["det"]
            fn_rows.append(r)
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] tp={tp} fn_rows={len(fn_rows)} "
                  f"({time.time()-t0:.0f}s)")

    print(f"\nTP={tp}  FN={len(fn_rows)}  det={ndet}  ({time.time()-t0:.0f}s)")
    print("FN reasons:")
    for k, v in reasons.most_common():
        print(f"  {k:18s} {v:5d}  ({v*100.0/len(fn_rows):5.1f}%)")

    out = {
        "gt": RAW, "n_images": len(ids), "det": ndet, "tp": tp,
        "fn": len(fn_rows), "fn_reasons": dict(reasons),
        "fn_detail": fn_rows,
        "prod_params": {k: PROD[k] for k in
                        ("hue_max", "sat_min", "val_min", "denoise_min_area",
                         "min_area", "min_side", "max_side", "max_aspect",
                         "ortho_tol", "boundary_band", "dedup_iou", "fit_band",
                         "fit_min_pts", "fit_jump")},
    }
    path = os.path.join(ROOT, "benchmark", args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")

    if args.viz:
        d = viz_samples(rows_by_id, ids, gt_by_id,
                        os.path.join(ROOT, "tmp_fn_viz"), args.viz)
        print(f"Viz -> tmp_fn_viz/ {d}")


if __name__ == "__main__":
    main()