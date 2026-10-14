"""CV-domain hybrid scope: classical-CV quad extraction algorithms on the red mask.

Families (all consume the same denoised outer contours):
  contour : adaptive approxPolyDP        (current winner, canny_quad.detect_contour)
  hull    : convexHull + adaptive approxPolyDP
  mar     : minAreaRect -> boxPoints     (rotating calipers)
  sector  : 4 farthest points along rect-diagonal directions
  edgefit : sector corners + adjacent-side TLS line intersection refinement
Hybrids: priority-dedup unions (first method wins overlaps):
  hybrid_cem = contour -> edgefit -> mar
  hybrid_all = contour -> edgefit -> mar -> hull -> sector

Usage:
  python proto/cv_hybrid.py                       # detect 500 ids, eval at D=3/5/8/12
  python proto/cv_hybrid.py --viz <id> --method edgefit
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, iou_obb, load_expected
from quad_ortho import is_orthogonal_quad
from canny_quad import (denoise, _quad_adaptive, _side_ok, _order_quad,
                        detect_contour, get_ids, to_json, DEFAULT as CQ_DEFAULT)
from eval_pointmatch import load_boxes, match_boxes_point

ROOT = r"D:\src\CodeArts\BRobbDetector"
DIST_VALUES = [3, 5, 8, 12]
METHODS = ["contour", "hull", "mar", "sector", "edgefit"]
HYBRIDS = {
    "hybrid_cem": ["contour", "edgefit", "mar"],
    "hybrid_all": ["contour", "edgefit", "mar", "hull", "sector"],
}
FIT_DEFAULT = {"fit_band": 6.0, "fit_min_pts": 12, "fit_jump": 20.0}


def outer_contours(mask, p):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        if cv2.contourArea(c) < p["min_area"]:
            continue
        out.append(c)
    return out


def _pass(q, W, H, p):
    if q is None or not _side_ok(q, p):
        return None
    if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                              boundary_band=p["boundary_band"]):
        return None
    return q.astype(np.int32)


def dedup_keep_order(quads, iou_thr):
    keep = []
    for r in quads:
        if all(iou_obb(r, k) < iou_thr for k in keep):
            keep.append(r)
    return keep


def _prep(img, p):
    mask = red_mask(img)
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    return mask


def quads_hull(img, p):
    H, W = img.shape[:2]
    mask = _prep(img, p)
    out = []
    for c in outer_contours(mask, p):
        h = cv2.convexHull(c)
        q = _pass(_quad_adaptive(h, p["approx_eps_list"]), W, H, p)
        if q is not None:
            out.append(q)
    return dedup_keep_order(out, p["dedup_iou"])


def quads_mar(img, p):
    H, W = img.shape[:2]
    mask = _prep(img, p)
    out = []
    for c in outer_contours(mask, p):
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if rw < 1 or rh < 1:
            continue
        q = _pass(cv2.boxPoints(cv2.minAreaRect(c)), W, H, p)
        if q is not None:
            out.append(q)
    return dedup_keep_order(out, p["dedup_iou"])


def _sector_corners(c):
    pts = c.reshape(-1, 2).astype(float)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw < 1 or rh < 1:
        return None
    a = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    b = np.array([-a[1], a[0]])
    v = pts - np.array([cx, cy])
    corners = []
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            proj = v @ (su * a + sv * b)
            corners.append(pts[int(np.argmax(proj))])
    return np.array(corners, float)


def _fit_line_tls(pts):
    mu = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    return mu, vt[0]


def _line_isect(m1, d1, m2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    rhs = np.array([m2[0] - m1[0], m2[1] - m1[1]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(A, rhs)
    return m1 + t[0] * d1


def _edgefit_corners(c, p):
    sec = _sector_corners(c)
    if sec is None:
        return None
    o = _order_quad(sec)
    pts = c.reshape(-1, 2).astype(float)
    lines = []
    for i in range(4):
        a, b = o[i], o[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 2:
            return None
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = pts - a
        along = rel @ u
        perp = rel @ n
        sel = ((along > -0.10 * L) & (along < 1.10 * L) &
               (np.abs(perp) < p["fit_band"]))
        if int(sel.sum()) < p["fit_min_pts"]:
            return None
        lines.append(_fit_line_tls(pts[sel]))
    refined = []
    for i in range(4):
        x = _line_isect(*lines[i - 1], *lines[i])
        if x is None or np.linalg.norm(x - o[i]) > p["fit_jump"]:
            return None
        refined.append(x)
    return np.array(refined, float)


def quads_sector(img, p):
    H, W = img.shape[:2]
    mask = _prep(img, p)
    out = []
    for c in outer_contours(mask, p):
        q = _pass(_sector_corners(c), W, H, p)
        if q is not None:
            out.append(q)
    return dedup_keep_order(out, p["dedup_iou"])


def quads_edgefit(img, p):
    H, W = img.shape[:2]
    mask = _prep(img, p)
    out = []
    for c in outer_contours(mask, p):
        q = _pass(_edgefit_corners(c, p), W, H, p)
        if q is not None:
            out.append(q)
    return dedup_keep_order(out, p["dedup_iou"])


DETECTORS = {
    "contour": lambda img, p: detect_contour(img, p),
    "hull": quads_hull,
    "mar": quads_mar,
    "sector": quads_sector,
    "edgefit": quads_edgefit,
}


def run_detect(ids, method, p):
    out_dir = os.path.join(ROOT, f"tmp_cv_{method}")
    os.makedirs(out_dir, exist_ok=True)
    det_map = {}
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        quads = DETECTORS[method](img, p)
        det_map[iid] = quads
        total += len(quads)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, quads), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            print(f"  [{method} {n+1}/{len(ids)}] boxes={total}")
    print(f"  {method}: {total} boxes -> {out_dir}")
    return det_map


def evaluate(ids, gt_map, det_map, dist_thr):
    TP = raw_fp = 0
    for iid in ids:
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        tp, unmatched = match_boxes_point(det, gt_map[iid], dist_thr)
        TP += tp
        raw_fp += len(unmatched)
    return TP, raw_fp


def viz(iid, method, p):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if method in DETECTORS:
        quads = DETECTORS[method](img, p)
    else:
        order = HYBRIDS[method]
        per = {m: DETECTORS[m](img, p) for m in order}
        quads = dedup_keep_order(
            [q for m in order for q in per[m]], p["dedup_iou"])
    vis = img.copy()
    for q in quads:
        cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 0), 2)
    gt_path = os.path.join(LAB, iid + ".json")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        for info in gt.get("infos", []):
            pts = np.array([[int(x), int(y)] for x, y in
                            (pt.split(",") for pt in info["points"].split(";"))], np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 0), 1)
    out_dir = os.path.join(ROOT, "tmp_cv_viz")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{iid}_{method}.jpg")
    cv2.imwrite(out, vis)
    print(f"{iid} [{method}]: {len(quads)} quads -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default=None)
    ap.add_argument("--method", default="edgefit",
                    choices=METHODS + list(HYBRIDS))
    args = ap.parse_args()

    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)

    if args.viz:
        viz(args.viz, args.method, p)
        return

    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"CV hybrid scope on {len(ids)} images, {total_gt} GT boxes, D={DIST_VALUES}")

    det_maps = {}
    for m in METHODS:
        det_maps[m] = run_detect(ids, m, p)
    for hname, order in HYBRIDS.items():
        det_maps[hname] = {
            iid: dedup_keep_order(
                [q for m in order for q in det_maps[m].get(iid, [])],
                p["dedup_iou"])
            for iid in ids
        }

    results = {}
    print(f"\n{'method':12s} {'det':>5s}  " +
          "  ".join(f"D={d:<15d}" for d in DIST_VALUES))
    for name in METHODS + list(HYBRIDS):
        dm = det_maps[name]
        ndet = sum(len(v) for v in dm.values())
        row = {}
        for D in DIST_VALUES:
            tp, fp = evaluate(ids, gt_map, dm, D)
            row[str(D)] = {"tp": tp, "raw_fp": fp}
        results[name] = {"det": ndet, "per_D": row}
        print(f"{name:12s} {ndet:5d}  " + "  ".join(
            f"TP={row[str(d)]['tp']:4d} FP={row[str(d)]['raw_fp']:4d}"
            for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_hybrid_scope.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(ids), "total_gt": total_gt,
            "dist_values": DIST_VALUES, "config": dict(FIT_DEFAULT),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()