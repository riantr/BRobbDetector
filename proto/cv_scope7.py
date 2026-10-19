"""LOOP-7: 6 new CV algorithms tested one-by-one against pm_v5 baseline.

Base (pm_v5 winner): hybrid(hull->edgefit->mar) + admission filter o15_ba.
Each algorithm adds exactly ONE new dimension:

  7a ws     : watershed splitting for touching/overlapping boxes (pre-seg)
  7b lab    : adaptive LAB a*-channel mask (different mask generation)
  7c hough  : Hough-lines on Canny edges as 4th union member (new proposal)
  7d skel   : skeleton + endpoint corner detection (new proposal)
  7e subpix : cv2.cornerSubPix gradient refinement (post-proc)
  7f tree   : RETR_TREE hole-based box splitting (pre-seg)

Usage:
  python proto/cv_scope7.py            # run all 6
  python proto/cv_scope7.py --only ws  # run one
"""
import sys, os, json, argparse, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, iou_obb, load_expected
from canny_quad import get_ids, to_json, DEFAULT as CQ_DEFAULT, denoise, _quad_adaptive, _side_ok, _order_quad
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                       FIT_DEFAULT, evaluate, ROOT, DIST_VALUES,
                       _edgefit_corners, _sector_corners)
from cv_scope4 import _base_quads, HY_ORDER
from eval_pointmatch import load_boxes
from quad_ortho import is_orthogonal_quad

PM5 = {"ortho_tol": 15.0, "boundary_band": 12.0, "max_aspect": 12.0}
FROZEN_PM5 = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")


def pm5_params():
    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    p.update(PM5)
    return p


def build_hybrid(quads_per_method, ids, p):
    """Union hull->edgefit->mar with keep-order dedup."""
    return {iid: dedup_keep_order(
        [q for m in HY_ORDER for q in quads_per_method[m].get(iid, [])],
        p["dedup_iou"]) for iid in ids}


def run_base_pipeline(img, p):
    """Standard pm5 pipeline: prep -> outer contours -> base quads -> filter."""
    H, W = img.shape[:2]
    mask = _prep(img, p)
    cnts = outer_contours(mask, p)
    res = {m: [] for m in HY_ORDER}
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                res[m].append(q)
    for m in HY_ORDER:
        res[m] = dedup_keep_order(res[m], p["dedup_iou"])
    return res


# ============ 7a: Watershed splitting ============

def watershed_contours(mask, p):
    """Split touching components via distance-transform watershed."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_disp = (dist / (dist.max() + 1e-9) * 255).astype(np.uint8)
    _, binary = cv2.threshold(dist_disp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, markers = cv2.connectedComponents(binary)
    markers[markers == 0] = 0
    markers[mask == 0] = -1
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    out = []
    for label in range(1, n):
        region = (markers == label).astype(np.uint8)
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) >= p["min_area"]:
                out.append(c)
    return out


def run_watershed(img, p):
    H, W = img.shape[:2]
    mask = _prep(img, p)
    cnts = watershed_contours(mask, p)
    res = {m: [] for m in HY_ORDER}
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                res[m].append(q)
    for m in HY_ORDER:
        res[m] = dedup_keep_order(res[m], p["dedup_iou"])
    return res


# ============ 7b: Adaptive LAB a*-channel mask ============

def lab_red_mask(img, smin=80, vmin=110):
    """HSV red mask OR LAB a*-channel Otsu threshold."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    hi = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    hsv_mask = lo | hi
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1]
    _, a_bin = cv2.threshold(a_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab_mask = a_bin & hsv_mask
    return hsv_mask | lab_mask


def run_lab(img, p):
    H, W = img.shape[:2]
    mask = lab_red_mask(img, p.get("smin", 80), p.get("vmin", 110))
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is not None:
        hier = hier[0]
        for i, c in enumerate(cnts):
            if hier[i][3] != -1:
                continue
            if cv2.contourArea(c) >= p["min_area"]:
                out.append(c)
    res = {m: [] for m in HY_ORDER}
    for c in out:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                res[m].append(q)
    for m in HY_ORDER:
        res[m] = dedup_keep_order(res[m], p["dedup_iou"])
    return res


# ============ 7c: Hough-lines on Canny edges ============

def hough_quads(img, p):
    """Detect quads by bridging red-mask gaps with Hough lines, then
    running the standard contour pipeline on the enhanced mask."""
    H, W = img.shape[:2]
    mask = _prep(img, p)
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=int(p["min_side"]), maxLineGap=20)
    if lines is None:
        return []
    bridge = np.zeros(mask.shape, np.uint8)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(bridge, (x1, y1), (x2, y2), 255, 1)
    bridge = cv2.dilate(bridge, np.ones((3, 3), np.uint8))
    enhanced = cv2.bitwise_or(mask, bridge & cv2.dilate(mask, np.ones((5, 5), np.uint8)))
    cnts = outer_contours(enhanced, p)
    quads = []
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                quads.append(q)
    return dedup_keep_order(quads, p["dedup_iou"])


def run_hough(img, p):
    """pm5 pipeline + Hough as 4th union member."""
    res = run_base_pipeline(img, p)
    hq = hough_quads(img, p)
    res["hough"] = hq
    return res


# ============ 7d: Skeleton + endpoint corners ============

def zhang_suen_thinning(mask):
    """Vectorized Zhang-Suen thinning to 1px skeleton."""
    skel = (mask > 0).astype(np.uint8)
    H, W = skel.shape
    for _ in range(100):
        changed = False
        for step in (1, 2):
            p2 = np.zeros_like(skel); p3 = np.zeros_like(skel)
            p4 = np.zeros_like(skel); p5 = np.zeros_like(skel)
            p6 = np.zeros_like(skel); p7 = np.zeros_like(skel)
            p8 = np.zeros_like(skel); p9 = np.zeros_like(skel)
            p2[1:, :] = skel[:-1, :]; p4[:, 1:] = skel[:, :-1]
            p6[:-1, :] = skel[1:, :]; p8[:, :-1] = skel[:, 1:]
            p3[1:, 1:] = skel[:-1, :-1]; p5[:-1, 1:] = skel[1:, :-1]
            p7[:-1, :-1] = skel[1:, 1:]; p9[1:, :-1] = skel[:-1, 1:]
            n = [p2.astype(bool), p3.astype(bool), p4.astype(bool),
                 p5.astype(bool), p6.astype(bool), p7.astype(bool),
                 p8.astype(bool), p9.astype(bool)]
            sn = sum(n)
            cn = sum((n[i] & ~n[(i + 1) % 8]) for i in range(8))
            cond = (skel > 0) & (sn >= 2) & (sn <= 6) & (cn == 1)
            if step == 1:
                cond &= ~(n[0] & n[2] & n[4]) & ~(n[2] & n[4] & n[6])
            else:
                cond &= ~(n[0] & n[2] & n[6]) & ~(n[0] & n[4] & n[6])
            if cond.any():
                skel[cond] = 0
                changed = True
        if not changed:
            break
    return skel * 255


def skeleton_quads(img, p):
    """Detect quads from skeleton endpoints."""
    H, W = img.shape[:2]
    mask = _prep(img, p)
    skel = zhang_suen_thinning(mask)
    skel_bin = (skel > 0).astype(np.uint8)
    n2 = np.zeros_like(skel_bin); n4 = np.zeros_like(skel_bin)
    n6 = np.zeros_like(skel_bin); n8 = np.zeros_like(skel_bin)
    n2[1:, :] = skel_bin[:-1, :]; n4[:, 1:] = skel_bin[:, :-1]
    n6[:-1, :] = skel_bin[1:, :]; n8[:, :-1] = skel_bin[:, 1:]
    n3 = np.zeros_like(skel_bin); n5 = np.zeros_like(skel_bin)
    n7 = np.zeros_like(skel_bin); n9 = np.zeros_like(skel_bin)
    n3[1:, 1:] = skel_bin[:-1, :-1]; n5[:-1, 1:] = skel_bin[1:, :-1]
    n7[:-1, :-1] = skel_bin[1:, 1:]; n9[1:, :-1] = skel_bin[:-1, 1:]
    nb_count = n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9
    ep_mask = (skel_bin > 0) & (nb_count == 1)
    ys, xs = np.where(ep_mask)
    endpoints = list(zip(xs, ys))
    if len(endpoints) < 4:
        return []
    pts = np.array(endpoints, float)
    quads = []
    used = set()
    for i in range(len(endpoints)):
        if i in used:
            continue
        dists = np.linalg.norm(pts - pts[i], axis=1)
        order = np.argsort(dists)[1:]
        group = [i]
        for j in order:
            if j in used:
                continue
            if len(group) >= 4:
                break
            group.append(int(j))
        if len(group) == 4:
            q = pts[group]
            q = _pass(q, W, H, p)
            if q is not None:
                quads.append(q)
                used.update(group)
    return dedup_keep_order(quads, p["dedup_iou"])


def run_skeleton(img, p):
    """pm5 pipeline + skeleton as 4th union member."""
    res = run_base_pipeline(img, p)
    sq = skeleton_quads(img, p)
    res["skel"] = sq
    return res


# ============ 7e: cornerSubPix refinement ============

def subpix_refine(quads, img, p):
    """Refine quad corners with cv2.cornerSubPix on grayscale gradient."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for q in quads:
        corners = q.astype(np.float32)
        try:
            refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                        20, 0.01))
            q2 = refined.astype(np.int32)
            H, W = img.shape[:2]
            if np.all(q2[:, 0] >= 0) and np.all(q2[:, 0] < W) and \
               np.all(q2[:, 1] >= 0) and np.all(q2[:, 1] < H):
                if _side_ok(q2, p) and is_orthogonal_quad(q2.astype(np.float32), W, H,
                                                           tol=p["ortho_tol"],
                                                           boundary_band=p["boundary_band"]):
                    out.append(q2)
                else:
                    out.append(q)
            else:
                out.append(q)
        except cv2.error:
            out.append(q)
    return out


def run_subpix(img, p):
    """pm5 pipeline + cornerSubPix refinement on final hybrid."""
    res = run_base_pipeline(img, p)
    for m in HY_ORDER:
        res[m] = subpix_refine(res[m], img, p)
    return res


# ============ 7f: RETR_TREE hole-based splitting ============

def tree_split_contours(mask, p):
    """Use RETR_TREE to split contours with multiple holes."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        if cv2.contourArea(c) < p["min_area"]:
            continue
        children = []
        child = hier[i][2]
        while child != -1:
            if cv2.contourArea(cnts[child]) >= 100:
                children.append(child)
            child = hier[child][0]
        if len(children) >= 2:
            holes = [cnts[ch].reshape(-1, 2).mean(axis=0) for ch in children]
            holes = np.array(holes)
            for ch_idx in range(len(children)):
                mask_region = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(mask_region, cnts, i, 255, -1)
                cv2.drawContours(mask_region, cnts, children[ch_idx], 0, -1)
                sub_cnts, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                for sc in sub_cnts:
                    if cv2.contourArea(sc) >= p["min_area"]:
                        out.append(sc)
        else:
            out.append(c)
    return out


def run_tree(img, p):
    H, W = img.shape[:2]
    mask = _prep(img, p)
    cnts = tree_split_contours(mask, p)
    res = {m: [] for m in HY_ORDER}
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                res[m].append(q)
    for m in HY_ORDER:
        res[m] = dedup_keep_order(res[m], p["dedup_iou"])
    return res


# ============ driver ============

ALGORITHMS = {
    "ws":     ("7a watershed",     run_watershed,  False),
    "lab":    ("7b lab-mask",      run_lab,        False),
    "hough":  ("7c hough-lines",   run_hough,      True),
    "skel":   ("7d skeleton",      run_skeleton,   True),
    "subpix": ("7e cornerSubPix",  run_subpix,     False),
    "tree":   ("7f retr-tree",     run_tree,       False),
}


def run_algorithm(name, runner, has_extra, ids, p):
    """Run one algorithm on all ids, return hybrid det_map."""
    extra_order = HY_ORDER + (["hough"] if name == "hough" else
                              ["skel"] if name == "skel" else [])
    det_map = {}
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        res = runner(img, p)
        all_quads = []
        for m in extra_order:
            all_quads.extend(res.get(m, []))
        hy = dedup_keep_order(all_quads, p["dedup_iou"])
        det_map[iid] = hy
        total += len(hy)
        out_dir = os.path.join(ROOT, f"tmp_sc7_{name}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in hy]), f,
                      ensure_ascii=False)
        if (n + 1) % 100 == 0:
            print(f"  [{name} {n+1}/{len(ids)}] boxes={total}")
    print(f"  {name}: {total} boxes -> tmp_sc7_{name}")
    return det_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only one algorithm")
    args = ap.parse_args()

    p = pm5_params()
    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"LOOP-7 CV scope on {len(ids)} images, {total_gt} GT boxes, D={DIST_VALUES}")
    print(f"Base = pm_v5 (frozen {FROZEN_PM5})\n")

    base_det = {iid: [q.astype(np.int32) for q in load_boxes(FROZEN_PM5, iid)]
                for iid in ids}
    base_results = {}
    for D in DIST_VALUES:
        tp, fp = evaluate(ids, gt_map, base_det, D)
        base_results[str(D)] = {"tp": tp, "raw_fp": fp}
    base_ndet = sum(len(v) for v in base_det.values())
    print(f"{'base(pm5)':12s} {base_ndet:5d}  " + "  ".join(
        f"TP={base_results[str(d)]['tp']:4d}" for d in DIST_VALUES))

    algos = list(ALGORITHMS.items())
    if args.only:
        algos = [(k, v) for k, v in algos if k == args.only]

    results = {}
    for name, (label, runner, has_extra) in algos:
        print(f"\n--- {label} ---")
        det_map = run_algorithm(name, runner, has_extra, ids, p)
        ndet = sum(len(v) for v in det_map.values())
        row = {}
        for D in DIST_VALUES:
            tp, fp = evaluate(ids, gt_map, det_map, D)
            row[str(D)] = {"tp": tp, "raw_fp": fp}
        results[name] = {"label": label, "det": ndet, "per_D": row}
        print(f"{name:12s} {ndet:5d}  " + "  ".join(
            f"TP={row[str(d)]['tp']:4d}" for d in DIST_VALUES))
        for D in DIST_VALUES:
            d_tp = row[str(D)]["tp"] - base_results[str(D)]["tp"]
            flag = " ***" if abs(d_tp) >= 5 else ""
            print(f"  D={D:2d}: TP {base_results[str(D)]['tp']}->{row[str(D)]['tp']} "
                  f"({d_tp:+d}){flag}")

    print(f"\n{'='*70}")
    print(f"{'Algorithm':12s} {'det':>5s}  " + "  ".join(f"D={d:<10d}" for d in DIST_VALUES))
    print(f"{'base(pm5)':12s} {base_ndet:5d}  " + "  ".join(
        f"TP={base_results[str(d)]['tp']:4d}     " for d in DIST_VALUES))
    for name, (label, _, _) in algos:
        if name in results:
            r = results[name]
            print(f"{name:12s} {r['det']:5d}  " + "  ".join(
                f"TP={r['per_D'][str(d)]['tp']:4d}     " for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_scope7.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(ids), "total_gt": total_gt,
            "dist_values": DIST_VALUES,
            "base": {"det": base_ndet, "per_D": base_results},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()