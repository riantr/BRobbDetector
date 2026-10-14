"""4-axis orthogonal CV scope — adds exactly ONE new axis on top of pm_v4.

Baseline (pm_v4 winner): hybrid(hull -> edgefit -> mar) = hy_chm.
Axes A (mask=raw), B (proposal=outer CC), C (corners=hull/edgefit/mar) are all
FROZEN at the pm_v4 config. The one new axis D varies ONLY the post-extractor
corner-refinement stage:

  D = refine : none / lines / snap
    none  : no refinement  (loaded from frozen pm_v4 det dirs -> exact anchor)
    lines : TLS line-fit along each chosen quad's own edges, corner = intersect
            of adjacent fitted lines (generalized edgefit; refines hull/mar too)
    snap  : snap each corner to the nearest outer-contour point within snap_rad

Pipeline for lines/snap:  extract -> refine(c, q) -> _pass(ortho/side) -> dedup.
Refinement NEVER drops a quad: on any fit failure it returns the input quad.

Hybrid re-built per option: hy = hull -> edgefit -> mar (dedup IoU<0.30).

Usage:
  python proto/cv_scope4.py
  python proto/cv_scope4.py --viz <id>
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, load_expected
from canny_quad import (_quad_adaptive, _order_quad, get_ids, to_json,
                        DEFAULT as CQ_DEFAULT)
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                       _edgefit_corners, _fit_line_tls, _line_isect,
                       FIT_DEFAULT, evaluate, viz as cv_viz, ROOT, DIST_VALUES)
from eval_pointmatch import load_boxes
from cv_scope3 import SC3_DEFAULT

SC4_DEFAULT = {
    "snap_rad": 5.0,
}

NONE_REUSE = {  # pm_v4 frozen per-method det dirs (identical to hy_chm members)
    "hull":    r"D:\src\CodeArts\BRobbDetector\tmp_sc3_h_raw",
    "edgefit": r"D:\src\CodeArts\BRobbDetector\tmp_cv_edgefit",
    "mar":     r"D:\src\CodeArts\BRobbDetector\tmp_cv_mar",
}
FROZEN_HY = r"D:\src\CodeArts\BRobbDetector\tmp_sc3_hy_chm"  # pm_v4 hybrid (cross-check)
HY_ORDER = ["hull", "edgefit", "mar"]


# ---------------- axis D: corner refinement ----------------

def refine_lines(c, q, p):
    """TLS line-fit along q's 4 edges; return corners at adjacent-line isections.

    Never drops q: returns q unchanged on any fit failure."""
    o = _order_quad(np.asarray(q, float))
    pts = c.reshape(-1, 2).astype(float)
    lines = []
    for i in range(4):
        a, b = o[i], o[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 2:
            return q
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = pts - a
        along = rel @ u
        perp = rel @ n
        sel = ((along > -0.10 * L) & (along < 1.10 * L) &
               (np.abs(perp) < p["fit_band"]))
        if int(sel.sum()) < p["fit_min_pts"]:
            return q
        lines.append(_fit_line_tls(pts[sel]))
    refined = []
    for i in range(4):
        x = _line_isect(*lines[i - 1], *lines[i])
        if x is None or np.linalg.norm(x - o[i]) > p["fit_jump"]:
            return q
        refined.append(x)
    return np.array(refined, float)


def snap_to_pts(pts, q, p):
    """Snap each corner of q to the nearest point in pts within snap_rad."""
    out = np.asarray(q, float).copy()
    r = p["snap_rad"]
    for i in range(4):
        d = np.linalg.norm(pts - out[i], axis=1)
        j = int(np.argmin(d))
        if d[j] <= r:
            out[i] = pts[j]
    return out


def refine_snap(c, q, p):
    """Snap each corner of q to the nearest outer-contour point within snap_rad."""
    return snap_to_pts(c.reshape(-1, 2).astype(float), q, p)


REFINE = {"none": None, "lines": refine_lines, "snap": refine_snap}

# post-dedup application of snap on the FINAL hybrid output (radius variants)
POST = {"post5": 5.0, "post3": 3.0}


def build_post(hy_none, ids, p, rad):
    """Snap final hybrid quads (post-dedup) to nearest contour point."""
    out = {}
    for iid in ids:
        quads = hy_none[iid]
        if not quads:
            out[iid] = []
            continue
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            out[iid] = list(quads)
            continue
        cnts = outer_contours(_prep(img, p), p)
        if not cnts:
            out[iid] = list(quads)
            continue
        pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(float)
        pp = dict(p)
        pp["snap_rad"] = rad
        out[iid] = [snap_to_pts(pts, q, pp).astype(np.int32) for q in quads]
    return out


# ---------------- detection ----------------

def _base_quads(c, p):
    """Raw (unfiltered) hull / edgefit / mar candidates for one contour."""
    qh = _quad_adaptive(cv2.convexHull(c), p["approx_eps_list"])
    qe = _edgefit_corners(c, p)
    (_, (rw, rh), _) = cv2.minAreaRect(c)
    qm = cv2.boxPoints(cv2.minAreaRect(c)) if rw >= 1 and rh >= 1 else None
    return {"hull": qh, "edgefit": qe, "mar": qm}


def run_variants(ids, p, variants):
    """Single detection pass; returns variant -> method -> {iid: [quads]}."""
    det = {v: {m: {} for m in HY_ORDER} for v in variants if v != "none"}
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            for v in det:
                for m in HY_ORDER:
                    det[v][m][iid] = []
            continue
        H, W = img.shape[:2]
        mask = _prep(img, p)
        cnts = outer_contours(mask, p)
        buckets = {v: {m: [] for m in HY_ORDER} for v in det}
        for c in cnts:
            base = _base_quads(c, p)
            for v, rf in variants.items():
                if rf is None:
                    continue
                for m in HY_ORDER:
                    q = base[m]
                    if q is None:
                        continue
                    q = rf(c, q, p)
                    qq = _pass(q, W, H, p)
                    if qq is not None:
                        buckets[v][m].append(qq)
        for v in det:
            for m in HY_ORDER:
                det[v][m][iid] = dedup_keep_order(buckets[v][m], p["dedup_iou"])
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] refined")
    return det


def build_hybrid(det_v, ids, p):
    hy = {}
    for iid in ids:
        hy[iid] = dedup_keep_order(
            [q for m in HY_ORDER for q in det_v[m].get(iid, [])], p["dedup_iou"])
    return hy


def write_dir(name, hy, ids):
    out_dir = os.path.join(ROOT, f"tmp_sc4_{name}")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for iid in ids:
        quads = hy[iid]
        total += len(quads)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, quads), f, ensure_ascii=False)
    print(f"  {name}: {total} boxes -> {out_dir}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default=None)
    args = ap.parse_args()

    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    p.update(SC3_DEFAULT)
    p.update(SC4_DEFAULT)

    if args.viz:
        cv_viz(args.viz, "edgefit", p)  # helper viz (base extractor)
        return

    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"4-axis CV scope (new axis D: refine) on {len(ids)} images, "
          f"{total_gt} GT boxes, D={DIST_VALUES}")

    # variant per-method det maps
    var_det = {"none": {m: {iid: [q.astype(np.int32) for q in load_boxes(d, iid)]
                            for iid in ids}
                        for m, d in NONE_REUSE.items()}}
    det = run_variants(ids, p, REFINE)
    var_det.update(det)

    # build hybrids per variant
    hy = {v: build_hybrid(var_det[v], ids, p) for v in REFINE}
    for v in ("lines", "snap"):
        write_dir(f"hy_{v}", hy[v], ids)
    write_dir("hy_none", hy["none"], ids)

    frozen = sum(1 for iid in ids for _ in hy["none"][iid])  # == none hybrid
    frozen_ref = sum(1 for iid in ids for _ in load_boxes(FROZEN_HY, iid))
    print(f"  cross-check: none-hybrid={frozen} vs tmp_sc3_hy_chm={frozen_ref}")

    # post-dedup refinement of the frozen hybrid output
    hy_post = {}
    for v, rad in POST.items():
        hy_post[v] = build_post(hy["none"], ids, p, rad)
        write_dir(f"hy_{v}", hy_post[v], ids)

    # evaluate: per-method rows for pre-dedup variants; hybrid-only for post
    results = {}
    for v in REFINE:
        results[v] = {}
        for m in HY_ORDER + ["hybrid"]:
            dm = hy[v] if m == "hybrid" else var_det[v][m]
            ndet = sum(len(x) for x in dm.values())
            row = {}
            for D in DIST_VALUES:
                tp, fp = evaluate(ids, gt_map, dm, D)
                row[str(D)] = {"tp": tp, "raw_fp": fp}
            results[v][m] = {"det": ndet, "per_D": row}
    for v in POST:
        dm = hy_post[v]
        ndet = sum(len(x) for x in dm.values())
        row = {}
        for D in DIST_VALUES:
            tp, fp = evaluate(ids, gt_map, dm, D)
            row[str(D)] = {"tp": tp, "raw_fp": fp}
        results[v] = {"hybrid": {"det": ndet, "per_D": row}}

    print(f"\n{'variant':6s} {'method':9s} {'det':>5s}  " +
          "  ".join(f"D={d:<14d}" for d in DIST_VALUES))
    for v in REFINE:
        for m in HY_ORDER + ["hybrid"]:
            r = results[v][m]
            print(f"{v:6s} {m:9s} {r['det']:5d}  " + "  ".join(
                f"TP={r['per_D'][str(d)]['tp']:4d} FP={r['per_D'][str(d)]['raw_fp']:4d}"
                for d in DIST_VALUES))
    for v in POST:
        r = results[v]["hybrid"]
        print(f"{v:6s} {'hybrid':9s} {r['det']:5d}  " + "  ".join(
            f"TP={r['per_D'][str(d)]['tp']:4d} FP={r['per_D'][str(d)]['raw_fp']:4d}"
            for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_scope4.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(ids), "total_gt": total_gt,
            "dist_values": DIST_VALUES,
            "new_axis": "D: refine in {none,lines,snap,post5,post3}",
            "config": {"snap_rad": p["snap_rad"], "post": dict(POST),
                       "fit": dict(FIT_DEFAULT), "sc3": dict(SC3_DEFAULT)},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()