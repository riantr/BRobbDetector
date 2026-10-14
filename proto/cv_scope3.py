"""3-axis orthogonal CV scope (classical CV only).

Axes (hold two at current best, vary one):
  A mask signal : raw / close5 / close9 / dil1
  B proposal    : outer CC / hole ring (largest child contour) / frag-join hull
  C corners     : hull / contour / mar / edgefit   (mar, edgefit, contour reused
                  from pm_v3 detection dirs)

Hybrids are priority-dedup unions (first method wins overlaps, IoU<0.30):
  hy_chm     = h_raw -> edgefit -> mar            (hull-primary version of v3 winner)
  hy_v4      = h_raw -> h_close5 -> h_dil1 -> h_hole -> h_frag -> edgefit -> mar
  hy_v4_lean = h_raw -> h_hole -> h_frag -> mar

Usage:
  python proto/cv_scope3.py
  python proto/cv_scope3.py --viz <id> --method h_hole
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, iou_obb, load_expected
from canny_quad import denoise, _quad_adaptive, get_ids, to_json, DEFAULT as CQ_DEFAULT
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _sector_corners,
                       _edgefit_corners, FIT_DEFAULT, evaluate, viz as cv_viz,
                       ROOT, DIST_VALUES)
from eval_pointmatch import load_boxes

SC3_DEFAULT = {
    "hole_min_area": 400.0,
    "frag_min_area": 10.0,
    "frag_dil_iter": 3,
    "frag_join_area": 150.0,
}

SCOPE = {
    "h_raw":    ("raw",    "outer", "hull"),
    "h_close5": ("close5", "outer", "hull"),
    "h_close9": ("close9", "outer", "hull"),
    "h_dil1":   ("dil1",   "outer", "hull"),
    "h_hole":   ("raw",    "hole",  "hull"),
    "h_frag":   ("raw",    "frag",  "hull"),
}
REUSE = {
    "hull": r"D:\src\CodeArts\BRobbDetector\tmp_cv_hull",
    "contour": r"D:\src\CodeArts\BRobbDetector\tmp_canny_out",
    "mar": r"D:\src\CodeArts\BRobbDetector\tmp_cv_mar",
    "edgefit": r"D:\src\CodeArts\BRobbDetector\tmp_cv_edgefit",
}
HYBRIDS = {
    "hy_chm":     ["h_raw", "edgefit", "mar"],
    "hy_v4":      ["h_raw", "h_close5", "h_dil1", "h_hole", "h_frag", "edgefit", "mar"],
    "hy_v4_lean": ["h_raw", "h_hole", "h_frag", "mar"],
}


# ---------------- axis A: mask variants ----------------

def mask_raw(img, p):
    m = red_mask(img)
    if p["denoise_min_area"] > 0:
        m = denoise(m, p["denoise_min_area"])
    return m


def mask_close(img, p, k):
    m = mask_raw(img, p)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)


def mask_dil(img, p, it):
    m = mask_raw(img, p)
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=it)


MASKS = {
    "raw": mask_raw,
    "close5": lambda img, p: mask_close(img, p, 5),
    "close9": lambda img, p: mask_close(img, p, 9),
    "dil1": lambda img, p: mask_dil(img, p, 1),
}


# ---------------- axis B: proposals ----------------

def prop_outer(img, p):
    return outer_contours(MASKS["raw"](img, p), p)


def prop_hole(img, p):
    mask = MASKS["raw"](img, p)
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
        child, best, best_a = hier[i][2], None, 0.0
        while child != -1:
            a = cv2.contourArea(cnts[child])
            if a > best_a:
                best_a, best = a, cnts[child]
            child = hier[child][0]
        if best is not None and best_a >= p["hole_min_area"]:
            out.append(best)
    return out


def prop_frag(img, p):
    m = red_mask(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    id2area = {}
    for i in range(1, n):
        a = int(stats[i, 4])
        if p["frag_min_area"] <= a < p["min_area"]:
            id2area[i] = a
    if not id2area:
        return []
    small = np.isin(lab, list(id2area)).astype(np.uint8) * 255
    dil = cv2.dilate(small, np.ones((5, 5), np.uint8),
                     iterations=int(p["frag_dil_iter"]))
    n2, lab2 = cv2.connectedComponents(dil, 8)
    out = []
    for j in range(1, n2):
        members = [i for i in np.unique(lab[lab2 == j]).tolist()
                   if i in id2area]
        if len(members) < 2:
            continue
        tot = sum(id2area[i] for i in members)
        if tot < p["frag_join_area"]:
            continue
        pts = []
        for i in members:
            ys, xs = np.where(lab == i)
            pts.append(np.column_stack([xs, ys]).astype(np.int32))
        h = cv2.convexHull(np.vstack(pts))
        if cv2.contourArea(h) >= p["min_area"]:
            out.append(h.reshape(-1, 1, 2))
    return out


PROPS = {"outer": prop_outer, "hole": prop_hole, "frag": prop_frag}


# ---------------- axis C: corners (parameterized) ----------------

def quads_from_contours(cnts, corner, p, W, H):
    out = []
    for c in cnts:
        if corner == "hull":
            q = _quad_adaptive(cv2.convexHull(c), p["approx_eps_list"])
        elif corner == "contour":
            q = _quad_adaptive(c, p["approx_eps_list"])
        elif corner == "mar":
            (_, (rw, rh), _) = cv2.minAreaRect(c)
            q = cv2.boxPoints(cv2.minAreaRect(c)) if rw >= 1 and rh >= 1 else None
        elif corner == "edgefit":
            q = _edgefit_corners(c, p)
        else:
            q = None
        qq = _pass(q, W, H, p)
        if qq is not None:
            out.append(qq)
    return dedup_keep_order(out, p["dedup_iou"])


def detect_spec(img, p, spec):
    mn, pn, cn = spec
    cnts = PROPS[pn](img, p)
    return quads_from_contours(cnts, cn, p, img.shape[1], img.shape[0])


def run_scope(ids, p):
    det_maps = {}
    for name, spec in SCOPE.items():
        out_dir = os.path.join(ROOT, f"tmp_sc3_{name}")
        os.makedirs(out_dir, exist_ok=True)
        det_map = {}
        total = 0
        for n, iid in enumerate(ids):
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                det_map[iid] = []
                continue
            quads = detect_spec(img, p, spec)
            det_map[iid] = quads
            total += len(quads)
            with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, quads), f, ensure_ascii=False)
            if (n + 1) % 100 == 0:
                print(f"  [{name} {n+1}/{len(ids)}] boxes={total}")
        print(f"  {name}: {total} boxes -> {out_dir}")
        det_maps[name] = det_map
    for name, d in REUSE.items():
        det_maps[name] = {iid: [q.astype(np.int32) for q in load_boxes(d, iid)]
                          for iid in ids}
    return det_maps


def build_hybrids(ids, det_maps, p):
    hy = {}
    for hname, order in HYBRIDS.items():
        hy[hname] = {
            iid: dedup_keep_order(
                [q for m in order for q in det_maps[m].get(iid, [])],
                p["dedup_iou"])
            for iid in ids
        }
        out_dir = os.path.join(ROOT, f"tmp_sc3_{hname}")
        os.makedirs(out_dir, exist_ok=True)
        total = 0
        for iid in ids:
            quads = hy[hname][iid]
            total += len(quads)
            with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, quads), f, ensure_ascii=False)
        print(f"  {hname}: {total} boxes -> {out_dir}")
    return hy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default=None)
    ap.add_argument("--method", default="h_hole",
                    choices=list(SCOPE) + list(HYBRIDS) + list(REUSE))
    args = ap.parse_args()

    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    p.update(SC3_DEFAULT)

    if args.viz:
        cv_viz(args.viz, "contour", p)  # fallback simple viz
        return

    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"3-axis CV scope on {len(ids)} images, {total_gt} GT boxes, D={DIST_VALUES}")

    det_maps = run_scope(ids, p)
    hy_maps = build_hybrids(ids, det_maps, p)

    results = {}
    names = list(SCOPE) + list(REUSE) + list(HYBRIDS)
    print(f"\n{'method':11s} {'det':>5s}  " +
          "  ".join(f"D={d:<15d}" for d in DIST_VALUES))
    for name in names:
        dm = hy_maps.get(name) or det_maps.get(name)
        ndet = sum(len(v) for v in dm.values())
        row = {}
        for D in DIST_VALUES:
            tp, fp = evaluate(ids, gt_map, dm, D)
            row[str(D)] = {"tp": tp, "raw_fp": fp}
        results[name] = {"det": ndet, "per_D": row}
        print(f"{name:11s} {ndet:5d}  " + "  ".join(
            f"TP={row[str(d)]['tp']:4d} FP={row[str(d)]['raw_fp']:4d}"
            for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_scope3.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(ids), "total_gt": total_gt,
            "dist_values": DIST_VALUES, "config": dict(SC3_DEFAULT),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()