"""LOOP-44: understand the TOPOLOGY of merged red components.

LOOP-43 showed that 500/1121 GT boxes share a red connected component with
another GT box and that the merged GT boxes are geometrically disjoint
(694/864 pairs have IoU ~ 0).  A component yields exactly one quad, so the
sibling boxes are structurally unreachable.

Before designing a splitter we need to know HOW they are joined:
  corner_touch   the two outlines touch at a point / short segment
  edge_overlap   the two outlines run along each other (shared stroke)
  bridge         a separate red blob (text, object) links the two
and how separable they are:
  * stroke width (distance transform peak)
  * how deep the notch is
  * whether approxPolyDP of the merged contour yields >4 vertices
  * the minimum distance between the two GT perimeters

Usage:
  python proto/scope44_merge_topology.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask
from canny_quad import get_ids, denoise
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, PROD_DEFAULT, outer_contours)
from cv_scope34 import load_boxes_any
from scope43_fn_forensics import perim_points, PROD, RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def poly_reflex_count(poly):
    """Number of reflex (interior angle > 180 deg) vertices of a simple polygon."""
    n = len(poly)
    if n < 3:
        return 0
    c = 0
    area = cv2.contourArea(poly.astype(np.float32), oriented=True)
    for i in range(n):
        p0 = poly[(i - 1) % n]
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        e1 = p0 - p1
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if area < 0:
            cross = -cross
        if cross < 0:
            c += 1
    return c


def poly_min_dist(a, b):
    """Min distance between the two polygon perimeters (approx, sampled)."""
    pa = perim_points(a, step=3)
    pb = perim_points(b, step=3)
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def main():
    ids = get_ids()
    rows = []
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        if len(gt) < 2:
            continue
        p = PROD
        mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
        den = denoise(mask, p["denoise_min_area"]) if p["denoise_min_area"] > 0 else mask
        H, W = den.shape
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(den, 8)
        if ncomp <= 1:
            continue

        # map every perimeter sample to its component
        comp_of_gt = []
        for g in gt:
            pts = perim_points(g, step=2)
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            vals = lab[ys, xs]
            vals = vals[vals > 0]
            if not len(vals):
                comp_of_gt.append((0, 0.0))
                continue
            cnt = collections.Counter(vals.tolist())
            comp, k = cnt.most_common(1)[0]
            comp_of_gt.append((comp, k / len(pts)))

        # groups of GT boxes per component
        groups = collections.defaultdict(list)
        for gi, (comp, cov) in enumerate(comp_of_gt):
            if cov >= 0.5:
                groups[comp].append(gi)

        for comp, gis in groups.items():
            if len(gis) < 2:
                continue
            x, y, w, h, area = stats[comp]
            cmask = (lab == comp).astype(np.uint8)
            dist = cv2.distanceTransform(cmask, cv2.DIST_L2, 3)
            stroke_w = float(dist.max()) * 2.0
            cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            approx_counts = {}
            for eps in (2, 3, 5, 8):
                approx_counts[eps] = len(cv2.approxPolyDP(c, eps, True))
            # reflex vertices of the simplified contour
            ap = cv2.approxPolyDP(c, 3, True).reshape(-1, 2)
            reflex = poly_reflex_count(ap)
            # pairwise GT geometry
            pairs = []
            for i in range(len(gis)):
                for j in range(i + 1, len(gis)):
                    a, b = gt[gis[i]], gt[gis[j]]
                    pairs.append({
                        "a": gis[i], "b": gis[j],
                        "min_perim_dist": round(poly_min_dist(a, b), 2),
                        "area_a": round(cv2.contourArea(a.astype(np.float32)), 1),
                        "area_b": round(cv2.contourArea(b.astype(np.float32)), 1),
                    })
            rows.append({
                "id": iid, "comp": int(comp), "n_gt": len(gis),
                "gt_idx": gis, "comp_area": int(area),
                "comp_w": int(w), "comp_h": int(h),
                "stroke_w": round(stroke_w, 2),
                "approx_verts": approx_counts, "reflex": reflex,
                "pairs": pairs,
            })
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] merged comps={len(rows)}")

    # ---------------- aggregate ----------------
    ngt = collections.Counter(r["n_gt"] for r in rows)
    print(f"\nmerged components: {len(rows)}")
    print("  GT boxes per merged component:", dict(sorted(ngt.items())))

    all_pairs = [p for r in rows for p in r["pairs"]]
    d = np.array([p["min_perim_dist"] for p in all_pairs])
    print(f"\npair perimeter distance: n={len(d)}  "
          f"min={d.min():.1f} p10={np.percentile(d,10):.1f} "
          f"med={np.median(d):.1f} p90={np.percentile(d,90):.1f}")
    for thr in (1, 2, 3, 5, 8, 12, 20):
        print(f"    <= {thr:3d} px : {int((d <= thr).sum()):4d} "
              f"({(d <= thr).mean()*100:.1f}%)")

    print("\napproxPolyDP vertex count of the merged contour:")
    for eps in (2, 3, 5, 8):
        cnt = collections.Counter(r["approx_verts"][eps] for r in rows)
        gt4 = sum(v for k, v in cnt.items() if k == 4)
        gt5 = sum(v for k, v in cnt.items() if k > 4)
        print(f"  eps={eps}: ==4 -> {gt4} ({gt4*100.0/len(rows):.1f}%)   "
              f">4 -> {gt5} ({gt5*100.0/len(rows):.1f}%)   top: {cnt.most_common(6)}")

    rf = collections.Counter(r["reflex"] for r in rows)
    print("\nreflex vertices (eps=3):", dict(sorted(rf.items())))
    print("  rows with >=1 reflex:",
          sum(v for k, v in rf.items() if k >= 1), "/", len(rows))

    sw = np.array([r["stroke_w"] for r in rows])
    print(f"\nstroke width (2*max distance transform): min={sw.min():.1f} "
          f"med={np.median(sw):.1f} p90={np.percentile(sw,90):.1f} "
          f"max={sw.max():.1f}")

    out = os.path.join(ROOT, "benchmark", "scope44_merge_topology.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_merged_components": len(rows), "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()