"""Render forensic overlays for selected FN cases: GT (red) + the red component
contour (green) + the quad edgefit fitted on it (yellow) + the red mask.

Usage:
  python proto/scope43_viz.py 20260913_102940_782 0 3
  python proto/scope43_viz.py --list filter_ortho 4
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids, denoise, _side_ok
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, PROD_DEFAULT,
                       outer_contours, _edgefit_corners, dedup_keep_order)
from quad_ortho import is_orthogonal_quad
from cv_scope34 import load_boxes_any

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
OUT = os.path.join(ROOT, "tmp_fn_viz")

PROD = dict(CQ_DEFAULT)
PROD.update(FIT_DEFAULT)
PROD.update(MASK_DEFAULT)
PROD.update(PROD_DEFAULT)


def render(iid, gi, out_dir=OUT, zoom=True):
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    H, W = img.shape[:2]
    gt = load_boxes_any(RAW, iid)
    g = gt[gi]

    mask = red_mask(img, PROD["sat_min"], PROD["val_min"], PROD["hue_max"])
    den = denoise(mask, PROD["denoise_min_area"])
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(den, 8)
    p0 = dict(PROD); p0["min_area"] = 0.0
    contours = outer_contours(den, p0)

    vis = img.copy()
    # mask overlay
    ov = vis.copy()
    ov[den > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 0.65, ov, 0.35, 0)

    best = None
    best_cov = 0
    for c in contours:
        comp = int(lab[int(c[0][0][1]), int(c[0][0][0])])
        compmask = (lab == comp).astype(np.uint8) * 255
        cnt = np.count_nonzero(compmask)
        if cnt < 1:
            continue
        # perimeter coverage
        pts = []
        for i in range(4):
            a = g[i].astype(float); b = g[(i + 1) % 4].astype(float)
            L = float(np.linalg.norm(b - a)); n = max(int(L / 2), 2)
            ts = np.linspace(0, 1, n, endpoint=False)
            pts.append(a[None, :] + ts[:, None] * (b - a)[None, :])
        pts = np.concatenate(pts)
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
        cov = np.count_nonzero(compmask[ys, xs]) / len(pts)
        if cov > best_cov:
            best_cov = cov
            best = c

    if best is not None:
        cv2.drawContours(vis, [best], -1, (0, 255, 0), 1)
        q = _edgefit_corners(best, PROD)
        if q is not None:
            cv2.polylines(vis, [np.asarray(q).astype(np.int32)], True, (0, 255, 255), 1)
    cv2.polylines(vis, [g.astype(np.int32)], True, (255, 0, 255), 1)

    if zoom:
        x0, y0 = g[:, 0].min(), g[:, 1].min()
        x1, y1 = g[:, 0].max(), g[:, 1].max()
        pad = max(int((x1 - x0) * 0.4), 20)
        x0 = max(int(x0) - pad, 0); y0 = max(int(y0) - pad, 0)
        x1 = min(int(x1) + pad, W - 1); y1 = min(int(y1) + pad, H - 1)
        crop = vis[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    else:
        crop = vis
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{iid}_{gi}.jpg")
    cv2.imwrite(path, crop)
    return path, best_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iid", nargs="?", default=None)
    ap.add_argument("gi", nargs="?", type=int, default=0)
    ap.add_argument("--list", default=None)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    if args.list:
        with open(os.path.join(ROOT, "benchmark", "scope43_fn_forensics.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        rows = [r for r in d["fn_detail"] if r["reason"] == args.list]
        rows.sort(key=lambda r: -r.get("comp_cov", 0))
        for r in rows[:args.n]:
            p, cov = render(r["id"], r["gi"])
            print(f"{args.list:16s} {r['id']} gt#{r['gi']} cov={cov:.2f} "
                  f"qdist={r.get('q_dist')} dev={r.get('ortho_dev')} -> {p}")
        return

    p, cov = render(args.iid, args.gi)
    print(f"{p}  comp_cov={cov:.2f}")


if __name__ == "__main__":
    main()