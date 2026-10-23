"""LOOP-50c: FN crops - photo with red-mask painted, GT corner markers only.

The GT polyline is NOT drawn: it would cover the very red pixels we want to
inspect.  GT corners are marked with green circles instead.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _hole_quad
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(HOLE_DEFAULT)


def main():
    per_reason = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    only = sys.argv[2] if len(sys.argv) > 2 else None
    zoom = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    out_dir = os.path.join(ROOT, "tmp_fn50")
    os.makedirs(out_dir, exist_ok=True)
    rows = json.load(open(os.path.join(ROOT, "benchmark", "scope47_fn2.json"),
                          encoding="utf-8"))["fn_detail"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["reason"]].append(r)

    gt_map = {}
    for reason, items in by.items():
        if only and reason != only:
            continue
        for k, r in enumerate(items[:per_reason]):
            iid, gi = r["id"], r["gi"]
            if iid not in gt_map:
                gt_map[iid] = load_boxes_any(RAW, iid)
            gt = gt_map[iid]
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
            mask = denoise(mask, P["denoise_min_area"])
            holes = _holes_of(mask, CFG)
            g = np.asarray(gt[gi], np.int32)
            x0, y0 = g.min(axis=0)
            x1, y1 = g.max(axis=0)
            pad = 25
            X0 = max(int(x0) - pad, 0); Y0 = max(int(y0) - pad, 0)
            X1 = min(int(x1) + pad, img.shape[1]); Y1 = min(int(y1) + pad, img.shape[0])
            crop = img[Y0:Y1, X0:X1].copy()
            m = mask[Y0:Y1, X0:X1] > 0
            vis = crop.copy()
            vis[m] = (0, 0, 255)
            for h in holes:
                cv2.drawContours(vis, [h - np.array([X0, Y0])], -1, (0, 255, 255), 1)
            for c in g:
                cv2.circle(vis, (int(c[0]) - X0, int(c[1]) - Y0), 3, (0, 255, 0), -1)
            for h in holes:
                q = _hole_quad(h, mask, CFG)
                if q is not None:
                    qq = q.astype(np.int32) - np.array([X0, Y0])
                    for c in qq:
                        cv2.circle(vis, (int(c[0]), int(c[1])), 2, (255, 200, 0), -1)
            mb = cv2.cvtColor((m.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
            for c in g:
                cv2.circle(mb, (int(c[0]) - X0, int(c[1]) - Y0), 3, (0, 255, 0), -1)
            both = np.vstack([vis, mb])
            both = cv2.resize(both, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            fn = os.path.join(out_dir, f"c{iid}_{gi}_{reason}_{k}.png")
            cv2.imwrite(fn, both)
            print(f"{fn}  best={r.get('best_ov')}")


if __name__ == "__main__":
    main()