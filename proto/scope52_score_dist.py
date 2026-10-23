"""LOOP-52: score distribution of the line-rectangle candidates.

For a sample of images, build every candidate rectangle and label it TP
(corner match within D of a GT box) or FP.  Then report the distribution of
the scoring features so the thresholds can be set from data.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
import scope51_linerect as lr
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
P = {"sat_min": 100, "val_min": 140, "hue_max": 4, "denoise_min_area": 40}


def main():
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cfg = dict(lr.LR_DEFAULT)
    cfg["lr_perim"] = 0.0
    cfg["lr_interior"] = 1.0
    ids = get_ids()[:n_img]
    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        strokes = []
        for ang in lr._angles(pts, cfg):
            strokes += lr._strokes_at(pts, ang, cfg)
        strokes = lr._dedup(strokes, cfg)
        fams = lr._families(strokes, cfg)
        # rebuild all candidates without scoring
        cands = []
        for fi, fa in enumerate(fams):
            for fb in fams[fi + 1:]:
                diff = min(abs(fa["ang"] - fb["ang"]) % 180.0,
                           180 - abs(fa["ang"] - fb["ang"]) % 180.0)
                if abs(diff - 90.0) > 8.0:
                    continue
                PA = np.array([L["point"] for L in fa["lines"]])
                DA = np.array([L["dir"] for L in fa["lines"]])
                PB = np.array([L["point"] for L in fb["lines"]])
                DB = np.array([L["dir"] for L in fb["lines"]])
                oA = PA @ lr._perp(DA[0]); oB = PB @ lr._perp(DB[0])
                ia, ja = np.triu_indices(len(PA), 1)
                ib, jb = np.triu_indices(len(PB), 1)
                sA = np.abs(oA[ja] - oA[ia]); sB = np.abs(oB[jb] - oB[ib])
                okA = (sA >= cfg["lr_min_side"]) & (sA <= cfg["lr_max_side"])
                okB = (sB >= cfg["lr_min_side"]) & (sB <= cfg["lr_max_side"])
                if not okA.any() or not okB.any():
                    continue
                AA, BB = np.meshgrid(np.nonzero(okA)[0], np.nonzero(okB)[0], indexing="ij")
                AA = AA.ravel(); BB = BB.ravel()
                sa = sA[AA]; sb = sB[BB]
                asp = np.maximum(sa, sb) / np.maximum(np.minimum(sa, sb), 1e-6)
                k = asp <= cfg["lr_max_aspect"]
                AA, BB = AA[k], BB[k]
                X, okm = lr._isect_matrix(PA, DA, PB, DB)
                ia2 = ia[AA]; ja2 = ja[AA]; ib2 = ib[BB]; jb2 = jb[BB]
                c = np.empty((len(AA), 4, 2))
                c[:, 0] = X[ia2, ib2]; c[:, 1] = X[ia2, jb2]
                c[:, 2] = X[ja2, jb2]; c[:, 3] = X[ja2, ib2]
                good = okm[ia2, ib2] & okm[ia2, jb2] & okm[ja2, jb2] & okm[ja2, ib2]
                inb = ((c[:, :, 0].min(1) > -5) & (c[:, :, 0].max(1) < img.shape[1] + 5)
                       & (c[:, :, 1].min(1) > -5) & (c[:, :, 1].max(1) < img.shape[0] + 5))
                c = c[good & inb]
                cands.append(c)
        if not cands:
            continue
        cands = np.vstack(cands)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
        sup = lr._perim_support_batch(mask_d, cands, 48)
        inter = lr._interior_red_batch(mask, cands)
        for q, s, it in zip(cands, sup, inter):
            best = min((corner_match_distance(q.astype(np.int32), g) for g in gt),
                       default=1e9)
            rows.append({"id": iid, "sup": round(float(s), 3),
                         "inter": round(float(it), 3), "d": round(float(best), 1),
                         "tp": bool(best <= 5)})
        print(iid, len(cands), "cands", flush=True)

    tp = [r for r in rows if r["tp"]]
    fp = [r for r in rows if not r["tp"]]
    print(f"\ncandidates: {len(rows)}  TP-like: {len(tp)}  FP: {len(fp)}")
    for name, key in (("support", "sup"), ("interior", "inter")):
        for label, grp in (("TP", tp), ("FP", fp)):
            v = np.array([r[key] for r in grp]) if grp else np.array([0.0])
            print(f"  {name:9s} {label}: p10={np.percentile(v,10):.3f} "
                  f"p50={np.percentile(v,50):.3f} p90={np.percentile(v,90):.3f} "
                  f"max={v.max():.3f}")
    out = os.path.join(ROOT, "benchmark", "scope52_scores.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f)
    print("saved", out)


if __name__ == "__main__":
    main()