"""LOOP-56: for a few hole-pipeline FNs, list the closest LINE candidates
with their support, to see whether the correct rectangle is in the candidate
set at all (and what stops it from being selected)."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
import scope51_linerect as lr
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _hole_quad
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(lr.LR_DEFAULT)
CFG["lr_perim"] = 0.0
CFG["lr_interior"] = 1.0


def main():
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    fn_rows = json.load(open(os.path.join(ROOT, "benchmark", "scope47_fn2.json"),
                              encoding="utf-8"))["fn_detail"]
    by = collections.defaultdict(list)
    for r in fn_rows:
        by[r["reason"]].append(r)
    ids = get_ids()[:n_img]
    shown = 0
    for iid in ids:
        rows = [r for r in fn_rows if r["id"] == iid]
        if not rows:
            continue
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
        for ang in lr._angles(pts, CFG):
            strokes += lr._strokes_at(pts, ang, CFG)
        strokes = lr._dedup(strokes, CFG)
        fams = lr._families(strokes, CFG)
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
                okA = (sA >= CFG["lr_min_side"]) & (sA <= CFG["lr_max_side"])
                okB = (sB >= CFG["lr_min_side"]) & (sB <= CFG["lr_max_side"])
                if not okA.any() or not okB.any():
                    continue
                AA, BB = np.meshgrid(np.nonzero(okA)[0], np.nonzero(okB)[0], indexing="ij")
                AA = AA.ravel(); BB = BB.ravel()
                sa = sA[AA]; sb = sB[BB]
                asp = np.maximum(sa, sb) / np.maximum(np.minimum(sa, sb), 1e-6)
                k = asp <= CFG["lr_max_aspect"]
                AA, BB = AA[k], BB[k]
                X, okm = lr._isect_matrix(PA, DA, PB, DB)
                ia2 = ia[AA]; ja2 = ja[AA]; ib2 = ib[BB]; jb2 = jb[BB]
                c = np.empty((len(AA), 4, 2))
                c[:, 0] = X[ia2, ib2]; c[:, 1] = X[ia2, jb2]
                c[:, 2] = X[ja2, jb2]; c[:, 3] = X[ja2, ib2]
                good = okm[ia2, ib2] & okm[ia2, jb2] & okm[ja2, jb2] & okm[ja2, ib2]
                inb = ((c[:, :, 0].min(1) > -5) & (c[:, :, 0].max(1) < img.shape[1] + 5)
                       & (c[:, :, 1].min(1) > -5) & (c[:, :, 1].max(1) < img.shape[0] + 5))
                cands.append(c[good & inb])
        if not cands:
            continue
        cands = np.vstack(cands)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
        sup = lr._perim_support_batch(mask_d, cands, 48)
        inter = lr._interior_red_batch(mask, cands)
        holes = _holes_of(mask, dict(HOLE_DEFAULT))
        for r in rows:
            g = np.asarray(gt[r["gi"]], np.int32)
            ds = np.array([corner_match_distance(c.astype(np.int32), g) for c in cands])
            order = np.argsort(ds)[:4]
            hd = []
            for h in holes:
                q = _hole_quad(h, mask, dict(HOLE_DEFAULT))
                if q is None:
                    continue
                hd.append((corner_match_distance(q.astype(np.int32), g), float(sup[0])))
            hs = min(hd)[0] if hd else -1
            print(f"\n{iid} gt#{r['gi']} {r['reason']} best_hole_d={hs:.1f}")
            for k in order:
                print(f"   line cand d={ds[k]:6.1f} sup={sup[k]:.3f} inter={inter[k]:.3f} "
                      f"q={np.round(cands[k],1).astype(int).tolist()}")
            print(f"   GT {g.tolist()}")
            shown += 1
            if shown > 14:
                return


if __name__ == "__main__":
    main()