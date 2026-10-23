"""LOOP-53: does the LINE detector see the boxes the HOLE detector misses?

For every false negative of the hole pipeline, report the closest line
candidate (and its support) - that separates "the strokes are not found" from
"the rectangle is found but inaccurate".
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise, get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
import scope51_linerect as lr
from scope45_holes import P, HOLE_DEFAULT
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"
CFG = dict(lr.LR_DEFAULT)
CFG["lr_perim"] = 0.95


def candidates(img, cfg):
    mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    mask = denoise(mask, P["denoise_min_area"])
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return mask, np.zeros((0, 4, 2))
    pts = np.stack([xs, ys], 1).astype(float)
    strokes = []
    for ang in lr._angles(pts, cfg):
        strokes += lr._strokes_at(pts, ang, cfg)
    strokes = lr._dedup(strokes, cfg)
    fams = lr._families(strokes, cfg)
    allc = []
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
            allc.append(c[good & inb])
    if not allc:
        return mask, np.zeros((0, 4, 2))
    return mask, np.vstack(allc)


def main():
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    fn_rows = json.load(open(os.path.join(ROOT, "benchmark", "scope47_fn2.json"),
                              encoding="utf-8"))["fn_detail"]
    fn_by_id = collections.defaultdict(list)
    for r in fn_rows:
        fn_by_id[r["id"]].append(r)

    stats = collections.Counter()
    dist_hist = collections.Counter()
    examples = []
    ids = [i for i in get_ids()[:n_img] if i in fn_by_id]
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        mask, cands = candidates(img, CFG)
        if len(cands) == 0:
            for r in fn_by_id[iid]:
                stats["no_cand"] += 1
            continue
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
        sup = lr._perim_support_batch(mask_d, cands, 48)
        inter = lr._interior_red_batch(mask, cands)
        good = (sup >= CFG["lr_perim"]) & (inter <= CFG["lr_interior"])
        cands = cands[good]
        sup = sup[good]
        for r in fn_by_id[iid]:
            g = np.asarray(gt[r["gi"]], np.int32)
            if len(cands) == 0:
                stats["no_cand"] += 1
                continue
            ds = np.array([corner_match_distance(c.astype(np.int32), g) for c in cands])
            k = int(np.argmin(ds))
            d = float(ds[k])
            dist_hist[min(int(d // 5) * 5, 40)] += 1
            if d <= 5:
                stats["found<=5"] += 1
            elif d <= 12:
                stats["found<=12"] += 1
            elif d <= 25:
                stats["found<=25"] += 1
            else:
                stats["miss>25"] += 1
            if len(examples) < 25:
                examples.append({"id": iid, "gi": r["gi"], "reason": r["reason"],
                                 "d": round(d, 1), "sup": round(float(sup[k]), 2),
                                 "inter": round(float(inter[good][k]), 3)
                                 if False else round(float(inter[:len(cands)][k]), 3)})
        print(iid, len(cands), "kept", flush=True)

    print(f"\nFN boxes checked: {sum(stats.values())}")
    for k, v in stats.most_common():
        print(f"  {k:12s} {v:5d} ({v*100.0/max(sum(stats.values()),1):5.1f}%)")
    print("distance histogram (5px buckets):", dict(sorted(dist_hist.items())))
    out = os.path.join(ROOT, "benchmark", "scope53_fn_lin.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": dict(stats), "dist_hist": {str(k): v for k, v in dist_hist.items()},
                   "examples": examples}, f, ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()