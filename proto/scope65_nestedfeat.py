"""LOOP-65: what separates a USEFUL nested line candidate from a useless one?

scope59 showed that the nested line candidates contain the missing boxes but
also add a flood of false positives.  This script labels every kept candidate
(TP-like = within 5px of a GT box) and dumps the features that might separate
them:

  sup        red fraction of the perimeter
  red        red fraction of the interior
  cov        fraction of the interior covered by hole regions
  single     largest fraction covered by a SINGLE hole
  holes      number of hole regions the candidate touches
  rel_area   candidate area / area of the hole it lives in
  perim      perimeter length
  aspect     side ratio
  cont       how much of the candidate's perimeter lies on the HOLE boundary
             (a real box side of a merged hole follows the hole's edge)

Usage: python proto/scope65_nestedfeat.py [--n 40]
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from eval_pointmatch import corner_match_distance
import scope51_linerect as lr
import scope59_nested as sn
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, _holes_of
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--perim", type=float, default=0.90)
    ap.add_argument("--cov", type=float, default=0.60)
    ap.add_argument("--single", type=float, default=0.30)
    ap.add_argument("--inter", type=float, default=0.35)
    args = ap.parse_args()

    ids = get_ids()[:args.n]
    LRC = dict(lr.LR_DEFAULT)
    CFG = dict(HOLE_DEFAULT)
    rows = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(RAW, iid)
        if not gt:
            continue
        mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        mask = denoise(mask, P["denoise_min_area"])
        cands = sn.line_candidates(mask, img, LRC)
        if not len(cands):
            continue
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0
        sup = lr._perim_support_batch(mask_d, cands, 48)
        lab, nh = sn.hole_label_mask(mask, CFG)
        cov, single, red = sn.nested_stats(lab, cands)
        keep = (sup >= args.perim) & (red <= args.inter) & (cov >= args.cov) \
            & (single >= args.single)
        if not keep.any():
            continue
        kc = cands[keep]
        ks, kcov, ksingle, kred = sup[keep], cov[keep], single[keep], red[keep]
        # number of distinct holes touched + hole-relative area
        H, W = lab.shape
        s = (np.arange(5) + 0.5) / 5
        S, T = np.meshgrid(s, s, indexing="ij")
        a = kc[:, 0][:, None, None, :]
        b = kc[:, 1][:, None, None, :]
        c = kc[:, 3][:, None, None, :]
        Pm = a + S[None, :, :, None] * (b - a) + T[None, :, :, None] * (c - a)
        Q = Pm.reshape(len(kc), -1, 2)
        xs = np.clip(np.round(Q[..., 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(Q[..., 1]).astype(int), 0, H - 1)
        v = lab[ys, xs].reshape(len(kc), -1)
        nlab = int(lab.max()) + 1
        M = np.zeros((len(kc), nlab), np.int32)
        rowsidx = np.repeat(np.arange(len(kc)), v.shape[1])
        np.add.at(M, (rowsidx, v.ravel()), 1)
        M[:, 0] = 0
        M[:, 1] = 0
        nholes = (M > 0).sum(axis=1)
        hole_area = M.sum(axis=1) / 25.0
        # candidate area
        def quad_area(q):
            x = q[:, :, 0]
            y = q[:, :, 1]
            return 0.5 * np.abs(np.sum(x * np.roll(y, 1, axis=1) - y * np.roll(x, 1, axis=1), axis=1))
        c_area = quad_area(kc)
        rel_area = c_area / np.maximum(hole_area, 1.0)
        side = np.stack([np.linalg.norm(kc[:, (i + 1) % 4] - kc[:, i], axis=1)
                         for i in range(4)], 1)
        perim = side.sum(1)
        aspect = side.max(1) / np.maximum(side.min(1), 1e-6)
        # how much of the perimeter lies on the hole boundary
        on_hole = 0
        for i in range(4):
            p1 = kc[:, i]
            p2 = kc[:, (i + 1) % 4]
            nn = 24
            ts = (np.arange(nn) + 0.5) / nn
            pts = p1[:, None, :] + ts[None, :, None] * (p2 - p1)[:, None, :]
            xx = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
            yy = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
            vv = lab[yy, xx]
            on_hole += (vv >= 2).sum(axis=1)
        cont = on_hole / (4 * nn)
        for k in range(len(kc)):
            ds = [corner_match_distance(kc[k].astype(np.int32), np.asarray(g, np.int32))
                  for g in gt]
            j = int(np.argmin(ds))
            rows.append({"id": iid, "d": round(float(ds[j]), 1),
                         "tp": bool(ds[j] <= 5.0),
                         "sup": round(float(ks[k]), 3),
                         "red": round(float(kred[k]), 3),
                         "cov": round(float(kcov[k]), 3),
                         "single": round(float(ksingle[k]), 3),
                         "nholes": int(nholes[k]),
                         "rel_area": round(float(rel_area[k]), 3),
                         "perim": round(float(perim[k]), 1),
                         "aspect": round(float(aspect[k]), 2),
                         "cont": round(float(cont[k]), 3),
                         "area": round(float(c_area[k]), 1)})
    tp = [r for r in rows if r["tp"]]
    fp = [r for r in rows if not r["tp"]]
    print(f"kept={len(rows)}  tp_like={len(tp)}  fp={len(fp)}")

    def stat(name, key, buckets):
        print(f"\n-- {name} --")
        ct = collections.Counter()
        cf = collections.Counter()
        for r in tp:
            ct[min(int(r[key] // buckets[1]) * buckets[1], buckets[0])] += 1
        for r in fp:
            cf[min(int(r[key] // buckets[1]) * buckets[1], buckets[0])] += 1
        for k in sorted(set(ct) | set(cf)):
            print(f"   {k:8}: tp={ct[k]:4d}  fp={cf[k]:5d}   ratio={ct[k]/max(cf[k],1):.3f}")

    stat("sup", "sup", (1.0, 0.05))
    stat("red", "red", (0.4, 0.05))
    stat("cov", "cov", (1.0, 0.1))
    stat("single", "single", (1.0, 0.1))
    stat("nholes", "nholes", (5, 1))
    stat("rel_area", "rel_area", (2.0, 0.1))
    stat("cont", "cont", (1.0, 0.1))
    stat("aspect", "aspect", (12, 1))
    out = os.path.join(ROOT, "benchmark", "scope65_nestedfeat.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()