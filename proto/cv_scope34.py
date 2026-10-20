"""LOOP-34: Find the best orthogonality-enforcement method.

User chose to ENFORCE orthogonality (rule: every detected target must have
4 orthogonal edges). This compares enforcement methods by TP cost vs the
benchmark GT (Compare-V1@CARS_202609_annotated, 746 boxes, reproduces TP=609).

Methods:
  none        : raw quad
  minarect    : cv2.minAreaRect of the 4 corners
  lsq_center  : center + mean-extent orthogonal reconstruction
  ortho_edge  : keep edge midpoints, average edge directions, make the two
                families exactly perpendicular, re-intersect -> minimal corner move
  ortho_blend : blend(original, ortho_edge, 0.5)

Usage: python proto/cv_scope34.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, red_mask, load_expected
from canny_quad import get_ids
from cv_hybrid import ROOT
from eval_pointmatch import parse_points, match_boxes_point, is_real_unlabeled

D_EVAL = 5.0
DET_DIR = os.path.join(ROOT, "tmp_sc31_pm_v5_2")
GT_DIR = r"E:\Xray\CARS\data\Compare-V1@CARS_202609_annotated"


def load_boxes_any(d, iid):
    p = os.path.join(d, iid + ".json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _line_isect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], float)
    b = np.asarray(p2, float) - np.asarray(p1, float)
    try:
        ts = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return (np.asarray(p1, float) + np.asarray(p2, float)) / 2
    return np.asarray(p1, float) + ts[0] * np.asarray(d1, float)


def minarect(q):
    return np.round(cv2.boxPoints(cv2.minAreaRect(
        np.asarray(q, np.float32).reshape(-1, 1, 2)))).astype(np.int32)


def lsq_center(q):
    q = np.asarray(q, float)
    c = q.mean(axis=0)
    e = [q[(i + 1) % 4] - q[i] for i in range(4)]
    u1 = _unit(e[0] - e[2])          # opposite edges are anti-parallel -> subtract
    u2 = np.array([-u1[1], u1[0]])
    d = q - c
    a = np.abs(d @ u1).mean(); b = np.abs(d @ u2).mean()
    return np.round(np.array([c - a*u1 - b*u2, c + a*u1 - b*u2,
                              c + a*u1 + b*u2, c - a*u1 + b*u2])).astype(np.int32)


def ortho_edge(q):
    q = np.asarray(q, float)
    mids = [(q[i] + q[(i+1) % 4]) / 2 for i in range(4)]
    e = [q[(i+1) % 4] - q[i] for i in range(4)]
    uA = _unit(e[0] - e[2])
    uB = _unit(e[1] - e[3])
    perpA = np.array([-uA[1], uA[0]])
    if np.dot(perpA, uB) < 0:
        perpA = -perpA
    d = [uA, perpA, uA, perpA]
    out = []
    for i in range(4):
        out.append(_line_isect(mids[i], d[i], mids[(i-1) % 4], d[(i-1) % 4]))
    return np.round(np.array(out)).astype(np.int32)


def ortho_blend(q, alpha=0.5):
    q = np.asarray(q, float)
    o = ortho_edge(q).astype(float)
    return np.round((1-alpha) * q + alpha * o).astype(np.int32)


def eval_vs_gt(det_map, ids, dist=5.0):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(GT_DIR, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        tp, unmatched = match_boxes_point(det, gt, dist)
        TP += tp; FN += len(gt) - tp; raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], exp.get(iid, 0), len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
    tot = TP + FN
    rec = TP / tot * 100 if tot else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
            "real_unlabeled": real_unl, "recall": round(rec, 1),
            "corrected_f1": round(f1, 1)}


def main():
    ids = get_ids()
    print(f"LOOP-34: orthogonality enforcement methods vs GT={GT_DIR}")
    print(f"  {len(ids)} images, D={D_EVAL:.0f}  (GT total = "
          f"{sum(len(load_boxes_any(GT_DIR, i)) for i in ids)} boxes)\n")

    methods = {"none": lambda q: np.asarray(q, np.int32),
               "minarect": minarect, "lsq_center": lsq_center,
               "ortho_edge": ortho_edge,
               "ortho_blend0.5": lambda q: ortho_blend(q, 0.5),
               "ortho_blend0.3": lambda q: ortho_blend(q, 0.3)}
    print(f"  {'method':16s} {'TP':>5s} {'rawFP':>6s} {'trueFP':>7s} "
          f"{'realUnl':>8s} {'recall':>7s} {'corr_f1':>8s}")
    results = {}
    for name, fn in methods.items():
        dm = {iid: [fn(q) for q in load_boxes_any(DET_DIR, iid)] for iid in ids}
        m = eval_vs_gt(dm, ids, D_EVAL)
        results[name] = m
        print(f"  {name:16s} {m['tp']:5d} {m['raw_fp']:6d} {m['true_fp']:7d} "
              f"{m['real_unlabeled']:8d} {m['recall']:6.1f}% {m['corrected_f1']:7.1f}%")

    out = os.path.join(ROOT, "benchmark", "cv_scope34.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"gt_dir": GT_DIR,
                   "results": {k: {"tp": v["tp"], "raw_fp": v["raw_fp"],
                                   "true_fp": v["true_fp"],
                                   "real_unlabeled": v["real_unlabeled"],
                                   "recall": v["recall"],
                                   "corrected_f1": v["corrected_f1"]}
                               for k, v in results.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()