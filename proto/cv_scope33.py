"""LOOP-33: Orthogonality enforcement.

FINDING: the pipeline only GATES on orthogonality (_pass -> is_orthogonal_quad),
it never CORRECTS the quad. Frozen pm_v5_2 output has median 2.0deg / p90 14.9deg
deviation from orthogonal.

This tests whether projecting each output quad to the nearest orthogonal
rectangle improves point-match vs the TRUE GT (Label@CARS_202609_raw, utf-8-sig).

Variants:
  none     : raw quad (hull/edgefit/mar) as-is
  minarect : cv2.minAreaRect of the quad's 4 corners -> boxPoints
  lsq      : least-squares orthogonal rectangle fit

Usage: python proto/cv_scope33.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, red_mask, load_expected
from canny_quad import get_ids
from cv_hybrid import ROOT
from quad_ortho import interior_angles
from eval_pointmatch import match_boxes_point, is_real_unlabeled

D_EVAL = 5.0
DET_DIR = os.path.join(ROOT, "tmp_sc31_pm_v5_2")
GT_DIR = r"E:\Xray\CARS\data\Compare-V1@CARS_202609_annotated"


def load_boxes_sig(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    out = []
    for info in data.get("infos", []):
        pts = [tuple(int(v) for v in p.split(",")) for p in info["points"].split(";")]
        out.append(np.array(pts, np.int32))
    return out


def max_dev(q):
    angs = interior_angles(q)
    devs = [abs(a - 90.0) for a in angs if math.isfinite(a)]
    return max(devs) if devs else 0.0


def ortho_minarect(q):
    pts = np.asarray(q, np.float32).reshape(-1, 1, 2)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    return np.round(box).astype(np.int32)


def ortho_lsq(q):
    q = np.asarray(q, float)
    c = q.mean(axis=0)
    e1 = (q[1] - q[0]) + (q[2] - q[3])
    n1 = np.linalg.norm(e1)
    if n1 < 1e-6:
        return np.asarray(q, np.int32)
    u1 = e1 / n1
    u2 = np.array([-u1[1], u1[0]])
    d = q - c
    a = np.abs(d @ u1).mean()
    b = np.abs(d @ u2).mean()
    return np.round(np.array([c - a*u1 - b*u2, c + a*u1 - b*u2,
                              c + a*u1 + b*u2, c - a*u1 + b*u2])).astype(np.int32)


def eval_vs_gt(det_map, ids, dist=5.0):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_sig(GT_DIR, iid)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)
        tp, unmatched = match_boxes_point(det, gt, dist)
        TP += tp
        FN += len(gt) - tp
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
    tot = TP + FN
    rec = TP / tot * 100 if tot else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
            "real_unl": real_unl, "total_gt": tot,
            "recall": round(rec, 1), "corrected_f1": round(f1, 1)}


def main():
    ids = get_ids()
    print(f"LOOP-33: orthogonality enforcement vs TRUE GT ({GT_DIR})")
    print(f"  {len(ids)} benchmark images, D={D_EVAL:.0f}\n")

    # 1. deviation of frozen output
    devs = []
    for iid in ids:
        for q in load_boxes_sig(DET_DIR, iid):
            devs.append(max_dev(q))
    devs = np.array(devs)
    print("1. FROZEN pm_v5_2 output orthogonality deviation")
    print(f"   {len(devs)} boxes: mean={devs.mean():.2f} med={np.median(devs):.2f} "
          f"p90={np.percentile(devs,90):.2f} max={devs.max():.2f} deg")
    print(f"   >1deg: {np.sum(devs>1)} ({np.sum(devs>1)/len(devs)*100:.1f}%)  "
          f">5deg: {np.sum(devs>5)} ({np.sum(devs>5)/len(devs)*100:.1f}%)  "
          f">10deg: {np.sum(devs>10)} ({np.sum(devs>10)/len(devs)*100:.1f}%)")

    # 2. GT orthogonality
    gdevs = []
    for iid in ids:
        for q in load_boxes_sig(GT_DIR, iid):
            gdevs.append(max_dev(q))
    gdevs = np.array(gdevs)
    print(f"\n2. TRUE GT orthogonality ({len(gdevs)} boxes)")
    print(f"   mean={gdevs.mean():.2f} med={np.median(gdevs):.2f} "
          f"p90={np.percentile(gdevs,90):.2f} max={gdevs.max():.2f} deg")
    print(f"   >1deg: {np.sum(gdevs>1)} ({np.sum(gdevs>1)/len(gdevs)*100:.1f}%)  "
          f">5deg: {np.sum(gdevs>5)} ({np.sum(gdevs>5)/len(gdevs)*100:.1f}%)")

    # 3. TP impact
    print("\n3. TP impact of orthogonalization (vs RAW GT)")
    print(f"   {'variant':10s} {'TP':>5s} {'rawFP':>6s} {'trueFP':>7s} "
          f"{'realUnl':>8s} {'recall':>7s} {'corr_f1':>8s}")
    variants = {"none": lambda q: np.asarray(q, np.int32),
                "minarect": ortho_minarect, "lsq": ortho_lsq}
    results = {}
    for name, fn in variants.items():
        dm = {iid: [fn(q) for q in load_boxes_sig(DET_DIR, iid)] for iid in ids}
        m = eval_vs_gt(dm, ids, D_EVAL)
        results[name] = m
        print(f"   {name:10s} {m['tp']:5d} {m['raw_fp']:6d} {m['true_fp']:7d} "
              f"{m['real_unl']:8d} {m['recall']:6.1f}% {m['corrected_f1']:7.1f}%")

    out = os.path.join(ROOT, "benchmark", "cv_scope33.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"gt_dir": GT_DIR,
                   "output_dev": {"n": int(len(devs)), "mean": float(devs.mean()),
                                  "med": float(np.median(devs)),
                                  "p90": float(np.percentile(devs, 90)),
                                  "max": float(devs.max()),
                                  "gt_1deg": int(np.sum(devs > 1)),
                                  "gt_5deg": int(np.sum(devs > 5))},
                   "gt_dev": {"n": int(len(gdevs)), "mean": float(gdevs.mean()),
                              "med": float(np.median(gdevs)),
                              "p90": float(np.percentile(gdevs, 90)),
                              "max": float(gdevs.max())},
                   "variants": {k: {"tp": v["tp"], "raw_fp": v["raw_fp"],
                                    "true_fp": v["true_fp"],
                                    "recall": v["recall"],
                                    "corrected_f1": v["corrected_f1"]}
                                for k, v in results.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()