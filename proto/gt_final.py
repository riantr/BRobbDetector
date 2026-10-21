"""GT arbitration, final evidence bundle (LOOP-36 close-out).

The benchmark GT was "Compare-V1@CARS_202609_annotated". This script proves
which label set is the real hand-drawn ground truth, using evidence that is
independent of every candidate dir:

  A. RED-LINE SUPPORT IN THE DETECTOR'S INPUT IMAGE
     Image@CARS_202609_annotated is what the detector consumes. The human
     annotation is burned into it as a red rectangle, so the TRUE GT must lie
     exactly on red pixels of that image. This is the decisive test.
     (Control: the same boxes measured on Image@CARS_202609_raw, which has no
     annotation burned in.)

  B. ANNOTATION-CSV COUNT AGREEMENT
     2026-09-*_detail.csv records how many annotations exist per image. That
     record is independent of every label dir.

  C. REPO'S OWN HAND LABELS
     data/Step3@test/*.json at commit 37c3232 are the original human
     annotations shipped with the project (40 images, 120 boxes).

  D. ENFORCEMENT x GT MATRIX
     Re-runs the orthogonality-enforcement comparison against each candidate GT
     so the "enforcing orthogonality costs 20% TP" claim can be re-checked
     against the real GT.

Usage: python proto/gt_final.py
"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import red_support, match_boxes_point, is_real_unlabeled
from canny_quad import get_ids
from cv_scope34 import (minarect, lsq_center, ortho_edge, ortho_blend,
                        load_boxes_any, DET_DIR)

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
CMP1 = os.path.join(DATA, "Compare-V1@CARS_202609_annotated")
CMP = os.path.join(DATA, "Compare@CARS_202609_annotated")
DETOUT = os.path.join(DATA, "Label@CARS_202609_annotated")
RAW_IMG = os.path.join(DATA, "Image@CARS_202609_raw")
HAND_DIR = os.path.join(ROOT, "data", "Step3@test")
BASE_COMMIT = "37c3232"

CANDS = [("raw", RAW), ("Compare-V1", CMP1), ("Compare", CMP),
         ("detector-output", DETOUT)]
DIST_VALUES = [3, 5, 8, 12]


def tight_mask(img):
    return cv2.dilate(red_mask(img), np.ones((3, 3), np.uint8), iterations=1)


def support_stats(d, ids, img_dir):
    vals = []
    for iid in ids:
        p = os.path.join(img_dir, iid + ".jpg")
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        md = tight_mask(img)
        for b in load_boxes_any(d, iid):
            vals.append(red_support(b.astype(np.int32), md))
    v = np.array(vals)
    if not len(v):
        return {}
    return {"n_boxes": int(len(v)), "mean": round(float(v.mean()), 3),
            "pct_ge_0.8": round(float((v >= 0.8).mean() * 100), 1),
            "pct_lt_0.2": round(float((v < 0.2).mean() * 100), 1)}


def git_hand(iid):
    r = subprocess.run(["git", "show", f"{BASE_COMMIT}:data/Step3@test/{iid}.json"],
                       capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        return []
    j = json.loads(r.stdout.decode("utf-8-sig"))
    return [np.array([[int(x), int(y)] for x, y in
                      (q.split(",") for q in i["points"].split(";"))], np.int32)
            for i in j.get("infos", [])]


def greedy_assign(a, b):
    """1-1 assignment minimising max-corner distance; returns matched distances."""
    from eval_pointmatch import corner_match_distance
    pairs = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d = corner_match_distance(x, y)
            if np.isfinite(d):
                pairs.append((d, i, j))
    pairs.sort()
    ua, ub, res = set(), set(), []
    for d, i, j in pairs:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); res.append(d)
    return res


def eval_vs_gt(det_map, ids, gt_dir, dist):
    exp = load_expected()
    TP = FN = raw_fp = true_fp = real_unl = n_gt = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes_any(gt_dir, iid)
        n_gt += len(gt)
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        md = cv2.dilate(red_mask(img), np.ones((3, 3), np.uint8), iterations=3)
        tp, unmatched = match_boxes_point(det, gt, dist)
        TP += tp; FN += len(gt) - tp; raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], exp.get(iid, 0), len(gt), md):
                real_unl += 1
            else:
                true_fp += 1
    tot = TP + FN
    rec = TP / tot * 100 if tot else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": TP, "fn": FN, "gt": n_gt, "raw_fp": raw_fp, "true_fp": true_fp,
            "real_unlabeled": real_unl, "recall": round(rec, 1),
            "corrected_f1": round(f1, 1)}


def deviation(q):
    q = np.asarray(q, float)
    worst = 0.0
    for i in range(4):
        p0, p1, p2 = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        worst = max(worst, abs(np.degrees(np.arccos(c)) - 90.0))
    return worst


def make_threshold(alpha_fn, thresh):
    def f(q):
        if deviation(q) <= thresh:
            return np.asarray(q, np.int32)
        o = ortho_edge(q).astype(float)
        a = alpha_fn
        return np.round((1 - a) * np.asarray(q, float) + a * o).astype(np.int32)
    return f


METHODS = {
    "none": lambda q: np.asarray(q, np.int32),
    "minarect": minarect,
    "ortho_edge": ortho_edge,
    "blend0.3": lambda q: ortho_blend(q, 0.3),
    "thr5_strict": make_threshold(1.0, 5),
    "thr10_strict": make_threshold(1.0, 10),
    "thr10_blend": make_threshold(0.3, 10),
}


def main():
    ids = get_ids()
    exp = load_expected()
    out = {"n_images": len(ids), "det_dir": DET_DIR, "img_dir": IMG}

    # --- A. red-line support in the detector's input image -----------------
    print("=== A. red-line support of each candidate label set ===")
    print("    (the human annotation is burned into Image@CARS_202609_annotated)")
    sup = {}
    for name, d in CANDS:
        sup[name] = {"in_input_image": support_stats(d, ids, IMG),
                      "in_raw_image(control)": support_stats(d, ids, RAW_IMG)}
        a = sup[name]["in_input_image"]
        b = sup[name]["in_raw_image(control)"]
        print(f"  {name:16s} input-image: n={a.get('n_boxes',0):4d} mean={a.get('mean',0):.3f} "
              f">=0.8: {a.get('pct_ge_0.8',0):5.1f}%  <0.2: {a.get('pct_lt_0.2',0):5.1f}%   "
              f"| raw-image control: mean={b.get('mean',0):.3f}")
    out["red_support"] = sup

    # --- B. CSV count agreement -------------------------------------------
    print("\n=== B. count agreement with the annotation CSV ===")
    csvn = {i: exp.get(i, 0) for i in ids}
    cnt = {}
    for name, d in CANDS:
        diff = np.array([len(load_boxes_any(d, i)) - csvn[i] for i in ids])
        cnt[name] = {"total": int(sum(len(load_boxes_any(d, i)) for i in ids)),
                     "csv_total": int(sum(csvn.values())),
                     "exact": int((diff == 0).sum()), "within1": int((np.abs(diff) <= 1).sum()),
                     "mean_err": round(float(diff.mean()), 3)}
        print(f"  {name:16s} total={cnt[name]['total']:5d} csv={cnt[name]['csv_total']:5d} "
              f"exact={cnt[name]['exact']:3d}/500 within1={cnt[name]['within1']:3d}/500 "
              f"mean_err={cnt[name]['mean_err']:+.3f}")
    out["csv_count_agreement"] = cnt

    # --- C. repo's own hand labels ----------------------------------------
    print("\n=== C. agreement with the repo's own hand labels (commit %s) ===" % BASE_COMMIT)
    hids = sorted(f[:-5] for f in os.listdir(HAND_DIR) if f.endswith(".json"))
    hids = [i for i in hids if os.path.exists(os.path.join(IMG, i + ".jpg"))]
    hand = {i: git_hand(i) for i in hids}
    hand_out = {}
    for name, d in [("raw", RAW), ("Compare-V1", CMP1), ("Compare", CMP)]:
        ds, mism = [], 0
        for i in hids:
            ds += greedy_assign(hand[i], load_boxes_any(d, i))
        a = np.array(ds)
        hand_out[name] = {"n_hand": int(sum(len(v) for v in hand.values())),
                          "n_matched": int(len(a)),
                          "median": round(float(np.median(a)), 2),
                          "p90": round(float(np.percentile(a, 90)), 2),
                          "max": round(float(a.max()), 2),
                          "pct_within5": round(float((a <= 5).mean() * 100), 1)}
        print(f"  hand -> {name:12s} matched={len(a):3d}  median={np.median(a):6.2f} "
              f"p90={np.percentile(a,90):6.2f} max={a.max():7.2f}  within5={hand_out[name]['pct_within5']:.1f}%")
    out["hand_label_agreement"] = hand_out

    # --- D. enforcement x GT ----------------------------------------------
    print("\n=== D. orthogonality enforcement x candidate GT (D=5) ===")
    gts = {"raw": RAW, "Compare-V1": CMP1}
    rows = {}
    for mname, fn in METHODS.items():
        dm = {iid: [fn(q) for q in load_boxes_any(DET_DIR, iid)] for iid in ids}
        rows[mname] = {g: eval_vs_gt(dm, ids, gd, 5.0) for g, gd in gts.items()}
        r = rows[mname]["raw"]; c = rows[mname]["Compare-V1"]
        print(f"  {mname:12s} raw: TP={r['tp']:4d} corr_f1={r['corrected_f1']:5.1f} "
              f"recall={r['recall']:5.1f}%   |   Compare-V1: TP={c['tp']:4d} "
              f"corr_f1={c['corrected_f1']:5.1f}")
    out["enforcement_x_gt_d5"] = rows

    print("\n=== D2. D-sweep against the TRUE GT (raw) ===")
    dsweep = {}
    for mname, fn in METHODS.items():
        dm = {iid: [fn(q) for q in load_boxes_any(DET_DIR, iid)] for iid in ids}
        dsweep[mname] = {str(d): eval_vs_gt(dm, ids, RAW, float(d)) for d in DIST_VALUES}
        s = dsweep[mname]
        print(f"  {mname:12s} " + "  ".join(f"D{d}: TP={s[str(d)]['tp']:4d}/{s[str(d)]['corrected_f1']:5.1f}"
                                           for d in DIST_VALUES))
    out["d_sweep_vs_raw"] = dsweep

    path = os.path.join(ROOT, "benchmark", "gt_final.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()