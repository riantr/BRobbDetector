"""Validate the detector against the repo's OWN hand-annotated labels.

data/Step3@test/*.json at commit 37c3232 are the original human annotations
shipped with the project — independent of every candidate GT dir. This scores
the frozen pm_v5_2 detection against them, and against raw / Compare-V1 for
the same images, so we can tell whether the benchmark's 82% recall against
Compare-V1 is real accuracy or mere agreement with a prior CV detector.
"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import (parse_points, match_boxes_point, is_real_unlabeled,
                             corner_match_distance)
from cv_scope34 import load_boxes_any, DET_DIR

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
CMP1 = os.path.join(DATA, "Compare-V1@CARS_202609_annotated")
HAND_DIR = os.path.join(ROOT, "data", "Step3@test")
BASE_COMMIT = "37c3232"
D = 5.0


def git_hand(iid):
    r = subprocess.run(["git", "show", f"{BASE_COMMIT}:data/Step3@test/{iid}.json"],
                       capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        return []
    j = json.loads(r.stdout.decode("utf-8"))
    return [parse_points(i["points"]) for i in j.get("infos", [])]


def score(det_map, gt_map, ids, label):
    exp = load_expected()
    TP = FN = true_fp = real_unl = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = gt_map.get(iid, [])
        det = [d.astype(np.int32) for d in det_map.get(iid, [])]
        md = cv2.dilate(red_mask(img), np.ones((3, 3), np.uint8), iterations=3)
        tp, unmatched = match_boxes_point(det, gt, D)
        TP += tp; FN += len(gt) - tp
        for di in unmatched:
            if is_real_unlabeled(det[di], exp.get(iid, 0), len(gt), md):
                real_unl += 1
            else:
                true_fp += 1
    tot = TP + FN
    rec = TP / tot * 100 if tot else 0
    prec = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  vs {label:12s} TP={TP:4d} FN={FN:4d} trueFP={true_fp:4d} "
          f"realUnl={real_unl:4d} recall={rec:5.1f}% corr_f1={f1:5.1f}")
    return {"tp": TP, "fn": FN, "true_fp": true_fp, "real_unlabeled": real_unl,
            "recall": round(rec, 1), "corrected_f1": round(f1, 1)}


def main():
    ids = sorted(f[:-5] for f in os.listdir(HAND_DIR) if f.endswith(".json"))
    ids = [i for i in ids if os.path.exists(os.path.join(IMG, i + ".jpg"))]
    det_map = {i: load_boxes_any(DET_DIR, i) for i in ids}
    hand = {i: git_hand(i) for i in ids}
    raw = {i: load_boxes_any(RAW, i) for i in ids}
    cmp1 = {i: load_boxes_any(CMP1, i) for i in ids}

    n_h = sum(len(v) for v in hand.values())
    n_r = sum(len(v) for v in raw.values())
    n_c = sum(len(v) for v in cmp1.values())
    n_d = sum(len(v) for v in det_map.values())
    print(f"40-image gold set: hand={n_h} raw={n_r} cmp1={n_c} det={n_d}\n")

    print("=== detector (pm_v5_2) scored against each reference ===")
    res = {"hand": score(det_map, hand, ids, "HAND(git)"),
           "raw": score(det_map, raw, ids, "raw"),
           "cmp1": score(det_map, cmp1, ids, "Compare-V1")}

    print("\n=== reference-vs-reference corner distance (min per box) ===")
    for a_name, a in [("hand", hand), ("raw", raw), ("cmp1", cmp1)]:
        for b_name, b in [("hand", hand), ("raw", raw), ("cmp1", cmp1)]:
            if a_name >= b_name:
                continue
            ds = []
            for iid in ids:
                for x in a.get(iid, []):
                    best = min([corner_match_distance(x, y) for y in b.get(iid, [])],
                               default=float('inf'))
                    if np.isfinite(best):
                        ds.append(best)
            d = np.array(ds)
            if len(d):
                print(f"  {a_name:5s} -> {b_name:5s} n={len(d):4d} "
                      f"median={np.median(d):7.2f}  p90={np.percentile(d,90):7.2f}  "
                      f"max={d.max():7.2f}")

    out = os.path.join(ROOT, "benchmark", "cv_scope36_hand_labels.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "d_eval": D,
                   "hand_boxes": n_h, "raw_boxes": n_r, "cmp1_boxes": n_c,
                   "det_boxes": n_d, "det_vs": res}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()