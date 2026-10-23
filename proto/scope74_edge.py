"""LOOP-74: measure the impact of the new rectangle validity rule.

Rule: a detected rectangle is INVALID when one of its four vertices lies on
the image edge (previously such a vertex was WAIVED from the orthogonality
check instead).

Reports, for several edge bands:
  * how many GT boxes touch the edge,
  * how many detections touch the edge,
  * the benchmark score when edge-touching detections are dropped.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import get_ids
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, evaluate, load_all
from scope45_holes import mask_cache
from common import red_mask

BANDS = [0, 1, 2, 6]


def on_edge(quad, W, H, band):
    q = np.asarray(quad, float)
    for i in range(4):
        x, y = q[i]
        if x <= band or x >= W - band or y <= band or y >= H - band:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/scope74_edge.json")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()

    # evaluation mask (hue_max=10, dilated) - same heuristic as eval_pm_benchmark
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                      np.ones((3, 3), np.uint8), iterations=3) > 0

    # ---- GT statistics -------------------------------------------------
    gt_edge = {b: 0 for b in BANDS}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = img.shape[:2]
        for g in gt_map[iid]:
            for b in BANDS:
                if on_edge(g, W, H, b):
                    gt_edge[b] += 1

    # ---- detections + benchmark with/without the rule ------------------
    base = {}
    for b in BANDS:
        det_map = {}
        dropped = 0
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                det_map[iid] = []
                continue
            H, W = img.shape[:2]
            quads = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)
            keep = []
            for q in quads:
                if on_edge(q, W, H, b):
                    dropped += 1
                    continue
                keep.append(q)
            det_map[iid] = keep
        ndet, total_gt, res = evaluate(ids, gt_map, det_map, exp)
        base[b] = {"dropped": dropped, "ndet": ndet, "res": res}
        r5 = res["5"]
        print(f"band={b}: gt_edge={gt_edge[b]:4d}  det_dropped={dropped:4d}  "
              f"ndet={ndet:4d}  D=5 TP={r5['tp']:4d} trueFP={r5['true_fp']:4d} "
              f"real_unl={r5['real_unl']:3d} corr_f1={r5['corrected_f1']:5.1f}")

    # ---- reference: current production numbers (no edge rule) ----------
    det_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        det_map[iid] = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT) if img is not None else []
    ndet, total_gt, res = evaluate(ids, gt_map, det_map, exp)
    r5 = res["5"]
    print(f"NO EDGE RULE: ndet={ndet} D=5 TP={r5['tp']} trueFP={r5['true_fp']} "
          f"real_unl={r5['real_unl']} corr_f1={r5['corrected_f1']}")

    out = {"gt_edge": gt_edge, "no_rule": {"ndet": ndet, "res": res},
           "bands": {str(b): base[b] for b in BANDS}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()