"""LOOP-76: the edge rule applied CONSISTENTLY to detections and to the GT.

LOOP-74 measured the rule as a *detection filter only*: dropping every
detection that has a vertex on the image edge.  It showed a regression at
every band (-22..-54 TP), because 45 of the 1121 hand-annotated GT boxes
also have a vertex exactly on the border (the human drew a box that the
image edge clips).

If the new rule really declares "a rectangle with a vertex on the image edge
is not a valid rectangle", then the ground truth must obey the same rule:
those clipped GT boxes are not valid targets either.  This script measures:

  A) baseline                       (no rule)
  B) det-only filter                (LOOP-74 result, redone with a symmetric band)
  C) consistent filter (det + GT)   <- the honest benchmark of the new rule
  D) GT-only filter                 (reference: what the rule would cost if no
                                     detection were ever filtered)

Bands are symmetric: a vertex is on the edge when
    x <= band or x >= (W - 1 - band) or y <= band or y >= (H - 1 - band)
band=0 is the literal reading ("exactly on the outermost pixel ring").
"""
import os, sys, json, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, evaluate, load_all
from scope45_holes import mask_cache

BANDS = [0, 1, 2, 6]
CACHE = "benchmark/scope76_dets.pkl"


def on_edge(quad, W, H, band):
    q = np.asarray(quad, float)
    for i in range(4):
        x, y = q[i]
        if x <= band or x >= (W - 1 - band) or y <= band or y >= (H - 1 - band):
            return True
    return False


def get_dets(ids):
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    dets = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            dets[iid] = []
            continue
        dets[iid] = [np.asarray(q, np.int32) for q in
                     detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
    with open(CACHE, "wb") as f:
        pickle.dump(dets, f)
    return dets


def report(tag, ids, gt_map, det_map, exp):
    ndet, total_gt, res = evaluate(ids, gt_map, det_map, exp)
    r5 = res["5"]
    print(f"{tag:38s} ngt={total_gt:4d} ndet={ndet:4d} "
          f"D=5 TP={r5['tp']:4d} rec={r5['recall']:5.1f} prec={r5['precision']:5.1f} "
          f"trueFP={r5['true_fp']:4d} real_unl={r5['real_unl']:3d} "
          f"corr_f1={r5['corrected_f1']:5.1f}")
    return {"ngt": total_gt, "ndet": ndet, "res": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/scope76_edge_consistent.json")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()

    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                      np.ones((3, 3), np.uint8), iterations=3) > 0

    dets = get_dets(ids)
    shapes = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is not None:
            shapes[iid] = (img.shape[1], img.shape[0])

    out = {}

    out["A_baseline"] = report("A baseline (no rule)", ids, gt_map, dets, exp)

    for b in BANDS:
        # B) det-only
        dm = {}
        for iid in ids:
            W, H = shapes.get(iid, (0, 0))
            dm[iid] = [q for q in dets[iid] if not on_edge(q, W, H, b)]
        out[f"B_detonly_b{b}"] = report(f"B det-only  band={b}", ids, gt_map, dm, exp)

        # C) consistent: det + GT
        gm = {}
        for iid in ids:
            W, H = shapes.get(iid, (0, 0))
            gm[iid] = [g for g in gt_map[iid] if not on_edge(g, W, H, b)]
        out[f"C_consistent_b{b}"] = report(f"C consistent band={b}", ids, gm, dm, exp)

        # D) GT-only
        out[f"D_gtonly_b{b}"] = report(f"D gt-only   band={b}", ids, gm, dets, exp)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()