"""LOOP-77: the new edge rule INSIDE the orthogonality gate (end to end).

LOOP-76 measured the rule as a post-hoc filter on the finished detections.
This script puts the rule where it belongs -- in the orthogonality gate
(`proto/quad_ortho.py::is_orthogonal_quad`) -- and measures the real
end-to-end result for every plausible variant:

  V0  baseline (old code: waiver at the border, no edge rule)
  V1  edge_band=0, no waiver      -> literal rule, every corner must be 90 deg
  V2  edge_band=0, waiver kept    -> literal rule + legacy border waiver
  V3  edge_band=6, no waiver      -> production boundary_band as edge band
  V4  edge_band=1, no waiver      -> one pixel ring

Every variant is scored twice:
  * against the raw GT (1121 boxes) - the old comparison basis, and
  * against the GT filtered with the SAME rule (the honest comparison: a
    rectangle that touches the image edge is not a valid rectangle, so it is
    not a valid target either).
"""
import os, sys, json, argparse, pickle, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

import scope45_holes as S
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, evaluate, load_all
from scope45_holes import mask_cache
from quad_ortho import is_orthogonal_quad as ORIG_ORTHO
from common import IMG, red_mask

VARIANTS = [
    # name,        edge_band, waive, boundary_band
    ("V1_edge0_strict", 0.0, False, 6.0),
    ("V2_edge0_waive", 0.0, True, 6.0),
    ("V3_edge6_strict", 6.0, False, 6.0),
    ("V4_edge1_strict", 1.0, False, 6.0),
]
CACHE = "benchmark/scope77_dets.pkl"


def on_edge(quad, W, H, band):
    q = np.asarray(quad, float)
    for i in range(4):
        x, y = q[i]
        if x <= band or x >= (W - 1 - band) or y <= band or y >= (H - 1 - band):
            return True
    return False


def detect_all(ids, edge_band, waive, boundary_band):
    S.is_orthogonal_quad = functools.partial(
        ORIG_ORTHO, edge_band=edge_band, waive_boundary=waive,
        boundary_band=boundary_band)
    dets = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            dets[iid] = []
            continue
        dets[iid] = [np.asarray(q, np.int32) for q in
                     detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
    return dets


def score(tag, ids, gt_map, det_map, exp):
    ndet, total_gt, res = evaluate(ids, gt_map, det_map, exp)
    r5 = res["5"]
    print(f"{tag:34s} ngt={total_gt:4d} ndet={ndet:4d} "
          f"D=5 TP={r5['tp']:4d} rec={r5['recall']:5.1f} prec={r5['precision']:5.1f} "
          f"trueFP={r5['true_fp']:4d} unl={r5['real_unl']:3d} "
          f"corr_f1={r5['corrected_f1']:5.1f}")
    return {"ngt": total_gt, "ndet": ndet, "res": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/scope77_edge_gate.json")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                      np.ones((3, 3), np.uint8), iterations=3) > 0

    shapes = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is not None:
            shapes[iid] = (img.shape[1], img.shape[0])

    out = {}

    # ---- V0 baseline: detections produced by the ORIGINAL code ----------
    with open("benchmark/scope76_dets.pkl", "rb") as f:
        base_dets = pickle.load(f)
    out["V0_baseline"] = score("V0 baseline (raw GT)", ids, gt_map, base_dets, exp)

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)

    for name, edge_band, waive, bb in VARIANTS:
        key = f"{name}"
        if key in cache:
            dets = cache[key]
        else:
            dets = detect_all(ids, edge_band, waive, bb)
            cache[key] = dets
            with open(CACHE, "wb") as f:
                pickle.dump(cache, f)
        out[name + "_rawGT"] = score(f"{name} (raw GT)", ids, gt_map, dets, exp)
        gm = {iid: [g for g in gt_map[iid]
                    if not on_edge(g, *shapes.get(iid, (0, 0)), edge_band)]
              for iid in ids}
        out[name + "_consistentGT"] = score(f"{name} (consistent GT)", ids, gm, dets, exp)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()