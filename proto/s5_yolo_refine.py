"""S5: YOLO-OBB as a PROPOSER + geometric corner refinement.

YOLO-OBB recall is 97.7 vs pm_v10's 83.7 - a 14-point recall gap it loses purely
on corner precision under the point-match metric.  This keeps YOLO's box as a
PROPOSAL and refits its 4 sides on the red strokes with the shipped machinery
(`_refine_sides` + S4's nearest-run selection), re-intersects, re-enforces exact
orthogonality and applies the shipped gates.

It is NOT a union of raw YOLO boxes (LOOP-8a: +1 TP / +128 FP) and NOT classical
re-detection on a crop (LOOP-8b: -100).

  yref    refined YOLO proposals only
  yunion  refined YOLO proposals UNION the shipped pass, dedup at the shipped IoU

Proposals are extracted once from the production ONNX model (conf 0.55, the
LOOP-6 operating point) and cached, so the 3 sweeps cost one inference pass.

Usage: python proto/s5_yolo_refine.py --workers 12
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG, red_mask, load_expected
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from cv_hybrid import dedup_keep_order
from quad_ortho import quad_on_boundary
from scope45_holes import (P, HOLE_DEFAULT, PIPE_DEFAULT, detect, _refine_sides,
                           _gate, _enforce, _on_edge, _red_support)
from s1_inward_complete import score, tp_boxset
from s4_run_select import refine_runs

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"
MODEL = 'BRobbDetector/Assets/yolov8s_obb.onnx'
CACHE = 'benchmark/_tmp_s5_yolo.json'
CONF = 0.55


def extract_proposals(ids):
    if os.path.exists(CACHE):
        d = json.load(open(CACHE, encoding='utf-8'))
        print('proposals cached: %d images' % len(d))
        return {k: [np.asarray(q, np.int32) for q in v] for k, v in d.items()}
    from ultralytics import YOLO
    m = YOLO(MODEL, task='obb')
    out = {}
    for k, iid in enumerate(ids, 1):
        r = m.predict(os.path.join(IMG, iid + '.jpg'), conf=CONF, verbose=False)[0]
        qs = []
        if r.obb is not None and len(r.obb) > 0:
            for cx, cy, w, h, th in r.obb.xywhr.cpu().numpy():
                c, s = np.cos(float(th)), np.sin(float(th))
                R = np.array([[c, -s], [s, c]])
                pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                                [w / 2, h / 2], [-w / 2, h / 2]])
                qs.append((pts @ R.T + np.array([cx, cy])).astype(np.int32))
        out[iid] = qs
        if k % 100 == 0:
            print('  proposals %d/%d' % (k, len(ids)), flush=True)
    json.dump({k: [q.tolist() for q in v] for k, v in out.items()},
              open(CACHE, 'w', encoding='utf-8'))
    print('proposals cached -> %s' % CACHE)
    return out


def refine_proposal(mask, q, p):
    """Refit a YOLO rectangle's 4 sides on the red strokes, then gate it."""
    H, W = mask.shape
    q = np.asarray(q, float)
    r = _refine_sides(mask, q, band=5.0, min_pts=8, tol=3.0)
    if r is not None:
        q = r
    r2 = refine_runs(mask, q, 4.0, min_pts=8, tol=3.0)
    if r2 is not None:
        q = r2
    q = _enforce(q, PIPE_DEFAULT["ortho"])
    q = _gate(q, W, H, p)
    if q is None:
        return None
    if _on_edge(q, W, H, p):
        return None
    if _red_support(mask, q) < HOLE_DEFAULT["min_red_support"]:
        return None
    return np.asarray(q, np.int32)


def _work(task):
    iid, mode = task
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        return iid, mode, []
    m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
    if P["denoise_min_area"] > 0:
        m = denoise(m, P["denoise_min_area"])
    if mode == "base":
        det = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
    else:
        props = PROPS.get(iid, [])
        ref = [r for r in (refine_proposal(m, q, P) for q in props) if r is not None]
        if mode == "yref":
            det = dedup_keep_order(ref, PIPE_DEFAULT["dedup_iou"])
        else:                                   # yunion
            ship = detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT, mask=m)
            det = dedup_keep_order(list(ship) + list(ref), PIPE_DEFAULT["dedup_iou"])
    return iid, mode, [np.asarray(q, np.int32) for q in det]


PROPS = {}


def _init(props):
    global PROPS
    PROPS = props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="benchmark/s5_yolo_refine.json")
    ap.add_argument("--modes", default="base,yref,yunion")
    args = ap.parse_args()

    ids = get_ids()
    gt_map = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt_map[iid] = [g for g in load_boxes_any(RAW, iid)
                       if not quad_on_boundary(g, W, H, 0.0)]
    exp = load_expected()
    props = extract_proposals(ids)
    nprop = sum(len(v) for v in props.values())
    print('%d images, %d GT boxes, %d YOLO proposals (conf %.2f)'
          % (len(ids), sum(len(v) for v in gt_map.values()), nprop, CONF), flush=True)

    modes = args.modes.split(",")
    out, dets = {}, {}
    for mode in modes:
        det_map = {}
        with Pool(args.workers, initializer=_init, initargs=(props,)) as pool:
            for iid, _m, det in pool.imap_unordered(
                    _work, [(i, mode) for i in ids], chunksize=3):
                det_map[iid] = det
        dets[mode] = det_map
        r = score(det_map, ids, gt_map, exp)
        out[mode] = {"metrics": r}
        print("  %-7s D=3 %4d/%4d | D=5 %4d/%4d | D=8 %4d/%4d | D=12 %4d/%4d "
              "| dets=%4d corr_f1@5=%.1f"
              % (mode, r["3"]["tp"], r["3"]["true_fp"], r["5"]["tp"],
                 r["5"]["true_fp"], r["8"]["tp"], r["8"]["true_fp"],
                 r["12"]["tp"], r["12"]["true_fp"], r["5"]["n_det"],
                 r["5"]["corr_f1"]), flush=True)

    base = out["base"]["metrics"]
    print("\n== delta vs base ==")
    for m in modes:
        if m == "base":
            continue
        d = out[m]["metrics"]
        print("  %-7s dTP@3 %+4d dTP@5 %+4d dTP@8 %+4d dTrueFP@5 %+4d dcorr_f1 %+5.1f"
              % (m, d["3"]["tp"] - base["3"]["tp"], d["5"]["tp"] - base["5"]["tp"],
                 d["8"]["tp"] - base["8"]["tp"],
                 d["5"]["true_fp"] - base["5"]["true_fp"],
                 d["5"]["corr_f1"] - base["5"]["corr_f1"]))

    sb = tp_boxset(dets["base"], gt_map, ids)
    print("\n== TP-set decomposition at D=5 ==")
    for m in modes:
        if m == "base":
            continue
        sm = tp_boxset(dets[m], gt_map, ids)
        print("  %-7s base=%d cand=%d  GAINS=%d  LOSSES=%d"
              % (m, len(sb), len(sm), len(sm - sb), len(sb - sm)))
        out[m]["tp_set"] = {"gains": len(sm - sb), "losses": len(sb - sm)}

    json.dump({"results": out, "n_proposals": nprop, "conf": CONF},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()