"""LOOP-8: Hybrid DL+CV algorithms tested against pm_v5 baseline.

Base (pm_v5 winner): hybrid(hull->edgefit->mar) + admission filter o15_ba.
Each algorithm adds exactly ONE new dimension:

  8a yolo_union : YOLOv8-OBB + pm_v5 union with keep-order dedup (classical first)
  8b yolo_crop  : YOLOv8-OBB proposals -> crop -> classical corner extraction
  8c adaptive   : adaptive per-image HSV threshold (lower for faint-box images)
  8d kmeans     : K-means color clustering for red segmentation

Usage:
  python proto/cv_scope8.py            # run all 4
  python proto/cv_scope8.py --only yolo_union  # run one
"""
import sys, os, json, argparse, math, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, iou_obb, load_expected
from canny_quad import get_ids, to_json, DEFAULT as CQ_DEFAULT, denoise, _quad_adaptive, _side_ok, _order_quad
from cv_hybrid import (outer_contours, _pass, dedup_keep_order, _prep,
                       FIT_DEFAULT, evaluate, ROOT, DIST_VALUES,
                       _edgefit_corners, _sector_corners)
from cv_scope4 import _base_quads, HY_ORDER
from eval_pointmatch import load_boxes
from quad_ortho import is_orthogonal_quad

PM5 = {"ortho_tol": 15.0, "boundary_band": 12.0, "max_aspect": 12.0}
FROZEN_PM5 = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")
YOLO_MODEL = os.path.join(ROOT, "tmp_yolo", "train", "weights", "best.onnx")
CONF = 0.55


def pm5_params():
    p = dict(CQ_DEFAULT)
    p.update(FIT_DEFAULT)
    p.update(PM5)
    return p


def run_yolo_detect(ids):
    """Run YOLOv8-OBB on all ids, return {iid: [quad_array, ...]}."""
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL, task="obb")
    det_map = {}
    for n, iid in enumerate(ids):
        img_path = os.path.join(IMG, iid + ".jpg")
        results = model.predict(img_path, verbose=False, conf=CONF)
        boxes = []
        if results and results[0].obb is not None:
            for obb in results[0].obb:
                pts = obb.cpu().numpy().xyxyxyxy[0].astype(np.int32)
                if len(pts) == 4:
                    boxes.append(pts)
        det_map[iid] = boxes
        if (n + 1) % 100 == 0:
            print(f"  [YOLO {n+1}/{len(ids)}] boxes={sum(len(v) for v in det_map.values())}")
    return det_map


def save_yolo_output(det_map, ids):
    out_dir = os.path.join(ROOT, "tmp_yolo_pm_out")
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for iid in ids:
        quads = det_map.get(iid, [])
        total += len(quads)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in quads]), f,
                      ensure_ascii=False)
    print(f"  YOLO: {total} boxes -> {out_dir}")
    return out_dir


# ============ 8a: YOLO + pm_v5 union ============

def run_yolo_union(ids, pm5_det, yolo_det, p):
    """Union pm_v5 + YOLO with keep-order dedup (classical first)."""
    det_map = {}
    total = 0
    for iid in ids:
        pm5_boxes = [q.astype(np.int32) for q in pm5_det.get(iid, [])]
        yolo_boxes = [q.astype(np.int32) for q in yolo_det.get(iid, [])]
        combined = pm5_boxes + yolo_boxes
        hy = dedup_keep_order(combined, p["dedup_iou"])
        det_map[iid] = hy
        total += len(hy)
    print(f"  yolo_union: {total} boxes")
    return det_map


# ============ 8b: YOLO crop -> classical corner extraction ============

def yolo_crop_classical(img, yolo_boxes, p):
    """For each YOLO OBB, crop the image and run classical pipeline on the crop."""
    H, W = img.shape[:2]
    all_quads = []
    for yolo_q in yolo_boxes:
        yolo_q = np.asarray(yolo_q, np.int32)
        xs, ys = yolo_q[:, 0], yolo_q[:, 1]
        x0, x1 = max(0, xs.min() - 20), min(W, xs.max() + 20)
        y0, y1 = max(0, ys.min() - 20), min(H, ys.max() + 20)
        if x1 - x0 < p["min_side"] or y1 - y0 < p["min_side"]:
            all_quads.append(yolo_q)
            continue
        crop = img[y0:y1, x0:x1].copy()
        ch, cw = crop.shape[:2]
        mask = red_mask(crop)
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        outer = []
        if hier is not None:
            hier = hier[0]
            for i, c in enumerate(cnts):
                if hier[i][3] != -1:
                    continue
                if cv2.contourArea(c) >= p["min_area"] * 0.3:
                    outer.append(c)
        best_q = None
        best_score = -1
        for c in outer:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base[m], cw, ch, p)
                if q is not None:
                    q_global = q.copy().astype(np.float64)
                    q_global[:, 0] += x0
                    q_global[:, 1] += y0
                    q_global = q_global.astype(np.int32)
                    yolo_f = yolo_q.astype(np.float64)
                    qg_f = q_global.astype(np.float64)
                    from itertools import permutations
                    best_dist = float('inf')
                    for perm in permutations(range(4)):
                        mx = max(np.linalg.norm(qg_f[i] - yolo_f[perm[i]]) for i in range(4))
                        best_dist = min(best_dist, mx)
                    if best_dist < 50:
                        score = 50 - best_dist
                        if score > best_score:
                            best_score = score
                            best_q = q_global
        if best_q is not None:
            all_quads.append(best_q)
        else:
            all_quads.append(yolo_q)
    return dedup_keep_order(all_quads, p["dedup_iou"])


def run_yolo_crop(ids, yolo_det, p):
    det_map = {}
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        quads = yolo_crop_classical(img, yolo_det.get(iid, []), p)
        det_map[iid] = quads
        total += len(quads)
        if (n + 1) % 100 == 0:
            print(f"  [yolo_crop {n+1}/{len(ids)}] boxes={total}")
    print(f"  yolo_crop: {total} boxes -> tmp_sc8_yolo_crop")
    out_dir = os.path.join(ROOT, "tmp_sc8_yolo_crop")
    os.makedirs(out_dir, exist_ok=True)
    for iid in ids:
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in det_map[iid]]), f,
                      ensure_ascii=False)
    return det_map


# ============ 8c: Adaptive per-image HSV threshold ============

def adaptive_red_mask(img, p):
    """HSV red mask with adaptive threshold based on image statistics."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_pixels = ((h < 15) | (h > 165)) & (s > 50) & (v > 80)
    if red_pixels.sum() < 100:
        smin, vmin = 40, 60
    elif red_pixels.sum() < 1000:
        smin, vmin = 60, 90
    else:
        smin, vmin = 80, 110
    lo = cv2.inRange(hsv, (0, smin, vmin), (10, 255, 255))
    hi = cv2.inRange(hsv, (170, smin, vmin), (180, 255, 255))
    return lo | hi


def run_adaptive(ids, p):
    det_map = {}
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        H, W = img.shape[:2]
        mask = adaptive_red_mask(img, p)
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        outer = []
        if hier is not None:
            hier = hier[0]
            for i, c in enumerate(cnts):
                if hier[i][3] != -1:
                    continue
                if cv2.contourArea(c) >= p["min_area"]:
                    outer.append(c)
        res = {m: [] for m in HY_ORDER}
        for c in outer:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base[m], W, H, p)
                if q is not None:
                    res[m].append(q)
        for m in HY_ORDER:
            res[m] = dedup_keep_order(res[m], p["dedup_iou"])
        hy = dedup_keep_order(
            [q for m in HY_ORDER for q in res[m]], p["dedup_iou"])
        det_map[iid] = hy
        total += len(hy)
        if (n + 1) % 100 == 0:
            print(f"  [adaptive {n+1}/{len(ids)}] boxes={total}")
    print(f"  adaptive: {total} boxes -> tmp_sc8_adaptive")
    out_dir = os.path.join(ROOT, "tmp_sc8_adaptive")
    os.makedirs(out_dir, exist_ok=True)
    for iid in ids:
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in det_map[iid]]), f,
                      ensure_ascii=False)
    return det_map


# ============ 8d: K-means color clustering ============

def kmeans_red_mask(img, k=5):
    """K-means on RGB pixels, identify the red cluster."""
    data = img.reshape(-1, 3).astype(np.float32)
    if len(data) > 50000:
        idx = np.random.choice(len(data), 50000, replace=False)
        sample = data[idx]
    else:
        sample = data
    _, labels, centers = cv2.kmeans(sample, k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.int32)
    red_score = centers[:, 2].astype(float) - centers[:, 1].astype(float) - centers[:, 0].astype(float)
    red_cluster = int(np.argmax(red_score))
    if centers[red_cluster, 2] < 100:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 80, 110), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 110), (180, 255, 255))
    _, full_labels, _ = cv2.kmeans(data, k, None,
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                3, cv2.KMEANS_PP_CENTERS)
    mask = (full_labels.flatten() == red_cluster).astype(np.uint8).reshape(img.shape[:2]) * 255
    return mask


def run_kmeans(ids, p):
    det_map = {}
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        H, W = img.shape[:2]
        mask = kmeans_red_mask(img)
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        outer = []
        if hier is not None:
            hier = hier[0]
            for i, c in enumerate(cnts):
                if hier[i][3] != -1:
                    continue
                if cv2.contourArea(c) >= p["min_area"]:
                    outer.append(c)
        res = {m: [] for m in HY_ORDER}
        for c in outer:
            base = _base_quads(c, p)
            for m in HY_ORDER:
                q = _pass(base[m], W, H, p)
                if q is not None:
                    res[m].append(q)
        for m in HY_ORDER:
            res[m] = dedup_keep_order(res[m], p["dedup_iou"])
        hy = dedup_keep_order(
            [q for m in HY_ORDER for q in res[m]], p["dedup_iou"])
        det_map[iid] = hy
        total += len(hy)
        if (n + 1) % 100 == 0:
            print(f"  [kmeans {n+1}/{len(ids)}] boxes={total}")
    print(f"  kmeans: {total} boxes -> tmp_sc8_kmeans")
    out_dir = os.path.join(ROOT, "tmp_sc8_kmeans")
    os.makedirs(out_dir, exist_ok=True)
    for iid in ids:
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, [np.asarray(q, np.int32) for q in det_map[iid]]), f,
                      ensure_ascii=False)
    return det_map


# ============ driver ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    p = pm5_params()
    ids = get_ids()
    gt_map = {iid: load_boxes(LAB, iid) for iid in ids}
    total_gt = sum(len(v) for v in gt_map.values())
    print(f"LOOP-8 hybrid DL+CV scope on {len(ids)} images, {total_gt} GT, D={DIST_VALUES}")
    print(f"Base = pm_v5 (frozen)\n")

    base_det = {iid: [q.astype(np.int32) for q in load_boxes(FROZEN_PM5, iid)]
                for iid in ids}
    base_results = {}
    for D in DIST_VALUES:
        tp, fp = evaluate(ids, gt_map, base_det, D)
        base_results[str(D)] = {"tp": tp, "raw_fp": fp}
    base_ndet = sum(len(v) for v in base_det.values())
    print(f"{'base(pm5)':14s} {base_ndet:5d}  " + "  ".join(
        f"TP={base_results[str(d)]['tp']:4d}" for d in DIST_VALUES))

    results = {}
    algos_to_run = ["yolo_union", "yolo_crop", "adaptive", "kmeans"]
    if args.only:
        algos_to_run = [a for a in algos_to_run if a == args.only]

    yolo_det = None
    if any(a in algos_to_run for a in ["yolo_union", "yolo_crop"]):
        print("\nRunning YOLOv8-OBB detection...")
        t0 = time.time()
        yolo_det = run_yolo_detect(ids)
        print(f"  YOLO done in {time.time()-t0:.0f}s, "
              f"{sum(len(v) for v in yolo_det.values())} boxes")

    for name in algos_to_run:
        print(f"\n--- 8{ 'abcd'[algos_to_run.index(name)]} {name} ---")
        if name == "yolo_union":
            det_map = run_yolo_union(ids, base_det, yolo_det, p)
        elif name == "yolo_crop":
            det_map = run_yolo_crop(ids, yolo_det, p)
        elif name == "adaptive":
            det_map = run_adaptive(ids, p)
        elif name == "kmeans":
            det_map = run_kmeans(ids, p)

        ndet = sum(len(v) for v in det_map.values())
        row = {}
        for D in DIST_VALUES:
            tp, fp = evaluate(ids, gt_map, det_map, D)
            row[str(D)] = {"tp": tp, "raw_fp": fp}
        results[name] = {"det": ndet, "per_D": row}
        print(f"{name:14s} {ndet:5d}  " + "  ".join(
            f"TP={row[str(d)]['tp']:4d}" for d in DIST_VALUES))
        for D in DIST_VALUES:
            d_tp = row[str(D)]["tp"] - base_results[str(D)]["tp"]
            flag = " ***" if abs(d_tp) >= 5 else ""
            print(f"  D={D:2d}: TP {base_results[str(D)]['tp']}->{row[str(D)]['tp']} ({d_tp:+d}){flag}")

    print(f"\n{'='*70}")
    print(f"{'Algorithm':14s} {'det':>5s}  " + "  ".join(f"D={d:<10d}" for d in DIST_VALUES))
    print(f"{'base(pm5)':14s} {base_ndet:5d}  " + "  ".join(
        f"TP={base_results[str(d)]['tp']:4d}     " for d in DIST_VALUES))
    for name in results:
        r = results[name]
        print(f"{name:14s} {r['det']:5d}  " + "  ".join(
            f"TP={r['per_D'][str(d)]['tp']:4d}     " for d in DIST_VALUES))

    out = os.path.join(ROOT, "benchmark", "cv_scope8.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(ids), "total_gt": total_gt,
                   "dist_values": DIST_VALUES,
                   "base": {"det": base_ndet, "per_D": base_results},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()