"""GT-corrected evaluation harness (LOOP-1 revision).

The frozen GT label files are ~25% incomplete: many detected boxes are real
annotation boxes that were never labeled. The prior audit found 347/354 "FP"
were such unlabeled real boxes (criterion: image GT-incomplete AND red edge
support >= 0.3). This harness reports both:

  raw        - against the labeled GT files (current benchmark)
  corrected  - FP excludes unlabeled-real boxes, so FPR reflects the true
               spurious-detection rate.

A detection that fails to match GT counts as a "true FP" only if it is NOT an
unlabeled-real box.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb
from lsd_rect import detect_rects, DEFAULT

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
M8 = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
IOU_THRESHOLD = 0.3
SUP_THR = 0.30
MIN_SIDE = 12.0
MAX_ASPECT = 8.0


def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_boxes(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def match_boxes(det, gt, thr=IOU_THRESHOLD):
    """Return (tp, unmatched_det_indices, iou_sum)."""
    used = set()
    tp = 0
    iou_sum = 0.0
    unmatched = []
    for di, db in enumerate(det):
        best = 0.0; bj = -1; bmask = None
        for j, gb in enumerate(gt):
            if j in used:
                continue
            v = iou_obb(db, gb)
            if v > best:
                best = v; bj = j
        if bj >= 0 and best >= thr:
            used.add(bj); tp += 1; iou_sum += best
        else:
            unmatched.append(di)
    return tp, unmatched, iou_sum


def red_support(box, mask_d):
    sup = []
    for i in range(4):
        p, q = box[i].astype(float), box[(i + 1) % 4].astype(float)
        L = np.linalg.norm(q - p)
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = p[None, :] + ts[:, None] * (q - p)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, mask_d.shape[1] - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, mask_d.shape[0] - 1)
        sup.append(np.count_nonzero(mask_d[ys, xs]) / ns)
    return float(np.mean(sup))


def is_real_unlabeled(box, csv_n, n_gt, mask_d):
    """Unlabeled-real heuristic: image GT-incomplete + box-like + red support."""
    if csv_n <= n_gt:
        return False
    sides = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
    mn, mx = min(sides), max(sides)
    if mn < MIN_SIDE or (mx / max(mn, 1e-6)) > MAX_ASPECT:
        return False
    return red_support(box, mask_d) >= SUP_THR


def evaluate(mode="m8", lsd_params=None, verbose=False):
    with open(M8, encoding="utf-8") as f:
        base_json = json.load(f)
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]

    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    iou_sum = 0.0
    p = dict(DEFAULT)
    if lsd_params:
        p.update(lsd_params)

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes(LAB, iid)
        hole = load_boxes(OUT, iid)
        if mode == "lsd":
            mask = red_mask(img)
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
            lsd = detect_rects(mask, W, H, p)
            det = [h.astype(np.int32) for h in hole]
            for r in sorted(lsd, key=lambda x: -cv2.contourArea(x.astype(np.float32))):
                ri = r.astype(np.int32)
                if all(iou_obb(ri, m) < 0.3 for m in det):
                    det.append(ri)
        else:
            mask = red_mask(img)
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
            det = [h.astype(np.int32) for h in hole]

        csv_n = exp.get(iid, 0)
        tp, unmatched, iou_s = match_boxes(det, gt)
        TP += tp; FN += len(gt) - tp; iou_sum += iou_s
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
        if verbose and n % 100 == 0:
            print(f"  [{n}/500] TP={TP} rawFP={raw_fp} trueFP={true_fp} realUnl={real_unl}")

    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec_raw = TP / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
    fpr_raw = raw_fp / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
    f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    fpr_c = true_fp / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0

    return {
        "mode": mode,
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp, "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "raw_fpr": round(fpr_raw, 1), "raw_f1": round(f1_raw, 1),
        "corrected_precision": round(prec_c, 1), "corrected_fpr": round(fpr_c, 1),
        "corrected_f1": round(f1_c, 1),
        "avg_iou": round(iou_sum / max(TP, 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="m8", choices=["m8", "lsd"])
    ap.add_argument("--cover", type=float, default=0.55)
    ap.add_argument("--edge-sup", type=float, default=0.30)
    ap.add_argument("--min-edges", type=int, default=3)
    ap.add_argument("--min-len", type=float, default=12.0)
    ap.add_argument("--min-side", type=float, default=12.0)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--aspect-max", type=float, default=1e9)
    ap.add_argument("--side-min2", type=float, default=0.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    p = {
        "cover_thr": args.cover, "edge_sup_thr": args.edge_sup, "min_edges": args.min_edges,
        "min_len": args.min_len, "min_side": args.min_side, "angle_tol": args.angle_tol,
        "post_aspect_max": args.aspect_max, "post_side_min": args.side_min2,
    }
    r = evaluate(mode=args.mode, lsd_params=p, verbose=True)
    print("\n" + "=" * 60)
    print(json.dumps(r, indent=2))
    print("=" * 60)
    if args.save:
        out = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", args.save + ".json")
        r["lsd_params"] = p if args.mode == "lsd" else None
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()