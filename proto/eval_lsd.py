"""Evaluate LSD rectangle assembly merged with hole detections on all 500 CARS images."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb
from lsd_rect import detect_rects, parse_points, load_boxes, DEFAULT

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
BENCH = r"D:\src\CodeArts\BRobbDetector\benchmark"
M8 = os.path.join(BENCH, "m8_growbox_v3_final.json")
IOU_THRESHOLD = 0.3


def match_boxes(det_boxes, gt_boxes, iou_thr=IOU_THRESHOLD):
    used = set()
    tp = 0
    iou_sum = 0.0
    for db in det_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in used:
                continue
            iou = iou_obb(db, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            tp += 1
            iou_sum += best_iou
    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, iou_sum


def merge(hole, lsd, add_iou=0.3):
    """Keep hole dets, add LSD rects that don't overlap existing ones."""
    merged = [h.astype(np.int32) for h in hole]
    added = 0
    for r in sorted(lsd, key=lambda x: -cv2.contourArea(x.astype(np.float32))):
        ri = r.astype(np.int32)
        if all(iou_obb(ri, m) < add_iou for m in merged):
            merged.append(ri)
            added += 1
    return merged, added


def evaluate(params, sids=None, verbose=False):
    with open(M8, encoding="utf-8") as f:
        bench = json.load(f)
    per = {p["id"]: p for p in bench["per_image"]}

    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:500]
    if sids is not None:
        ids = sids

    TP = FP = FN = 0
    iou_sum = 0.0
    matched = 0
    total_added = 0
    t0 = time.time()
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = red_mask(img)
        lsd = detect_rects(mask, W, H, params)
        hole = load_boxes(OUT, iid)
        gt = load_boxes(LAB, iid)
        merged, added = merge(hole, lsd)
        total_added += added
        tp, fp, fn, iou_s = match_boxes(merged, gt)
        TP += tp; FP += fp; FN += fn; iou_sum += iou_s; matched += tp
        if verbose and (n % 100 == 0):
            print(f"  [{n}/500] added={total_added} TP={TP} FP={FP} FN={FN}")

    total_gt = TP + FN
    total_det = TP + FP
    recall = TP / total_gt if total_gt else 0
    precision = TP / total_det if total_det else 0
    fpr = FP / total_det if total_det else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    avg_iou = iou_sum / matched if matched else 0
    elapsed = time.time() - t0

    return {
        "params": params, "n_images": len(ids),
        "tp": TP, "fp": FP, "fn": FN,
        "recall": round(recall * 100, 1),
        "fpr": round(fpr * 100, 1),
        "f1": round(f1 * 100, 1),
        "avg_iou": round(avg_iou, 3),
        "lsd_added": total_added, "elapsed": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle-tol", type=float, default=DEFAULT["angle_tol"])
    ap.add_argument("--min-len", type=float, default=DEFAULT["min_len"])
    ap.add_argument("--min-side", type=float, default=DEFAULT["min_side"])
    ap.add_argument("--max-side", type=float, default=DEFAULT["max_side"])
    ap.add_argument("--overlap-frac", type=float, default=DEFAULT["min_overlap_frac"])
    ap.add_argument("--edge-sup", type=float, default=DEFAULT["edge_sup_thr"])
    ap.add_argument("--min-edges", type=int, default=DEFAULT["min_edges"])
    ap.add_argument("--cover", type=float, default=DEFAULT["cover_thr"])
    ap.add_argument("--add-iou", type=float, default=0.3)
    ap.add_argument("--aspect-max", type=float, default=1e9)
    ap.add_argument("--side-min", type=float, default=0.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    params = {
        "angle_tol": args.angle_tol, "min_len": args.min_len,
        "min_side": args.min_side, "max_side": args.max_side,
        "min_overlap_frac": args.overlap_frac, "edge_sup_thr": args.edge_sup,
        "min_edges": args.min_edges, "cover_thr": args.cover,
        "post_aspect_max": args.aspect_max, "post_side_min": args.side_min,
    }
    r = evaluate(params, verbose=True)
    print("\n" + "=" * 60)
    print(f"LSD MERGED RESULTS: {r}")
    print(f"Baseline m8: TP=677 FP=238 FN=69 Recall=90.8 FPR=26.0 F1=81.5")
    print(f"Delta: recall {r['recall']-90.8:+.1f}  fpr {r['fpr']-26.0:+.1f}  "
          f"f1 {r['f1']-81.5:+.1f}  (LSD added {r['lsd_added']} boxes)")
    print("=" * 60)
    if args.save:
        with open(os.path.join(BENCH, args.save + ".json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()