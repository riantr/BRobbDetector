"""LOOP-4: Multi-scale detection.

Run the C# detection pipeline at 0.5x, 1x, and 2x scales, merge results
with NMS, and evaluate with the corrected metric.

GATE (raw):       raw_f1 >= 81.7
GATE (corrected): corrected_f1 >= 94.6 AND true_fp <= 13

Usage:
  python proto/multiscale_loop4.py --prep          # resize images
  python proto/multiscale_loop4.py --eval          # merge + evaluate
  python proto/multiscale_loop4.py --full          # prep + run C# + eval
"""
import sys, os, json, argparse, subprocess, shutil
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
M8 = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
IOU_THRESHOLD = 0.3
SUP_THR = 0.30
MIN_SIDE = 12.0
MAX_ASPECT = 8.0
SCALES = [0.5, 2.0]

TMP = r"D:\src\CodeArts\BRobbDetector\tmp_multiscale"
CSPROJ = r"D:\src\CodeArts\BRobbDetector\BRobbDetector\BRobbDetector.csproj"


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_boxes(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def save_boxes(d, iid, boxes, img_name):
    os.makedirs(d, exist_ok=True)
    infos = [{"points": ";".join(f"{int(p[0])},{int(p[1])}" for p in b), "type": ""} for b in boxes]
    with open(os.path.join(d, iid + ".json"), "w", encoding="utf-8") as f:
        json.dump({"img_name": img_name, "infos": infos}, f, ensure_ascii=False)


# ── Prep: resize images ──

def prep(ids):
    for s in SCALES:
        in_dir = os.path.join(TMP, f"in_{s:.1f}")
        out_dir = os.path.join(TMP, f"out_{s:.1f}")
        os.makedirs(in_dir, exist_ok=True)
        print(f"Resizing {len(ids)} images to {s}x → {in_dir}")
        for n, iid in enumerate(ids):
            src = os.path.join(IMG, iid + ".jpg")
            img = cv2.imread(src)
            if img is None:
                continue
            h, w = img.shape[:2]
            nw, nh = int(w * s), int(h * s)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(in_dir, iid + ".jpg"), resized)
            if (n + 1) % 100 == 0:
                print(f"  [{n+1}/{len(ids)}]")
    print("Prep done.")


# ── Run C# detection ──

def run_csharp():
    for s in SCALES:
        in_dir = os.path.join(TMP, f"in_{s:.1f}")
        out_dir = os.path.join(TMP, f"out_{s:.1f}")
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            print(f"  {s}x output exists, skipping")
            continue
        print(f"Running C# detection at {s}x → {out_dir}")
        cmd = ["dotnet", "run", "--project", CSPROJ, "-c", "Debug", "--",
               "--cli", in_dir, out_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(result.stdout[-200:] if result.stdout else "")
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-300:]}")
            return False
    return True


# ── Merge + Evaluate ──

def scale_box(box, s):
    """Scale box from s× image back to 1× coordinates."""
    return np.int32(box / s)


def nms_merge(boxes, thr=0.4):
    """NMS on list of 4-corner int arrays."""
    sorted_boxes = sorted(boxes, key=lambda b: cv2.contourArea(b.astype(np.float32)), reverse=True)
    keep = []
    for b in sorted_boxes:
        if all(iou_obb(b, k) < thr for k in keep):
            keep.append(b)
    return keep


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
    if csv_n <= n_gt:
        return False
    sides = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
    mn, mx = min(sides), max(sides)
    if mn < MIN_SIDE or (mx / max(mn, 1e-6)) > MAX_ASPECT:
        return False
    return red_support(box, mask_d) >= SUP_THR


def match_all(det, gt, thr=IOU_THRESHOLD):
    used = set()
    tp, unmatched, iou_sum = 0, [], 0.0
    for di, db in enumerate(det):
        best, bj = 0.0, -1
        for j, gb in enumerate(gt):
            if j in used:
                continue
            v = iou_obb(db, gb)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= thr:
            used.add(bj); tp += 1; iou_sum += best
        else:
            unmatched.append(di)
    return tp, unmatched, iou_sum


def evaluate(ids, nms_thr=0.4, novel_only=False, verbose=False):
    exp = load_expected()
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    iou_sum = 0.0
    n_added_05 = n_added_20 = 0

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes(LAB, iid)

        # 1x detections (from tmp_cars_out)
        det_1x = load_boxes(OUT, iid)

        # Multi-scale detections
        all_boxes = list(det_1x)
        for s in SCALES:
            sd = os.path.join(TMP, f"out_{s:.1f}")
            scaled = load_boxes(sd, iid)
            back = [scale_box(b, s) for b in scaled]
            if novel_only:
                novel = [b for b in back
                         if all(iou_obb(b, m) < 0.1 for m in det_1x)]
                all_boxes.extend(novel)
                if s < 1.0:
                    n_added_05 += len(novel)
                else:
                    n_added_20 += len(novel)
            else:
                before = len(all_boxes)
                all_boxes.extend(back)
                after = len(all_boxes)
                if s < 1.0:
                    n_added_05 += after - before
                else:
                    n_added_20 += after - before

        # NMS merge
        merged = nms_merge(all_boxes, thr=nms_thr)

        # Match
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)
        tp, unmatched, iou_s = match_all(merged, gt)
        TP += tp; FN += len(gt) - tp; iou_sum += iou_s
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(merged[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
        if verbose and (n + 1) % 100 == 0:
            print(f"  [{n+1}/500] TP={TP} rawFP={raw_fp} trueFP={true_fp} realUnl={real_unl}")

    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec_raw = TP / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
    f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0

    return {
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp, "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "raw_f1": round(f1_raw, 1),
        "corrected_f1": round(f1_c, 1),
        "avg_iou": round(iou_sum / max(TP, 1), 3),
        "added_0.5x": n_added_05, "added_2.0x": n_added_20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--nms-thr", type=float, default=0.4)
    ap.add_argument("--novel-only", action="store_true", help="only add multi-scale boxes with IoU<0.1 to any 1x box")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    ids = get_ids()
    print(f"Images: {len(ids)}")

    if args.prep or args.full:
        prep(ids)

    if args.full:
        if not run_csharp():
            print("C# detection failed, aborting.")
            return

    if args.eval or args.full:
        print("\n=== Evaluating multi-scale merge ===")
        r = evaluate(ids, nms_thr=args.nms_thr, novel_only=args.novel_only, verbose=True)
        print("\n" + "=" * 60)
        print(json.dumps(r, indent=2))
        print("=" * 60)

        print("\n=== Comparison with baseline ===")
        print(f"  Baseline:       TP=677  FP=238  FN=69   Recall=90.8  raw_F1=81.5  corrected_F1=94.6")
        print(f"  LOOP-3 refine:  TP=694  FP=221  FN=52   Recall=93.0  raw_F1=83.6  corrected_F1=95.9")
        print(f"  Multi-scale:    TP={r['tp']}  FP={r['raw_fp']}  FN={r['fn']}   Recall={r['recall']}  raw_F1={r['raw_f1']}  corrected_F1={r['corrected_f1']}")
        print(f"  Added: 0.5x={r['added_0.5x']}  2.0x={r['added_2.0x']}")

        gate_raw = r["raw_f1"] >= 81.7
        gate_corr = r["corrected_f1"] >= 94.6 and r["true_fp"] <= 13
        print(f"\n  GATE (raw):       F1≥81.7 → {'PASS' if gate_raw else 'FAIL'}")
        print(f"  GATE (corrected): cF1≥94.6 AND tFP≤13 → {'PASS' if gate_corr else 'FAIL'}")

        if args.save:
            out = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", args.save + ".json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            print(f"  saved: {out}")


if __name__ == "__main__":
    main()