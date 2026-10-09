import os, sys, subprocess, json, shutil, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_expected, IMG, LAB, iou_obb

CSPROJ = r"D:\src\CodeArts\BRobbDetector\BRobbDetector\BRobbDetector.csproj"
TMP_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_cars_subset"
OUT_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
BENCH_DIR = r"D:\src\CodeArts\BRobbDetector\benchmark"

IOU_THRESHOLD = 0.3


def parse_points(s):
    pts = []
    for p in s.split(';'):
        x, y = p.split(',')
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)


def load_gt_boxes(image_id):
    path = os.path.join(LAB, image_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]


def load_det_boxes(image_id, out_dir):
    path = os.path.join(out_dir, image_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]


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
    return tp, fp, fn, iou_sum, max(tp, 1)


def run_detector(n, method, flt, cs, cm, purify="none"):
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:n]

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
    for k in ids:
        src = os.path.join(IMG, k + ".jpg")
        dst = os.path.join(TMP_DIR, k + ".jpg")
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    print(f"Running C# detector on {n} images (method={method}, filter={flt}, cs={cs}, cm={cm}, purify={purify})...")
    t0 = time.time()
    cmd = ["dotnet", "run", "--project", CSPROJ, "--", "--cli", TMP_DIR, OUT_DIR,
         "--method", method, "--filter", flt, "--colorspace", cs, "--closemode", cm]
    if purify != "none":
        cmd += ["--purify", purify]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return None
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s")
    return ids, elapsed


def compute_metrics(ids, out_dir=OUT_DIR):
    exact = over = under = 0
    total_exp_csv = 0
    TP = FP = FN = 0
    iou_sum = 0.0
    matched_count = 0
    per_image = []

    for k in ids:
        exp = load_expected()
        csv_expected = exp[k]
        gt_boxes = load_gt_boxes(k)
        det_boxes = load_det_boxes(k, out_dir)

        d = len(det_boxes) - csv_expected
        if d == 0: exact += 1
        elif d > 0: over += 1
        else: under += 1
        total_exp_csv += csv_expected

        tp, fp, fn, iou_s, _ = match_boxes(det_boxes, gt_boxes)
        TP += tp; FP += fp; FN += fn
        iou_sum += iou_s
        matched_count += tp
        per_image.append({"id": k, "gt": len(gt_boxes), "det": len(det_boxes),
                          "csv": csv_expected, "tp": tp, "fp": fp, "fn": fn})

    n = len(ids)
    total_gt = TP + FN
    total_det = TP + FP
    recall = TP / total_gt if total_gt else 0
    precision = TP / total_det if total_det else 0
    fpr = FP / total_det if total_det else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    avg_iou = iou_sum / matched_count if matched_count else 0

    return {
        "n_images": n,
        "image_level": {
            "exact": exact, "exact_pct": round(exact / n * 100, 1),
            "over": over, "under": under,
        },
        "box_level": {
            "tp": TP, "fp": FP, "fn": FN,
            "total_gt": total_gt, "total_det": total_det,
            "total_csv_expected": total_exp_csv,
            "recall": round(recall * 100, 1),
            "precision": round(precision * 100, 1),
            "fpr": round(fpr * 100, 1),
            "f1": round(f1 * 100, 1),
            "avg_iou": round(avg_iou, 3),
            "iou_threshold": IOU_THRESHOLD,
        },
        "per_image": per_image,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark BRobbDetector on CARS dataset")
    ap.add_argument("n", type=int, nargs="?", default=500, help="number of images")
    ap.add_argument("--method", default="hole")
    ap.add_argument("--filter", default="none")
    ap.add_argument("--colorspace", default="hsvLabRgb")
    ap.add_argument("--closemode", default="iter1")
    ap.add_argument("--purify", default="none", help="color purification mode: none|lab")
    ap.add_argument("--save", default=None, help="save benchmark JSON to benchmark/<name>.json")
    ap.add_argument("--compare", default=None, help="compare against benchmark/<name>.json")
    ap.add_argument("--no-run", action="store_true", help="skip detection, reuse existing output")
    args = ap.parse_args()

    if not args.no_run:
        result = run_detector(args.n, args.method, args.filter, args.colorspace, args.closemode, args.purify)
        if result is None:
            return
        ids, elapsed = result
    else:
        exp = load_expected()
        ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
        ids.sort()
        rng = np.random.RandomState(0)
        rng.shuffle(ids)
        ids = ids[:args.n]
        elapsed = 0

    metrics = compute_metrics(ids)
    if not args.no_run:
        metrics["config"] = {
            "method": args.method, "filter": args.filter,
            "colorspace": args.colorspace, "closemode": args.closemode,
            "purify": args.purify,
        }
        metrics["elapsed_seconds"] = round(elapsed, 1)

    il = metrics["image_level"]
    bl = metrics["box_level"]
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({metrics['n_images']} images)")
    print(f"{'='*60}")
    print(f"Config: method={args.method} filter={args.filter} cs={args.colorspace} cm={args.closemode}")
    print(f"\n--- Image-level (count match vs CSV expected) ---")
    print(f"  Exact: {il['exact_pct']}%  ({il['exact']}/{metrics['n_images']})")
    print(f"  Over:  {il['over']}  Under: {il['under']}")
    print(f"\n--- Box-level (IoU >= {IOU_THRESHOLD} vs GT label files) ---")
    print(f"  TP={bl['tp']}  FP={bl['fp']}  FN={bl['fn']}")
    print(f"  Total GT={bl['total_gt']}  Total Det={bl['total_det']}  (CSV expected={bl['total_csv_expected']})")
    print(f"  Recall (检出率):     {bl['recall']}%")
    print(f"  Precision (精确率):  {bl['precision']}%")
    print(f"  FPR (误检率):        {bl['fpr']}%")
    print(f"  F1:                  {bl['f1']}%")
    print(f"  Avg IoU (matched):   {bl['avg_iou']}")
    print(f"{'='*60}")

    if args.save:
        os.makedirs(BENCH_DIR, exist_ok=True)
        path = os.path.join(BENCH_DIR, args.save + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {path}")

    if args.compare:
        path = os.path.join(BENCH_DIR, args.compare + ".json")
        if not os.path.exists(path):
            print(f"\nBaseline {path} not found, skipping comparison.")
        else:
            with open(path, encoding="utf-8") as f:
                base = json.load(f)
            bl_base = base["box_level"]
            il_base = base["image_level"]
            print(f"\n--- Comparison vs baseline '{args.compare}' ---")
            print(f"  Exact:   {il_base['exact_pct']}% -> {il['exact_pct']}%  ({il['exact_pct'] - il_base['exact_pct']:+.1f})")
            print(f"  Recall:  {bl_base['recall']}% -> {bl['recall']}%  ({bl['recall'] - bl_base['recall']:+.1f})")
            print(f"  FPR:     {bl_base['fpr']}% -> {bl['fpr']}%  ({bl['fpr'] - bl_base['fpr']:+.1f})")
            print(f"  F1:      {bl_base['f1']}% -> {bl['f1']}%  ({bl['f1'] - bl_base['f1']:+.1f})")
            print(f"  AvgIoU:  {bl_base['avg_iou']} -> {bl['avg_iou']}  ({bl['avg_iou'] - bl_base['avg_iou']:+.3f})")

    if not args.no_run:
        for k in ids:
            p = os.path.join(TMP_DIR, k + ".jpg")
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()