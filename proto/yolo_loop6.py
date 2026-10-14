"""LOOP-6: YOLOv8-OBB end-to-end detector.

Train YOLOv8n-OBB on the 500-image CARS dataset with corrected GT
(GT + unlabeled-real detections). Evaluate with corrected metric.

GATE (raw): F1 >= 81.5
"""
import sys, os, json, argparse, shutil
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
M8 = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
IOU_THRESHOLD = 0.3
SUP_THR = 0.30
MIN_SIDE = 12.0
MAX_ASPECT = 8.0
YOLO_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_yolo"


def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_boxes(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


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
    tp, unmatched = 0, []
    iou_sum = 0.0
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


# ── Dataset prep ──

def prepare_dataset(ids, val_frac=0.2):
    exp = load_expected()
    n_val = int(len(ids) * val_frac)
    train_ids = ids[n_val:]
    val_ids = ids[:n_val]

    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

    n_boxes = 0
    for split, split_ids in [("train", train_ids), ("val", val_ids)]:
        for iid in split_ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            H, W = img.shape[:2]

            # Copy image
            src = os.path.join(IMG, iid + ".jpg")
            dst = os.path.join(YOLO_DIR, "images", split, iid + ".jpg")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            # Build corrected GT: GT + unlabeled-real detections
            gt = load_boxes(LAB, iid)
            det = load_boxes(OUT, iid)
            mask = red_mask(img)
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
            csv_n = exp.get(iid, 0)

            all_boxes = list(gt)
            tp, unmatched = match_all(det, gt)
            for di in unmatched:
                if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                    all_boxes.append(det[di])

            # Write YOLO OBB label
            label_path = os.path.join(YOLO_DIR, "labels", split, iid + ".txt")
            with open(label_path, "w") as f:
                for box in all_boxes:
                    pts = box.astype(float)
                    pts[:, 0] /= W
                    pts[:, 1] /= H
                    coords = " ".join(f"{p:.6f}" for p in pts.flatten())
                    f.write(f"0 {coords}\n")
                    n_boxes += 1

    # Write data.yaml
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: box\n")

    print(f"Dataset: {len(train_ids)} train, {len(val_ids)} val, {n_boxes} boxes")
    return yaml_path


# ── Train ──

def train_yolo(yaml_path, epochs=10, imgsz=640, variant="n"):
    from ultralytics import YOLO
    model = YOLO(f"yolov8{variant}-obb.pt")
    results = model.train(data=yaml_path, epochs=epochs, imgsz=imgsz,
                          device="cpu", batch=8, project=YOLO_DIR, name="train",
                          exist_ok=True, verbose=True)
    return model


# ── Predict + Evaluate ──

def predict_and_eval(ids, model_path, conf=0.25, verbose=False):
    from ultralytics import YOLO
    model = YOLO(model_path)
    exp = load_expected()

    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    iou_sum = 0.0

    for n, iid in enumerate(ids):
        img_path = os.path.join(IMG, iid + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes(LAB, iid)

        results = model.predict(img_path, verbose=False, conf=conf)
        det = []
        if results and results[0].obb is not None:
            for obb in results[0].obb:
                pts = obb.cpu().numpy()
                if hasattr(pts, 'xyxyxyxy'):
                    corners = pts.xyxyxyxy[0].astype(np.int32)
                else:
                    corners = np.array(pts, np.int32)
                if len(corners) == 4:
                    det.append(corners)

        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)
        if det:
            tp, unmatched, iou_s = match_all(det, gt)
        else:
            tp, unmatched, iou_s = 0, [], 0.0
        TP += tp; FN += len(gt) - tp; iou_sum += iou_s
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
        if verbose and (n + 1) % 100 == 0:
            print(f"  [{n+1}/500] TP={TP} rawFP={raw_fp} trueFP={true_fp}")

    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec_raw = TP / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
    f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0

    return {
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp, "real_unlabeled": real_unl,
        "recall": round(recall, 1), "raw_f1": round(f1_raw, 1),
        "corrected_f1": round(f1_c, 1), "avg_iou": round(iou_sum / max(TP, 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--variant", default="n", choices=["n", "s", "m"])
    args = ap.parse_args()

    ids = get_ids()
    print(f"Images: {len(ids)}")

    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    model_path = os.path.join(YOLO_DIR, "train", "weights", "best.pt")

    if args.prep or args.full:
        print("\n=== Preparing dataset ===")
        yaml_path = prepare_dataset(ids)

    if args.train or args.full:
        print(f"\n=== Training YOLOv8n-OBB ({args.epochs} epochs, imgsz={args.imgsz}) ===")
        train_yolo(yaml_path, epochs=args.epochs, imgsz=args.imgsz, variant=args.variant)

    if args.eval or args.full:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
        print(f"\n=== Evaluating ===")
        r = predict_and_eval(ids, model_path, conf=args.conf, verbose=True)
        print("\n" + "=" * 60)
        print(json.dumps(r, indent=2))
        print("=" * 60)
        print(f"\n  Baseline:  TP=677  FP=238  FN=69  Recall=90.8  raw_F1=81.5  corrected_F1=94.6")
        print(f"  YOLOv8:    TP={r['tp']}  FP={r['raw_fp']}  FN={r['fn']}  Recall={r['recall']}  raw_F1={r['raw_f1']}  corrected_F1={r['corrected_f1']}")
        gate = r["raw_f1"] >= 81.5
        print(f"\n  GATE: F1≥81.5 → {'PASS' if gate else 'FAIL'}")
        if args.save:
            out = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", args.save + ".json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            print(f"  saved: {out}")


if __name__ == "__main__":
    main()