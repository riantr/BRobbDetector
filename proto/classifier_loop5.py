"""LOOP-5: CNN box classifier.

Train a CNN to classify detected boxes as TP (keep) or FP (remove).
Crops each detection from the original image, trains a small CNN,
then filters detections by predicted probability.

GATE (raw):       raw_fp - >= 10 AND recall - >= -0.3
GATE (corrected): true_fp - >= 5 AND recall - >= -0.3
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from common import load_expected, IMG, LAB, red_mask, iou_obb

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
M8 = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
IOU_THRESHOLD = 0.3
SUP_THR = 0.30
MIN_SIDE = 12.0
MAX_ASPECT = 8.0
CROP_SIZE = 64
PAD_FRAC = 0.15


# ── I/O ──

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


# ── Crop extraction ──

def extract_crop(img, box, size=CROP_SIZE, pad_frac=PAD_FRAC):
    xs, ys = box[:, 0], box[:, 1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(img.shape[1], x2 + px); y2 = min(img.shape[0], y2 + py)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size))


# ── CNN ──

class BoxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ── Data collection ──

def collect_data(ids):
    crops, labels, meta = [], [], []
    n_tp, n_fp = 0, 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        used_gt = set()
        for di, db in enumerate(det):
            best, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if j in used_gt:
                    continue
                v = iou_obb(db, gb)
                if v > best:
                    best, bj = v, j
            is_tp = bj >= 0 and best >= IOU_THRESHOLD
            if is_tp:
                used_gt.add(bj)
            crop = extract_crop(img, db)
            crops.append(crop)
            labels.append(1 if is_tp else 0)
            meta.append((iid, di))
            if is_tp:
                n_tp += 1
            else:
                n_fp += 1
    return crops, labels, meta, n_tp, n_fp


# ── Train ──

def train_cnn(crops, labels, epochs=20, lr=0.001, batch_size=32):
    X = np.array(crops, dtype=np.float32) / 255.0
    X = X.transpose(0, 3, 1, 2)  # NCHW
    y = np.array(labels, dtype=np.float32)
    X_t = torch.from_numpy(X); y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BoxCNN()
    pos_weight = torch.tensor([len(y[y == 0]) / max(len(y[y == 1]), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in dl:
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={total_loss/len(dl):.4f}")
    return model


# ── Predict ──

def predict(model, crops, batch_size=64):
    model.eval()
    X = np.array(crops, dtype=np.float32) / 255.0
    X = X.transpose(0, 3, 1, 2)
    X_t = torch.from_numpy(X)
    with torch.no_grad():
        out = model(X_t).squeeze(1)
        probs = torch.sigmoid(out).numpy()
    return probs


# ── Corrected metric ──

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


# ── Evaluate ──

def evaluate(ids, probs, meta, threshold=0.5, sup_gate=0.0, verbose=False):
    """Filter detections. If sup_gate > 0, only remove boxes where
    CNN prob < threshold AND red_support < sup_gate."""
    prob_map = {}
    for (iid, di), p in zip(meta, probs):
        prob_map.setdefault(iid, {})[di] = p

    exp = load_expected()
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    iou_sum = 0.0
    n_removed = 0

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        pm = prob_map.get(iid, {})

        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)

        filtered = []
        for di, db in enumerate(det):
            p = pm.get(di, 1.0)
            if p < threshold:
                if sup_gate > 0:
                    sup = red_support(db, mask_d)
                    if sup < sup_gate:
                        n_removed += 1
                        continue
                else:
                    n_removed += 1
                    continue
            filtered.append(db)
        csv_n = exp.get(iid, 0)
        tp, unmatched, iou_s = match_all(filtered, gt)
        TP += tp; FN += len(gt) - tp; iou_sum += iou_s
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(filtered[di], csv_n, len(gt), mask_d):
                real_unl += 1
            else:
                true_fp += 1
        if verbose and (n + 1) % 100 == 0:
            print(f"  [{n+1}/500] TP={TP} rawFP={raw_fp} trueFP={true_fp} removed={n_removed}")

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
        "removed": n_removed, "threshold": threshold,
    }


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sup-gate", type=float, default=0.0, help="only remove if red support < this")
    ap.add_argument("--cv", type=int, default=0, help="k-fold cross-validation (0 = no CV)")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    ids = get_ids()
    print(f"Images: {len(ids)}")

    print("\n=== Collecting data ===")
    crops, labels, meta, n_tp, n_fp = collect_data(ids)
    print(f"  Total: {len(crops)}  TP: {n_tp}  FP: {n_fp}")

    if args.cv > 0:
        k = args.cv
        print(f"\n=== {k}-fold cross-validation ({args.epochs} epochs) ===")
        img_idx = {iid: i for i, iid in enumerate(ids)}
        fold_size = len(ids) // k
        all_probs = np.zeros(len(crops))
        for fold in range(k):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k - 1 else len(ids)
            test_ids = set(ids[test_start:test_end])
            train_mask = np.array([img_idx[m[0]] not in range(test_start, test_end) for m in meta])
            test_mask = ~train_mask
            train_crops = [crops[i] for i in range(len(crops)) if train_mask[i]]
            train_labels = [labels[i] for i in range(len(crops)) if train_mask[i]]
            print(f"  Fold {fold+1}/{k}: train={sum(train_mask)} test={sum(test_mask)}")
            model = train_cnn(train_crops, train_labels, epochs=args.epochs)
            test_crops = [crops[i] for i in range(len(crops)) if test_mask[i]]
            test_probs = predict(model, test_crops)
            for i, p in zip(np.where(test_mask)[0], test_probs):
                all_probs[i] = p
        probs = all_probs
    else:
        print(f"\n=== Training CNN ({args.epochs} epochs) ===")
        model = train_cnn(crops, labels, epochs=args.epochs)
        print("\n=== Predicting ===")
        probs = predict(model, crops)

    pred_tp = np.sum(probs >= 0.5)
    pred_fp = np.sum(probs < 0.5)
    print(f"  Predicted TP: {pred_tp}  Predicted FP: {pred_fp}")

    for thr in [0.05, 0.1, 0.15, 0.2]:
        print(f"\n=== Evaluating (threshold={thr}, sup_gate={args.sup_gate}) ===")
        r = evaluate(ids, probs, meta, threshold=thr, sup_gate=args.sup_gate, verbose=True)
        print(json.dumps(r, indent=2))
        print(f"\n  Baseline:  TP=677  FP=238  FN=69  Recall=90.8  raw_F1=81.5  corrected_F1=94.6")
        print(f"  Filtered:  TP={r['tp']}  FP={r['raw_fp']}  FN={r['fn']}  Recall={r['recall']}  raw_F1={r['raw_f1']}  corrected_F1={r['corrected_f1']}  removed={r['removed']}")

        gate_raw = (238 - r["raw_fp"]) >= 10 and r["recall"] >= 90.8 - 0.3
        gate_corr = (8 - r["true_fp"]) >= 5 and r["recall"] >= 90.8 - 0.3
        print(f"  GATE (raw):       FP-≥10 AND Recall≥90.5 → {'PASS' if gate_raw else 'FAIL'}")
        print(f"  GATE (corrected): tFP-≥5 AND Recall≥90.5 → {'PASS' if gate_corr else 'FAIL'}")

        if thr == args.threshold and args.save:
            out = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", f"{args.save}_t{thr}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            print(f"  saved: {out}")
            model_path = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", f"{args.save}_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  model: {model_path}")


if __name__ == "__main__":
    main()