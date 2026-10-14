"""LOOP-3: Learned box refinement.

Train an MLP that predicts (dx, dy, log_dw, log_dh, dtheta) correction
for each detected box, moving it closer to GT. Targets near-miss
detections (IoU 0.1-0.3) to push them above the 0.3 IoU threshold.

GATE (corrected): corrected_f1 >= 94.6 AND true_fp +<= 5
GATE (raw):       raw_f1 >= 82.0 AND raw_fp +<= 5
"""
import sys, os, json, argparse, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from common import load_expected, IMG, LAB, red_mask, iou_obb

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
M8 = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
IOU_THRESHOLD = 0.3
SUP_THR = 0.30
MIN_SIDE = 12.0
MAX_ASPECT = 8.0

CLIP = np.array([0.3, 0.3, 0.5, 0.5, 0.33])  # max |delta| per component


# ── Geometry ──

def corners_to_rect(corners):
    """4 corners → (cx, cy, w, h, angle) with w >= h, angle in [-90, 90)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(corners.astype(np.float32))
    if w < h:
        w, h = h, w
        ang += 90.0
    if ang >= 90.0:
        ang -= 180.0
    if ang < -90.0:
        ang += 180.0
    return float(cx), float(cy), float(w), float(h), float(ang)


def rect_to_corners(cx, cy, w, h, ang):
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    return np.int32(box)


def compute_delta(det_rect, gt_rect):
    dcx, dcy, dw, dh, dang = det_rect
    gcx, gcy, gw, gh, gang = gt_rect
    dx = (gcx - dcx) / max(dw, 1.0)
    dy = (gcy - dcy) / max(dh, 1.0)
    log_dw = np.log(gw / max(dw, 1.0))
    log_dh = np.log(gh / max(dh, 1.0))
    dtheta = gang - dang
    while dtheta > 90.0:
        dtheta -= 180.0
    while dtheta < -90.0:
        dtheta += 180.0
    return np.array([dx, dy, log_dw, log_dh, dtheta / 90.0])


def apply_delta(det_rect, delta):
    dcx, dcy, dw, dh, dang = det_rect
    dx, dy, log_dw, log_dh, dtheta_n = delta
    cx = dcx + dx * dw
    cy = dcy + dy * dh
    w = dw * np.exp(log_dw)
    h = dh * np.exp(log_dh)
    ang = dang + dtheta_n * 90.0
    if ang >= 90.0:
        ang -= 180.0
    if ang < -90.0:
        ang += 180.0
    return cx, cy, w, h, ang


# ── Features ──

def red_support_per_edge(box, mask_d):
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
    return sup


def extract_features(box, mask, mask_d, W, H):
    rect = corners_to_rect(box)
    cx, cy, w, h, ang = rect
    sup = red_support_per_edge(box, mask_d)

    contour = box.reshape(-1, 1, 2).astype(np.int32)
    mask_inside = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(mask_inside, [contour], 255)
    n_inside = np.count_nonzero(mask_inside)
    n_red = np.count_nonzero(mask_inside & mask)
    fill = n_red / max(n_inside, 1)

    mask_ann = cv2.dilate(mask_inside, np.ones((5, 5), np.uint8), iterations=3) - mask_inside
    n_ann = np.count_nonzero(mask_ann)
    n_red_ann = np.count_nonzero(mask_ann & mask)
    ann_red = n_red_ann / max(n_ann, 1)

    feats = np.array([
        cx / W, cy / H,
        w / W, h / H,
        w / max(h, 1.0),
        ang / 90.0,
        np.mean(sup), np.min(sup), np.std(sup),
        sup[0], sup[1], sup[2], sup[3],
        fill, 1.0 - fill, ann_red,
    ])
    return feats, rect


# ── I/O ──

def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_boxes(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


# ── Matching ──

def best_gt_match(box, gt_boxes, used=None):
    best, bj = 0.0, -1
    for j, gb in enumerate(gt_boxes):
        if used and j in used:
            continue
        v = iou_obb(box, gb)
        if v > best:
            best, bj = v, j
    return best, bj


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


# ── Data collection ──

def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def collect_training_data(ids, iou_min=0.1):
    """Collect (features, delta) pairs from detections matched to GT (IoU > iou_min)."""
    X, y, meta = [], [], []
    n_near, n_good, n_miss = 0, 0, 0

    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        if not det or not gt:
            continue
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)

        used_gt = set()
        for di, db in enumerate(det):
            best, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if j in used_gt:
                    continue
                v = iou_obb(db, gb)
                if v > best:
                    best, bj = v, j
            if bj < 0 or best < iou_min:
                continue
            used_gt.add(bj)
            feats, det_rect = extract_features(db, mask, mask_d, W, H)
            gt_rect = corners_to_rect(gt[bj])
            delta = compute_delta(det_rect, gt_rect)
            X.append(feats); y.append(delta)
            meta.append((iid, di, best))
            if best < IOU_THRESHOLD:
                n_near += 1
            else:
                n_good += 1

        for j in range(len(gt)):
            if j not in used_gt:
                n_miss += 1

    return np.array(X), np.array(y), meta, n_near, n_good, n_miss


# ── Train ──

def train_model(X, y, hidden=(64, 32, 16)):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                       solver='adam', max_iter=500, random_state=0,
                       learning_rate_init=0.001, early_stopping=True,
                       validation_fraction=0.15)
    mlp.fit(X_s, y)
    return scaler, mlp


def predict_deltas(scaler, mlp, X):
    return mlp.predict(scaler.transform(X))


# ── Evaluate ──

def evaluate(scaler, mlp, ids, clip=True, gate_sup=False, sup_margin=0.0,
             clip_scale=1.0, preserve_sup=False, verbose=False):
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    iou_sum = 0.0
    n_corrected_to_tp = 0
    n_was_tp = 0
    n_gated = 0

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        if not det:
            FN += len(gt)
            continue
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)

        # Extract features + predict deltas
        feats_list, rects = [], []
        for db in det:
            f, r = extract_features(db, mask, mask_d, W, H)
            feats_list.append(f); rects.append(r)
        X = np.array(feats_list)
        deltas = predict_deltas(scaler, mlp, X)
        if clip:
            deltas = np.clip(deltas, -CLIP * clip_scale, CLIP * clip_scale)

        # Apply corrections (with optional red-support gate)
        corrected = []
        for di, (r, d) in enumerate(zip(rects, deltas)):
            nr = apply_delta(r, d)
            nc = rect_to_corners(*nr)
            if gate_sup:
                sup_old = red_support(det[di], mask_d)
                sup_new = red_support(nc, mask_d)
                if sup_new < sup_old - sup_margin:
                    corrected.append(det[di])
                    n_gated += 1
                    continue
            if preserve_sup:
                sup_old = red_support(det[di], mask_d)
                sup_new = red_support(nc, mask_d)
                sides_o = [np.linalg.norm(det[di][(i+1)%4] - det[di][i]) for i in range(4)]
                sides_n = [np.linalg.norm(nc[(i+1)%4] - nc[i]) for i in range(4)]
                was_real = (sup_old >= SUP_THR and min(sides_o) >= MIN_SIDE
                            and max(sides_o)/max(min(sides_o), 1e-6) <= MAX_ASPECT)
                still_real = (sup_new >= SUP_THR and min(sides_n) >= MIN_SIDE
                              and max(sides_n)/max(min(sides_n), 1e-6) <= MAX_ASPECT)
                if was_real and not still_real:
                    corrected.append(det[di])
                    n_gated += 1
                    continue
            corrected.append(nc)

        # Check how many near-misses became TP
        for di in range(len(det)):
            was_tp = any(iou_obb(det[di], gb) >= IOU_THRESHOLD for gb in gt)
            now_tp = any(iou_obb(corrected[di], gb) >= IOU_THRESHOLD for gb in gt)
            if now_tp and not was_tp:
                n_corrected_to_tp += 1
            if was_tp:
                n_was_tp += 1

        # Match corrected boxes
        csv_n = load_expected().get(iid, 0)
        tp, unmatched, iou_s = match_all(corrected, gt)
        TP += tp; FN += len(gt) - tp; iou_sum += iou_s
        raw_fp += len(unmatched)
        for di in unmatched:
            if is_real_unlabeled(corrected[di], csv_n, len(gt), mask_d):
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
        "raw_f1": round(f1_raw, 1), "raw_fpr": round(raw_fp / (TP + raw_fp) * 100, 1) if (TP + raw_fp) else 0,
        "corrected_f1": round(f1_c, 1), "corrected_fpr": round(true_fp / (TP + true_fp) * 100, 1) if (TP + true_fp) else 0,
        "avg_iou": round(iou_sum / max(TP, 1), 3),
        "near_to_tp": n_corrected_to_tp, "was_tp": n_was_tp, "gated": n_gated,
    }


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--hidden", default="64,32,16")
    ap.add_argument("--iou-min", type=float, default=0.1, help="min IoU for training pair collection")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--gate-sup", action="store_true", help="only apply correction if red support doesn't decrease")
    ap.add_argument("--sup-margin", type=float, default=0.0, help="allowed red support decrease")
    ap.add_argument("--clip-scale", type=float, default=1.0, help="scale factor for clip range")
    ap.add_argument("--preserve-sup", action="store_true", help="revert correction if it drops red support below SUP_THR")
    ap.add_argument("--constant", action="store_true", help="use mean delta instead of MLP")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    ids = get_ids()
    print(f"Images: {len(ids)}")

    if args.analyze:
        print("\n=== Analyzing near-miss distribution ===")
        for thr_low, thr_high in [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]:
            X, y, meta, n_near, n_good, n_miss = collect_training_data(ids, iou_min=thr_low)
            count = sum(1 for _, _, iou in meta if thr_low <= iou < thr_high)
            print(f"  IoU [{thr_low:.2f}, {thr_high:.2f}): {count} pairs")
        X, y, meta, n_near, n_good, n_miss = collect_training_data(ids, iou_min=0.1)
        print(f"\n  Training pairs (IoU >= {args.iou_min}): {len(X)}")
        print(f"    near-miss (IoU < 0.3): {n_near}")
        print(f"    good (IoU >= 0.3): {n_good}")
        print(f"    GT unmatched (complete misses): {n_miss}")
        print(f"  Delta stats (mean ± std):")
        for i, name in enumerate(["dx", "dy", "log_dw", "log_dh", "dθ/90"]):
            print(f"    {name}: {y[:, i].mean():.4f} ± {y[:, i].std():.4f}  [min={y[:, i].min():.3f}, max={y[:, i].max():.3f}]")
        print(f"  Feature count: {X.shape[1]}")
        return

    # Collect
    print(f"\n=== Collecting training data (IoU >= {args.iou_min}) ===")
    X, y, meta, n_near, n_good, n_miss = collect_training_data(ids, iou_min=args.iou_min)
    print(f"  Pairs: {len(X)}  (near-miss: {n_near}, good: {n_good}, misses: {n_miss})")

    # Train
    if args.constant:
        mean_delta = y.mean(axis=0)
        print(f"\n=== Constant correction (mean delta) ===")
        print(f"  delta = {mean_delta}")
        class ConstModel:
            def predict(self, X): return np.tile(mean_delta, (len(X), 1))
        class IdentityScaler:
            def transform(self, X): return X
        scaler, mlp = IdentityScaler(), ConstModel()
    else:
        hidden = tuple(int(x) for x in args.hidden.split(","))
        print(f"\n=== Training MLP {hidden} ===")
        scaler, mlp = train_model(X, y, hidden=hidden)
        print(f"  iters: {mlp.n_iter_}, train R²: {mlp.score(scaler.transform(X), y):.4f}")

    # Evaluate
    print("\n=== Evaluating (apply correction to all 500 images) ===")
    r = evaluate(scaler, mlp, ids, clip=not args.no_clip,
                 gate_sup=args.gate_sup, sup_margin=args.sup_margin,
                 clip_scale=args.clip_scale, preserve_sup=args.preserve_sup,
                 verbose=True)
    print("\n" + "=" * 60)
    print(json.dumps(r, indent=2))
    print("=" * 60)

    # Compare with baseline
    print("\n=== Comparison with baseline ===")
    print(f"  Baseline:   TP=677  FP=238  FN=69   Recall=90.8  raw_F1=81.5  corrected_F1=94.6")
    print(f"  Refined:    TP={r['tp']}  FP={r['raw_fp']}  FN={r['fn']}   Recall={r['recall']}  raw_F1={r['raw_f1']}  corrected_F1={r['corrected_f1']}")
    print(f"  Near→TP:    {r['near_to_tp']} boxes corrected above threshold  (gated: {r['gated']})")

    # GATE
    gate_raw = r["raw_f1"] >= 82.0 and r["raw_fp"] <= 238 + 5
    gate_corr = r["corrected_f1"] >= 94.6 and r["true_fp"] <= 8 + 5
    print(f"\n  GATE (raw):       F1≥82.0 AND FP≤243 → {'PASS' if gate_raw else 'FAIL'}")
    print(f"  GATE (corrected): cF1≥94.6 AND tFP≤13 → {'PASS' if gate_corr else 'FAIL'}")

    if args.save:
        out = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", args.save + ".json")
        r["hidden"] = "constant" if args.constant else list(hidden); r["iou_min"] = args.iou_min; r["clipped"] = not args.no_clip
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"  saved: {out}")
        model_path = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", args.save + "_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "mlp": mlp, "hidden": "constant" if args.constant else list(hidden)}, f)
        print(f"  model:  {model_path}")


if __name__ == "__main__":
    main()