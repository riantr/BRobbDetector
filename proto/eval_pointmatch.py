"""Point-match evaluation: per-box, all 4 corners within D px (Hungarian 4-to-4).

New objective: a detected box is TP only if all 4 corners, after optimal
4-to-4 Hungarian assignment to a GT box's 4 corners, are each within D pixels
(default 1 px). Reports raw and GT-corrected (unlabeled-real) metrics.

Usage:
  python proto/eval_pointmatch.py --m8 --dist 1
  python proto/eval_pointmatch.py --yolo --conf 0.55 --dist 1
  python proto/eval_pointmatch.py --det-dir <path> --dist 1
"""
import sys, os, json, argparse, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
SUP_THR = 0.30
MIN_SIDE = 12.0
MAX_ASPECT = 8.0

PERMS_4 = list(itertools.permutations(range(4)))

W_FACTOR = 1.172
H_FACTOR = 1.187
DY_SHIFT = -0.025


def refine_box(box):
    """LOOP-3 constant correction: grow w/h, shift cy."""
    rect = cv2.minAreaRect(box.astype(np.float32))
    (cx, cy), (w, h), ang = rect
    cy2 = cy + DY_SHIFT * h
    w2 = w * W_FACTOR
    h2 = h * H_FACTOR
    r = cv2.RotatedRect((cx, cy2), (w2, h2), ang)
    pts = cv2.boxPoints(r)
    return np.round(pts).astype(np.int32)


def snap_box_edge_fit(box, mask, search_width=15, max_shift=30, use_all_red=False):
    """Snap box corners by fitting lines to red pixels along each edge,
    then computing intersections of adjacent fitted lines.

    For each position along an edge, search perpendicular and pick the NEAREST
    red pixel (smallest |offset|). This avoids contamination from text/noise
    inside the box, which would be farther from the edge center line.
    If use_all_red=True, use ALL red pixels in the search area instead."""
    box = box.astype(np.float64)
    H, W = mask.shape

    fitted_lines = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 2:
            fitted_lines.append(None)
            continue
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])

        n_along = max(int(length), 10)
        ts = np.linspace(0.0, 1.0, n_along)
        ss = np.arange(-search_width, search_width + 1)

        centers = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        pts = centers[:, None, :] + ss[None, :, None] * normal[None, None, :]

        xs = np.clip(np.round(pts[:, :, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, :, 1]).astype(int), 0, H - 1)

        red = mask[ys, xs] > 0  # (n_along, n_perp)
        if red.sum() < 8:
            fitted_lines.append(None)
            continue

        if use_all_red:
            edge_pts = pts[red]
        else:
            abs_s = np.abs(ss).astype(np.float64)
            cost = np.where(red, abs_s[None, :], np.inf)
            best_idx = np.argmin(cost, axis=1)
            has_red = cost[np.arange(n_along), best_idx] < np.inf
            edge_pts = pts[np.arange(n_along)[has_red], best_idx[has_red]]

        if len(edge_pts) < 8:
            fitted_lines.append(None)
            continue

        line = cv2.fitLine(edge_pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        fitted_lines.append(line)

    snapped = []
    for i in range(4):
        line_a = fitted_lines[(i - 1) % 4]
        line_b = fitted_lines[i]
        if line_a is None or line_b is None:
            snapped.append(box[i])
            continue

        vx1, vy1, x01, y01 = line_a.flatten()
        vx2, vy2, x02, y02 = line_b.flatten()

        A = np.array([[vx1, -vx2], [vy1, -vy2]])
        b = np.array([x02 - x01, y02 - y01])
        try:
            t1, t2 = np.linalg.solve(A, b)
            pt = np.array([x01 + t1 * vx1, y01 + t1 * vy1])
        except np.linalg.LinAlgError:
            pt = box[i]

        if np.linalg.norm(pt - box[i]) > max_shift:
            pt = box[i]
        snapped.append(pt)

    return np.array(snapped, dtype=np.int32)


def snap_box_angle_constrained(box, mask, search_width=20, max_shift=30):
    """Snap box by shifting edges to align with red pixels, keeping angles fixed.

    For each edge, find nearest red pixels perpendicular to the edge, compute
    the median perpendicular offset, and shift the edge by that amount.
    Intersections of adjacent shifted edges give the snapped corners."""
    box = box.astype(np.float64)
    H, W = mask.shape

    edge_lines = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 2:
            edge_lines.append(None)
            continue
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])

        n_along = max(int(length), 10)
        ts = np.linspace(0.0, 1.0, n_along)
        ss = np.arange(-search_width, search_width + 1)

        centers = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        pts = centers[:, None, :] + ss[None, :, None] * normal[None, None, :]

        xs = np.clip(np.round(pts[:, :, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, :, 1]).astype(int), 0, H - 1)

        red = mask[ys, xs] > 0
        if red.sum() < 8:
            edge_lines.append((p1.copy(), direction.copy()))
            continue

        abs_s = np.abs(ss).astype(np.float64)
        cost = np.where(red, abs_s[None, :], np.inf)
        best_idx = np.argmin(cost, axis=1)
        has_red = cost[np.arange(n_along), best_idx] < np.inf

        shifts = ss[best_idx[has_red]].astype(np.float64)
        if len(shifts) < 8:
            edge_lines.append((p1.copy(), direction.copy()))
            continue

        median_shift = np.median(shifts)
        shifted_p1 = p1 + median_shift * normal
        edge_lines.append((shifted_p1, direction.copy()))

    snapped = []
    for i in range(4):
        line_a = edge_lines[(i - 1) % 4]
        line_b = edge_lines[i]
        if line_a is None or line_b is None:
            snapped.append(box[i])
            continue

        p_a, d_a = line_a
        p_b, d_b = line_b

        A = np.array([[d_a[0], -d_b[0]], [d_a[1], -d_b[1]]])
        b = np.array([p_b[0] - p_a[0], p_b[1] - p_a[1]])
        try:
            t, s = np.linalg.solve(A, b)
            pt = p_a + t * d_a
        except np.linalg.LinAlgError:
            pt = box[i]

        if np.linalg.norm(pt - box[i]) > max_shift:
            pt = box[i]
        snapped.append(pt)

    return np.array(snapped, dtype=np.int32)


def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_boxes(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def corner_match_distance(det_box, gt_box):
    """Optimal 4-to-4 corner assignment; returns max corner distance (or inf)."""
    best = 1e18
    for perm in PERMS_4:
        mx = 0.0
        for i in range(4):
            dx = det_box[i, 0] - gt_box[perm[i], 0]
            dy = det_box[i, 1] - gt_box[perm[i], 1]
            d = dx * dx + dy * dy
            if d > mx:
                mx = d
        if mx < best:
            best = mx
    return best ** 0.5


def match_boxes_point(det, gt, dist_thr):
    """Greedy match: sort all (di, gj) by corner_match_distance ascending."""
    candidates = []
    for di in range(len(det)):
        for gj in range(len(gt)):
            d = corner_match_distance(det[di], gt[gj])
            if d <= dist_thr:
                candidates.append((d, di, gj))
    candidates.sort()
    used_det = set()
    used_gt = set()
    tp = 0
    for d, di, gj in candidates:
        if di in used_det or gj in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gj)
        tp += 1
    unmatched = [di for di in range(len(det)) if di not in used_det]
    return tp, unmatched


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


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def run_yolo(ids, conf=0.55):
    from ultralytics import YOLO
    model_path = r"D:\src\CodeArts\BRobbDetector\tmp_yolo\train\weights\best.onnx"
    model = YOLO(model_path, task="obb")
    det_map = {}
    for n, iid in enumerate(ids):
        img_path = os.path.join(IMG, iid + ".jpg")
        results = model.predict(img_path, verbose=False, conf=conf)
        boxes = []
        if results and results[0].obb is not None:
            for obb in results[0].obb:
                pts = obb.cpu().numpy().xyxyxyxy[0].astype(np.int32)
                if len(pts) == 4:
                    boxes.append(pts)
        det_map[iid] = boxes
        if (n + 1) % 100 == 0:
            print(f"  [YOLO {n+1}/{len(ids)}] done")
    return det_map


def evaluate(det_source, dist_thr=1.0, verbose=False, refine=False, snap=False, snap_window=30, snap_all_red=False, conf=0.55, snap_passes=1, snap_angle=False):
    exp = load_expected()
    ids = get_ids()

    yolo_map = None
    if det_source in ("yolo", "combine"):
        yolo_map = run_yolo(ids, conf=conf)

    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    dist_sum = 0.0

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)

        if det_source == "yolo":
            det = yolo_map.get(iid, [])
        elif det_source == "m8":
            det = load_boxes(OUT, iid)
        elif det_source == "combine":
            m8_det = list(load_boxes(OUT, iid))
            yolo_det = list(yolo_map.get(iid, []))
            det = m8_det + yolo_det
            m8_count = len(m8_det)
        else:
            det = load_boxes(det_source, iid)
            m8_count = 0

        det = [d.astype(np.int32) for d in det]
        if refine:
            det = [refine_box(d) for d in det]

        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)

        if snap:
            snap_fn = snap_box_angle_constrained if snap_angle else snap_box_edge_fit
            for _ in range(snap_passes):
                det = [snap_fn(d, mask, search_width=snap_window) for d in det]

        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp; FN += len(gt) - tp
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
        "source": det_source, "dist_thr": dist_thr,
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp, "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "raw_f1": round(f1_raw, 1), "corrected_f1": round(f1_c, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m8", action="store_true")
    ap.add_argument("--yolo", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap-window", type=int, default=30)
    ap.add_argument("--snap-all-red", action="store_true")
    ap.add_argument("--snap-passes", type=int, default=1)
    ap.add_argument("--snap-angle", action="store_true")
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--dist", type=float, default=1.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if args.m8:
        source = "m8"
    elif args.yolo:
        source = "yolo"
    elif args.combine:
        source = "combine"
    elif args.det_dir:
        source = args.det_dir
    else:
        ap.error("specify --m8, --yolo, --combine, or --det-dir <path>")

    print(f"Evaluating: source={source}, dist_thr={args.dist}, refine={args.refine}, snap={args.snap}")
    r = evaluate(source, dist_thr=args.dist, verbose=True, refine=args.refine,
                 snap=args.snap, snap_window=args.snap_window, snap_all_red=args.snap_all_red,
                 conf=args.conf, snap_passes=args.snap_passes, snap_angle=args.snap_angle)
    print("\n" + "=" * 60)
    print(json.dumps(r, indent=2))
    print("=" * 60)
    if args.save:
        out = os.path.join(r"D:\src\CodeArts\BRobbDetector\benchmark", args.save + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()