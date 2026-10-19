"""LOOP-9: CornerNet-style heatmap corner regression for OBB detection.

Train a small U-Net to predict corner heatmaps from red masks.
At inference, extract peaks and group into orthogonal quads.

Pipeline:
  1. Input: red mask (1 ch), resized to RESxRES with AR-preserving pad
  2. Output: 1 heatmap ch + 2 offset chs
  3. Loss: penalized focal loss (heatmap) + smooth L1 (offset at corners)
  4. Inference: 3x3 max-pool NMS -> top-K peaks -> offset refine -> group quads
  5. Grouping: for each corner, check K-nearest combos for orthogonal quads
  6. Eval: point-match metric on all 500, compare with pm_v5

Usage:
  python proto/heatmap_loop9.py            # train + infer + eval
  python proto/heatmap_loop9.py --train    # train only
  python proto/heatmap_loop9.py --infer    # infer + eval only
"""
import sys, os, json, argparse, itertools, time, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from common import IMG, LAB, red_mask, load_expected, iou_obb
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled, get_ids
from quad_ortho import is_orthogonal_quad
from canny_quad import to_json

ROOT = r"D:\src\CodeArts\BRobbDetector"
DET_DIR = os.path.join(ROOT, "tmp_heatmap_loop9")
MODEL_PATH = os.path.join(ROOT, "benchmark", "heatmap_loop9_model.pt")
RESULT_PATH = os.path.join(ROOT, "benchmark", "heatmap_loop9.json")
PM5_DIR = os.path.join(ROOT, "tmp_sc6_hy_o15_ba")

RES = 384
SIGMA = 2.0
BASE_CH = 16
BATCH = 8
EPOCHS = 15
LR = 1e-3
AUG_FLIPS = [(False, False), (True, False)]
TOP_K = 40
PEAK_THR = 0.3
GROUP_K = 10
ORTHO_TOL = 15.0
MIN_SIDE = 12.0
MAX_ASPECT = 12.0
DEDUP_IOU = 0.3
SEED = 42
POS_WEIGHT = 30.0
DIST_VALUES = [3, 5, 8, 12]


# ── Data ──

def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_gt(iid):
    path = os.path.join(LAB, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def resize_pad(mask, res=None):
    res = RES if res is None else res
    h, w = mask.shape[:2]
    scale = res / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    padded = np.zeros((res, res), dtype=np.uint8)
    padded[:nh, :nw] = resized
    return padded, scale


def make_heatmap(corners, res=None, sigma=None):
    res = RES if res is None else res
    sigma = SIGMA if sigma is None else sigma
    hm = np.zeros((res, res), dtype=np.float32)
    r = int(math.ceil(3 * sigma))
    for cx, cy in corners:
        cx, cy = float(cx), float(cy)
        px, py = int(round(cx)), int(round(cy))
        if 0 <= px < res and 0 <= py < res:
            hm[py, px] = 1.0
        x1, x2 = max(0, int(cx) - r), min(res, int(cx) + r + 1)
        y1, y2 = max(0, int(cy) - r), min(res, int(cy) + r + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        ys = np.arange(y1, y2, dtype=np.float32) - cy
        xs = np.arange(x1, x2, dtype=np.float32) - cx
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        v = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        hm[y1:y2, x1:x2] = np.maximum(hm[y1:y2, x1:x2], v)
    return hm


def make_offset(corners, res=None):
    res = RES if res is None else res
    ox = np.zeros((res, res), dtype=np.float32)
    oy = np.zeros((res, res), dtype=np.float32)
    mask = np.zeros((res, res), dtype=np.float32)
    for cx, cy in corners:
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < res and 0 <= iy < res:
            ox[iy, ix] = cx - ix
            oy[iy, ix] = cy - iy
            mask[iy, ix] = 1.0
    return ox, oy, mask


def prepare_data(ids):
    inputs, hms, oxs, oys, masks = [], [], [], [], []
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            inputs.append(np.zeros((RES, RES), np.float32))
            hms.append(np.zeros((RES, RES), np.float32))
            oxs.append(np.zeros((RES, RES), np.float32))
            oys.append(np.zeros((RES, RES), np.float32))
            masks.append(np.zeros((RES, RES), np.float32))
            continue
        m = red_mask(img)
        m_pad, scale = resize_pad(m)
        boxes = load_gt(iid)
        corners = []
        for box in boxes:
            for pt in box:
                corners.append([pt[0] * scale, pt[1] * scale])
        hm = make_heatmap(corners)
        ox, oy, cmask = make_offset(corners)
        inputs.append(m_pad.astype(np.float32) / 255.0)
        hms.append(hm)
        oxs.append(ox)
        oys.append(oy)
        masks.append(cmask)
        if (n + 1) % 100 == 0:
            print(f"  prep {n+1}/{len(ids)}")
    return (np.stack(inputs), np.stack(hms), np.stack(oxs),
            np.stack(oys), np.stack(masks))


def augment(inp, hm, ox, oy, cm, flip_h, flip_v):
    if flip_h:
        inp = inp[:, ::-1].copy()
        hm = hm[:, ::-1].copy()
        ox = -ox[:, ::-1].copy()
        oy = oy[:, ::-1].copy()
        cm = cm[:, ::-1].copy()
    if flip_v:
        inp = inp[::-1, :].copy()
        hm = hm[::-1, :].copy()
        ox = ox[::-1, :].copy()
        oy = -oy[::-1, :].copy()
        cm = cm[::-1, :].copy()
    return inp, hm, ox, oy, cm


# ── Model ──

class CornerUNet(nn.Module):
    def __init__(self, in_ch=1, base=BASE_CH):
        super().__init__()
        c = base
        self.enc1 = self._block(in_ch, c)
        self.enc2 = self._block(c, c * 2)
        self.enc3 = self._block(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bot = self._block(c * 4, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = self._block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = self._block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = self._block(c * 2, c)
        self.hm_head = nn.Conv2d(c, 1, 1)
        self.off_head = nn.Conv2d(c, 2, 1)

    @staticmethod
    def _block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        hm = torch.sigmoid(self.hm_head(d1))
        off = self.off_head(d1)
        return hm, off


# ── Loss ──

def focal_loss(pred, target, alpha=2.0, beta=4.0, pos_weight=None):
    pos_weight = POS_WEIGHT if pos_weight is None else pos_weight
    pred = pred.clamp(1e-6, 1 - 1e-6)
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    pos_loss = -(1 - pred).pow(alpha) * torch.log(pred) * pos
    neg_loss = -(1 - target).pow(beta) * pred.pow(alpha) * torch.log(1 - pred) * neg
    num_pos = pos.sum().clamp(min=1)
    return (pos_weight * pos_loss + neg_loss).sum() / num_pos


def offset_loss(pred, gt, mask):
    mask = mask.unsqueeze(1).expand_as(pred)
    num = mask.sum().clamp(min=1)
    return F.smooth_l1_loss(pred * mask, gt * mask, reduction='sum') / num


# ── Train ──

def train_model(train_ids, val_ids):
    print("Preparing training data...")
    ti, th, tox, toy, tcm = prepare_data(train_ids)
    vi, vh, vox, voy, vcm = prepare_data(val_ids)

    def to_tensor(*arrs):
        return [torch.from_numpy(a) for a in arrs]

    ti_t, th_t, tox_t, toy_t, tcm_t = to_tensor(ti, th, tox, toy, tcm)
    vi_t, vh_t, vox_t, voy_t, vcm_t = to_tensor(vi, vh, vox, voy, vcm)

    model = CornerUNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)

    n_aug = len(AUG_FLIPS)
    aug_inps, aug_hms, aug_oxs, aug_oys, aug_cms = [], [], [], [], []
    for i in range(len(ti)):
        for fh, fv in AUG_FLIPS:
            a = augment(ti[i], th[i], tox[i], toy[i], tcm[i], fh, fv)
            aug_inps.append(a[0]); aug_hms.append(a[1])
            aug_oxs.append(a[2]); aug_oys.append(a[3]); aug_cms.append(a[4])
    aug_inps = np.stack(aug_inps); aug_hms = np.stack(aug_hms)
    aug_oxs = np.stack(aug_oxs); aug_oys = np.stack(aug_oys)
    aug_cms = np.stack(aug_cms)
    print(f"  augmented: {len(aug_inps)} samples ({len(train_ids)} x {n_aug})")

    ds = TensorDataset(
        torch.from_numpy(aug_inps).unsqueeze(1),
        torch.from_numpy(aug_hms).unsqueeze(1),
        torch.from_numpy(aug_oxs),
        torch.from_numpy(aug_oys),
        torch.from_numpy(aug_cms),
    )
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

    best_val = 1e18
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss = 0
        for bi, (x, hm, ox, oy, cm) in enumerate(dl):
            opt.zero_grad()
            pred_hm, pred_off = model(x)
            gt_off = torch.stack([ox, oy], dim=1)
            loss_hm = focal_loss(pred_hm, hm)
            loss_off = offset_loss(pred_off, gt_off, cm)
            loss = loss_hm + loss_off
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv_hm, pv_off = model(vi_t.unsqueeze(1))
            v_loss_hm = focal_loss(pv_hm, vh_t.unsqueeze(1)).item()
            v_loss_off = offset_loss(pv_off, torch.stack([vox_t, voy_t], dim=1), vcm_t).item()
            v_loss = v_loss_hm + v_loss_off

        dt = time.time() - t0
        print(f"  epoch {epoch+1}/{EPOCHS}  train={total_loss/len(dl):.4f}  "
              f"val={v_loss:.4f}  (hm={v_loss_hm:.4f} off={v_loss_off:.4f})  {dt:.1f}s")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Best val loss: {best_val:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    return model


# ── Inference ──

def extract_peaks(heatmap, k=TOP_K, thr=PEAK_THR):
    pooled = F.max_pool2d(
        torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0),
        3, stride=1, padding=1
    ).squeeze().numpy()
    mask = (heatmap == pooled) & (heatmap >= thr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    vs = heatmap[ys, xs]
    idx = np.argsort(-vs)[:k]
    return [(int(xs[i]), int(ys[i]), float(vs[i])) for i in idx]


def group_quads(peaks, img_w, img_h):
    if len(peaks) < 4:
        return []
    corners = np.array([(p[0], p[1]) for p in peaks], dtype=np.float32)
    scores = np.array([p[2] for p in peaks])

    dists = np.sqrt(((corners[:, None] - corners[None, :]) ** 2).sum(axis=2))
    K = min(GROUP_K, len(corners))

    candidates = []
    for i in range(len(corners)):
        nn_idx = np.argsort(dists[i])[1:K]
        for combo in itertools.combinations(nn_idx, 3):
            quad_idx = [i] + list(combo)
            pts = corners[quad_idx]
            hull = cv2.convexHull(pts.astype(np.float32))
            if len(hull) != 4:
                continue
            hull = hull.reshape(4, 2)
            if not is_orthogonal_quad(hull, img_w, img_h, tol=ORTHO_TOL,
                                      boundary_band=12.0):
                continue
            sides = [np.linalg.norm(hull[(j + 1) % 4] - hull[j]) for j in range(4)]
            mn, mx = min(sides), max(sides)
            if mn < MIN_SIDE or mx / max(mn, 1e-6) > MAX_ASPECT:
                continue
            score = float(sum(scores[q] for q in quad_idx))
            candidates.append((score, hull))

    candidates.sort(key=lambda c: -c[0])
    candidates = candidates[:50]

    result = []
    for score, quad in candidates:
        keep = True
        for _, rq in result:
            if iou_obb(quad.astype(np.int32), rq.astype(np.int32)) > DEDUP_IOU:
                keep = False
                break
        if keep:
            result.append((score, quad))
    return [q.astype(np.int32) for _, q in result]


def run_inference(model, ids):
    os.makedirs(DET_DIR, exist_ok=True)
    model.eval()
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            with open(os.path.join(DET_DIR, iid + ".json"), "w", encoding="utf-8") as f:
                json.dump(to_json(iid, []), f, ensure_ascii=False)
            continue
        H, W = img.shape[:2]
        m = red_mask(img)
        m_pad, scale = resize_pad(m)
        x = torch.from_numpy(m_pad.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred_hm, pred_off = model(x)
        hm = pred_hm.squeeze().numpy()
        off = pred_off.squeeze().numpy()

        peaks = extract_peaks(hm)
        refined = []
        for px, py, pv in peaks:
            dx = off[0, py, px]
            dy = off[1, py, px]
            fx = (px + dx) / scale
            fy = (py + dy) / scale
            fx = np.clip(fx, 0, W - 1)
            fy = np.clip(fy, 0, H - 1)
            refined.append((fx, fy, pv))

        quads = group_quads(refined, W, H)
        total += len(quads)
        with open(os.path.join(DET_DIR, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, quads), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            print(f"  infer {n+1}/{len(ids)}  boxes={total}")
    print(f"  total boxes: {total}")
    return total


# ── Eval ──

def evaluate_dir(det_dir, ids, dists=DIST_VALUES):
    exp = load_expected()
    results = {}
    for D in dists:
        TP = FN = raw_fp = true_fp = real_unl = 0
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            gt = load_gt(iid)
            det = load_boxes(det_dir, iid)
            det = [d.astype(np.int32) for d in det]
            mask = red_mask(img)
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
            csv_n = exp.get(iid, 0)
            tp, unmatched = match_boxes_point(det, gt, float(D))
            TP += tp; FN += len(gt) - tp
            raw_fp += len(unmatched)
            for di in unmatched:
                if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                    real_unl += 1
                else:
                    true_fp += 1
        total_gt = TP + FN
        recall = TP / total_gt * 100 if total_gt else 0
        prec_raw = TP / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
        f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
        prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
        f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0
        results[f"D{D}"] = {
            "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp,
            "real_unlabeled": real_unl,
            "recall": round(recall, 1),
            "raw_f1": round(f1_raw, 1), "corrected_f1": round(f1_c, 1),
        }
    return results


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--infer", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(12)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ids = get_ids()
    ids.sort()
    rng = np.random.RandomState(SEED)
    rng.shuffle(ids)
    train_ids = ids[:400]
    val_ids = ids[400:]
    print(f"LOOP-9: Heatmap corner regression")
    print(f"  train={len(train_ids)}  val={len(val_ids)}  total={len(ids)}")

    model = None
    if not args.infer:
        model = train_model(train_ids, val_ids)

    if model is None and os.path.exists(MODEL_PATH):
        model = CornerUNet()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"Loaded model: {MODEL_PATH}")

    if model is not None:
        print("\nRunning inference on all 500 images...")
        run_inference(model, ids)

    print("\nEvaluating heatmap (LOOP-9)...")
    hm_all = evaluate_dir(DET_DIR, ids)
    hm_val = evaluate_dir(DET_DIR, val_ids)

    print("Evaluating pm_v5 on val-100 for fair comparison...")
    pm5_val = evaluate_dir(PM5_DIR, val_ids)

    print("\n" + "=" * 70)
    print("HEATMAP LOOP-9  (all 500)")
    for k, v in hm_all.items():
        print(f"  {k}: TP={v['tp']} rawFP={v['raw_fp']} trueFP={v['true_fp']} "
              f"recall={v['recall']} corr_f1={v['corrected_f1']}")
    print("\nHEATMAP LOOP-9  (val-100)")
    for k, v in hm_val.items():
        print(f"  {k}: TP={v['tp']} rawFP={v['raw_fp']} trueFP={v['true_fp']} "
              f"recall={v['recall']} corr_f1={v['corrected_f1']}")
    print("\nPM_V5  (val-100)")
    for k, v in pm5_val.items():
        print(f"  {k}: TP={v['tp']} rawFP={v['raw_fp']} trueFP={v['true_fp']} "
              f"recall={v['recall']} corr_f1={v['corrected_f1']}")

    verdict = "DEAD" if hm_val.get("D5", {}).get("tp", 0) <= pm5_val.get("D5", {}).get("tp", 0) else "PROMISING"
    print(f"\nVERDICT: {verdict}")
    print("=" * 70)

    out = {
        "loop": "LOOP-9",
        "method": "CornerNet-style heatmap corner regression",
        "config": {"res": RES, "sigma": SIGMA, "base_ch": BASE_CH,
                    "batch": BATCH, "epochs": EPOCHS, "lr": LR},
        "heatmap_all500": hm_all,
        "heatmap_val100": hm_val,
        "pm5_val100": pm5_val,
        "verdict": verdict,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()