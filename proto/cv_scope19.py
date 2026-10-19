"""LOOP-19: Train YOLOv11-OBB on corner-gap synthetic data, evaluate on real 500.

Strategy (2026-10-15):
  1. Generate training data: 50 X-ray backgrounds × ~20 diverse OBBs each
     (variable size, angle, position, brokenness 0/0.05/0.10/0.20/0.30)
     with GT = minAreaRect of rendered red pixels (LOOP-18 corner-gap model).
  2. YOLO OBB label format: cx, cy, w, h, angle (normalized 0-1)
  3. Train yolo11n-obb for 20 epochs on CPU (no CUDA).
  4. Run inference on 500 real images.
  5. Evaluate with point-match metric vs pm_v5_1 (TP=502, corr_f1=71.5).

Time budget: ~30-60 min for 2000 imgs × 20 epochs on CPU.
"""
import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import red_mask, IMG, LAB
from canny_quad import get_ids
from cv_hybrid import ROOT
from cv_scope13 import obb_corners, draw_obb, compute_gt_minarearect
from eval_pointmatch import match_boxes_point, load_boxes

BG_DIR = os.path.join(ROOT, "data", "Image@Xray_CSHH_Cargo_TF_P01_A")
TRAIN_DIR = os.path.join(ROOT, "tmp_yolo19_train")
MODEL_DIR = os.path.join(ROOT, "tmp_yolo19_model")
DET_DIR = os.path.join(ROOT, "tmp_yolo19_det")
RESULT_PATH = os.path.join(ROOT, "benchmark", "cv_scope19.json")
PM5_REF = os.path.join(ROOT, "tmp_sc6_hy_o1000_ba")  # pm_v5_1 frozen baseline
SEED = 42
D = 5.0


def yolo_obb_label(gt_corners, W, H):
    """Convert 4-corner quad to YOLO OBB label format.

    YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4 (9 columns, all normalized 0-1)
    The 4 corner points must be in the SAME ORDER as the model output.
    """
    pts = np.asarray(gt_corners, np.float32)
    # cv2.boxPoints returns 4 corners in a specific order (bottom-right, bottom-left, top-left, top-right)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    # Normalize to 0-1
    box_n = box / np.array([W, H], dtype=np.float32)
    coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in box_n)
    return f"0 {coords}"


def gen_train_data(n_per_bg=40):
    """Generate training images with corner-gap OBBs on X-ray backgrounds.

    Per background, generate n_per_bg OBBs at various sizes/angles/positions/brokenness.
    80% train, 20% val.
    """
    bg_files = sorted(glob.glob(os.path.join(BG_DIR, "*.jpg")))[:50]
    if not bg_files:
        raise FileNotFoundError(f"No background images in {BG_DIR}")
    print(f"  Using {len(bg_files)} X-ray backgrounds")

    img_train = os.path.join(TRAIN_DIR, "images", "train")
    img_val = os.path.join(TRAIN_DIR, "images", "val")
    lab_train = os.path.join(TRAIN_DIR, "labels", "train")
    lab_val = os.path.join(TRAIN_DIR, "labels", "val")
    for d in [img_train, img_val, lab_train, lab_val]:
        os.makedirs(d, exist_ok=True)

    rng = np.random.RandomState(SEED)
    total = 0
    for bi, bg_file in enumerate(bg_files):
        bg = cv2.imread(bg_file)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        for k in range(n_per_bg):
            # Diverse OBB params
            bw = int(rng.randint(40, 200))
            bh = int(rng.randint(30, 120))
            ang = int(rng.randint(0, 90))
            # Position
            cx = int(rng.randint(80, max(81, W - 80)))
            cy = int(rng.randint(60, max(61, H - 60)))
            # Brokenness: 0 to 0.30
            b = float(rng.choice([0.0, 0.05, 0.10, 0.20, 0.30]))
            # Number of boxes per image: 1-2
            n_obbs = int(rng.randint(1, 3))
            bp_list = []
            img = bg.copy()
            for _ in range(n_obbs):
                # Avoid overlap (simple: random position)
                cx2 = int(rng.randint(80, max(81, W - 80)))
                cy2 = int(rng.randint(60, max(61, H - 60)))
                bp = {'cx': cx2, 'cy': cy2, 'box_w': bw, 'box_h': bh,
                      'angle': ang, 'brokenness': b, 'stroke_w': 2}
                bp_list.append(bp)
                corners = obb_corners(bp['cx'], bp['cy'], bp['box_w'], bp['box_h'], bp['angle'])
                draw_obb(img, corners, brokenness=bp['brokenness'],
                         stroke_w=bp['stroke_w'], color=(0, 0, 255), rng=rng,
                         brokenness_mode='corner')
                gt = compute_gt_minarearect(img, corners, padding=10)
                # Add label
                label_str = yolo_obb_label(gt, W, H)
                bp['label'] = label_str
            # Save image + label
            iid = f"xray_{bi:03d}_{k:03d}"
            split = "val" if (bi * n_per_bg + k) % 5 == 0 else "train"
            img_path = os.path.join(TRAIN_DIR, "images", split, iid + ".jpg")
            lab_path = os.path.join(TRAIN_DIR, "labels", split, iid + ".txt")
            cv2.imwrite(img_path, img)
            with open(lab_path, "w") as f:
                for bp in bp_list:
                    f.write(bp['label'] + "\n")
            total += 1
    print(f"  Generated {total} training+val images")
    return total


def write_data_yaml():
    data = {
        "path": TRAIN_DIR,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "redbox"},
    }
    yaml_path = os.path.join(TRAIN_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        for k, v in data.items():
            if isinstance(v, dict):
                f.write(f"{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")
    return yaml_path


def train_yolo(epochs=20, imgsz=640, batch=4):
    from ultralytics import YOLO
    yaml_path = os.path.join(TRAIN_DIR, "data.yaml")
    model = YOLO("yolo11n-obb.pt")
    t0 = time.time()
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        project=os.path.join(MODEL_DIR, "runs"),
        name="yolo11n_obb_train",
        exist_ok=True,
        verbose=False,
        seed=SEED,
        workers=2,
        patience=10,
    )
    dt = time.time() - t0
    best_pt = os.path.join(MODEL_DIR, "runs", "yolo11n_obb_train", "weights", "best.pt")
    print(f"  Trained in {dt:.0f}s, best.pt at {best_pt}")
    return best_pt


def run_inference(best_pt, conf=0.25, iou=0.5, imgsz=640):
    from ultralytics import YOLO
    model = YOLO(best_pt)
    os.makedirs(DET_DIR, exist_ok=True)
    for f in os.listdir(DET_DIR):
        os.remove(os.path.join(DET_DIR, f))

    ids = get_ids()
    t0 = time.time()
    total_dets = 0
    for n, iid in enumerate(ids):
        img_path = os.path.join(IMG, iid + ".jpg")
        if not os.path.exists(img_path):
            with open(os.path.join(DET_DIR, iid + ".json"), "w") as f:
                json.dump({"infos": []}, f)
            continue
        # Run inference
        results = model.predict(img_path, conf=conf, iou=iou, imgsz=imgsz,
                                 verbose=False, device="cpu")
        # Convert OBB to 4-corner format
        det_quads = []
        for r in results:
            if r.obb is None:
                continue
            # r.obb.xyxyxyxy is (N, 4, 2) tensor
            for obb in r.obb:
                # obb.xyxyxyxy is (4, 2) — 4 corners
                try:
                    corners = obb.xyxyxyxy.cpu().numpy().reshape(4, 2)
                    det_quads.append(corners.astype(np.int32))
                except Exception:
                    # Fallback: use obb.xywhr to get cx,cy,w,h,angle
                    try:
                        xywhr = obb.xywhr.cpu().numpy()
                        cx, cy, w, h, angle_rad = xywhr
                        angle_deg = float(angle_rad) * 180.0 / np.pi
                        # Convert to 4 corners using obb_corners convention
                        corners = obb_corners(float(cx), float(cy), float(w), float(h), angle_deg)
                        det_quads.append(corners.astype(np.int32))
                    except Exception:
                        continue
        total_dets += len(det_quads)
        # Save in cvhybrid JSON format
        with open(os.path.join(DET_DIR, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "image_id": iid,
                "infos": [
                    {"points": ";".join(f"{int(x)},{int(y)}" for x, y in q)}
                    for q in det_quads
                ],
            }, f, ensure_ascii=False)
        if (n + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [yolo {n+1}/{len(ids)}] dets={total_dets} elapsed={dt:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"  Inference done: {total_dets} total dets in {dt:.0f}s", flush=True)


def evaluate(dist_thr=5.0):
    ids = get_ids()
    exp_path = os.path.join(ROOT, "benchmark", "pm_v5_1_baseline.json")
    with open(exp_path, encoding="utf-8") as f:
        pm5 = json.load(f)
    pm5_d5 = pm5["results"]["o1000_ba"]["5"]
    print(f"  pm_v5_1 baseline: TP={pm5_d5['tp']} corr_f1={pm5_d5['corrected_f1']}%")

    TP = FN = FP = 0
    n_gt = 0
    n_det = 0
    for iid in ids:
        gt = load_boxes(LAB, iid)
        det = load_boxes(DET_DIR, iid)
        tp, unmatched = match_boxes_point(
            [d.astype(np.int32) for d in det],
            [g.astype(np.int32) for g in gt],
            dist_thr,
        )
        TP += tp
        FN += len(gt) - tp
        FP += len(unmatched)
        n_gt += len(gt)
        n_det += len(det)
    recall = TP / n_gt * 100 if n_gt else 0
    raw_prec = TP / n_det * 100 if n_det else 0
    f1 = 2 * raw_prec * recall / (raw_prec + recall) if (raw_prec + recall) else 0
    # Note: this is a point-match-only metric, not the GT-corrected F1
    # For comparison, use pm_v5_1 raw_f1 as the analog
    pm5_raw_f1 = pm5_d5['raw_f1']
    pm5_corr_f1 = pm5_d5['corrected_f1']
    print(f"  yolo11n-obb: TP={TP} FN={FN} FP={FP}")
    print(f"    n_gt={n_gt} n_det={n_det}")
    print(f"    recall={recall:.1f}% raw_prec={raw_prec:.1f}% raw_f1={f1:.1f}%")
    return {
        "tp": TP, "fn": FN, "fp": FP, "n_gt": n_gt, "n_det": n_det,
        "recall": round(recall, 2),
        "raw_precision": round(raw_prec, 2),
        "raw_f1": round(f1, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-bg', type=int, default=40,
                    help='Number of OBBs per X-ray background (default 40 → 2000 imgs)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-gen', action='store_true')
    args = ap.parse_args()

    print(f"LOOP-19: YOLOv11-OBB on corner-gap synthetic (D={D})", flush=True)
    print(f"  Backgrounds: 50 X-ray, n_per_bg={args.n_per_bg}, brokenness=0/0.05/0.10/0.20/0.30", flush=True)
    print(f"  Training: yolo11n-obb, {args.epochs} epochs, imgsz={args.imgsz}, batch={args.batch}", flush=True)

    if not args.skip_gen:
        print(f"\n[1/4] Generating training data ({args.n_per_bg} OBBs per bg)...", flush=True)
        t0 = time.time()
        n = gen_train_data(args.n_per_bg)
        dt = time.time() - t0
        print(f"  Generated {n} images in {dt:.1f}s", flush=True)
        yaml_path = write_data_yaml()
        print(f"  Wrote {yaml_path}", flush=True)

    if not args.skip_train:
        print(f"\n[2/4] Training yolo11n-obb ({args.epochs} epochs)...", flush=True)
        t0 = time.time()
        best_pt = train_yolo(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
        dt = time.time() - t0
        print(f"  Training done in {dt:.0f}s", flush=True)
    else:
        best_pt = os.path.join(MODEL_DIR, "runs", "yolo11n_obb_train", "weights", "best.pt")
        if not os.path.exists(best_pt):
            print(f"  No pre-trained model at {best_pt}; run without --skip-train first")
            return

    print(f"\n[3/4] Running inference on 500 real images (conf={args.conf})...", flush=True)
    t0 = time.time()
    run_inference(best_pt, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    dt = time.time() - t0
    print(f"  Inference done in {dt:.0f}s", flush=True)

    print(f"\n[4/4] Evaluating (D={D})...", flush=True)
    r = evaluate(D)
    print(f"\nVerdict:")
    pm5_tp = 502
    pm5_corr_f1 = 71.5
    delta_tp = r['tp'] - pm5_tp
    delta_f1 = r['raw_f1'] - pm5_corr_f1
    if r['tp'] > pm5_tp and r['fp'] <= 161:
        verdict = "WIN"
    elif r['tp'] >= pm5_tp - 5 and r['fp'] <= 161 + 10:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"
    print(f"  yolo11n-obb vs pm_v5_1: TP {r['tp']} vs {pm5_tp} (Δ={delta_tp:+d}), "
          f"raw_f1 {r['raw_f1']} vs corr_f1 {pm5_corr_f1} ({verdict})", flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "yolo11n_obb": r,
            "verdict": verdict,
            "pm_v5_1_baseline": {
                "tp": pm5_tp, "corr_f1": pm5_corr_f1,
                "raw_f1": pm5_corr_f1,  # alias
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()