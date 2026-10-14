"""Diagnose LSD-added boxes: which features separate true-added from spurious-added?"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb
from lsd_rect import lsd_segments, assemble_rects, edge_supports, check_orthogonal, DEFAULT, parse_points, load_boxes

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
M8 = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"


def features(mask, mask_d, rect):
    sup = edge_supports(mask_d, rect)
    # undilated support band=1: sample mask directly
    sup0 = edge_supports(mask, rect)
    sides = [np.linalg.norm(rect[(i + 1) % 4] - rect[i]) for i in range(4)]
    # interior red density
    x, y, w, h = cv2.boundingRect(rect.astype(np.int32))
    roi = mask[y:y + h, x:x + w]
    dens = np.count_nonzero(roi) / roi.size if roi.size else 0
    area = cv2.contourArea(rect.astype(np.float32))
    # rotated fill ratio + hole fraction
    H, W = mask.shape[:2]
    rm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(rm, [rect.astype(np.int32)], 255)
    ra = np.count_nonzero(rm)
    red_in = np.count_nonzero(cv2.bitwise_and(rm, mask))
    fill = red_in / ra if ra else 1.0
    inv = cv2.bitwise_and(rm, cv2.bitwise_not(mask))
    ncc, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    hole = 0
    for i in range(1, ncc):
        hole = max(hole, stats[i, cv2.CC_STAT_AREA])
    hole_frac = hole / ra if ra else 0.0
    return {
        "sup_mean": float(np.mean(sup)),
        "sup_min": float(np.min(sup)),
        "sup0_mean": float(np.mean(sup0)),
        "sup0_min": float(np.min(sup0)),
        "n_edges_sup30": int(sum(1 for s in sup if s >= 0.3)),
        "n_edges_sup50": int(sum(1 for s in sup if s >= 0.5)),
        "min_side": float(min(sides)),
        "max_side": float(max(sides)),
        "aspect": float(max(sides) / max(min(sides), 1e-6)),
        "area": float(area),
        "interior_red_density": float(dens),
        "fill_rot": float(fill),
        "hole_frac": float(hole_frac),
    }


def main():
    n_subset = 200
    with open(M8, encoding="utf-8") as f:
        bench = json.load(f)
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:n_subset]

    p = dict(DEFAULT)
    rows = []
    t0 = time.time()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        segs = lsd_segments(mask, p["min_len"])
        rects = assemble_rects(segs, W, H, mask_d, p)
        # dedup
        rects = sorted(rects, key=lambda r: -cv2.contourArea(r.astype(np.float32)))
        keep = []
        for r in rects:
            if all(iou_obb(r.astype(np.int32), k.astype(np.int32)) < 0.3 for k in keep):
                keep.append(r)
        hole = load_boxes(OUT, iid)
        gt = load_boxes(LAB, iid)
        csv_n = exp.get(iid, 0)
        for r in keep:
            ri = r.astype(np.int32)
            if any(iou_obb(ri, h) >= 0.3 for h in hole):
                continue  # duplicate of existing detection
            best_gt = max((iou_obb(ri, g) for g in gt), default=0.0)
            f = features(mask, mask_d, r)
            f["tp"] = 1 if best_gt >= 0.3 else 0
            f["best_gt"] = best_gt
            f["image_id"] = iid
            f["csv_n"] = csv_n
            f["gt_n"] = len(gt)
            f["gt_incomplete"] = 1 if csv_n > len(gt) else 0
            rows.append(f)
    print(f"elapsed {time.time()-t0:.0f}s, added boxes analyzed: {len(rows)}")
    import csv as _csv
    csv_path = r"D:\src\CodeArts\BRobbDetector\benchmark\lsd_diag_rows.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"rows saved: {csv_path}")
    tp = [r for r in rows if r["tp"] == 1]
    fp = [r for r in rows if r["tp"] == 0]
    print(f"TP-added={len(tp)} FP-added={len(fp)}")

    feats = ["sup_mean", "sup_min", "sup0_mean", "sup0_min", "n_edges_sup30",
             "n_edges_sup50", "aspect", "area", "interior_red_density"]
    print(f"\n{'feature':22s} {'TP_mean':>10s} {'FP_mean':>10s}")
    for f in feats:
        tpm = np.mean([r[f] for r in tp]) if tp else 0
        fpm = np.mean([r[f] for r in fp]) if fp else 0
        print(f"{f:22s} {tpm:10.3f} {fpm:10.3f}")

    # threshold scan on sup0_mean
    print("\nThreshold scan (keep if feature >= thr):")
    for f in ["sup0_mean", "sup0_min", "sup_mean", "n_edges_sup30", "interior_red_density"]:
        vals = sorted(set(round(r[f], 2) for r in rows))
        print(f"  {f}:")
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            ktp = sum(1 for r in tp if r[f] >= thr)
            kfp = sum(1 for r in fp if r[f] >= thr)
            pct = 100 * ktp / max(ktp + kfp, 1)
            print(f"    thr={thr:.1f}: kept TP={ktp} FP={kfp} prec={pct:.0f}%")

    print("\nAspect scan (keep if aspect <= thr):")
    for thr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        ktp = sum(1 for r in tp if r["aspect"] <= thr)
        kfp = sum(1 for r in fp if r["aspect"] <= thr)
        pct = 100 * ktp / max(ktp + kfp, 1)
        print(f"    thr={thr:.1f}: kept TP={ktp} FP={kfp} prec={pct:.0f}%")

    print("\nAspect distribution FP  :", np.percentile([r["aspect"] for r in fp], [10, 25, 50, 75, 90]).round(2) if fp else "n/a")
    print("Aspect distribution TP  :", np.percentile([r["aspect"] for r in tp], [10, 25, 50, 75, 90]).round(2) if tp else "n/a")

    print("\nCombo scan (aspect<=A AND min_side>=B):")
    for A in [2.0, 2.5, 3.0, 3.5]:
        for B in [20, 30, 40, 50]:
            ktp = sum(1 for r in tp if r["aspect"] <= A and r["min_side"] >= B)
            kfp = sum(1 for r in fp if r["aspect"] <= A and r["min_side"] >= B)
            pct = 100 * ktp / max(ktp + kfp, 1)
            print(f"    A<={A:.1f} side>={B}: TP={ktp} FP={kfp} prec={pct:.0f}%")

    print("\nFill scan (keep if fill_rot <= thr):")
    for thr in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        ktp = sum(1 for r in tp if r["fill_rot"] <= thr)
        kfp = sum(1 for r in fp if r["fill_rot"] <= thr)
        pct = 100 * ktp / max(ktp + kfp, 1)
        print(f"    thr={thr:.2f}: kept TP={ktp} FP={kfp} prec={pct:.0f}%")

    print("\nHole scan (keep if hole_frac >= thr):")
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
        ktp = sum(1 for r in tp if r["hole_frac"] >= thr)
        kfp = sum(1 for r in fp if r["hole_frac"] >= thr)
        pct = 100 * ktp / max(ktp + kfp, 1)
        print(f"    thr={thr:.1f}: kept TP={ktp} FP={kfp} prec={pct:.0f}%")

    print("\nCombo (aspect<=3.0 AND fill<=0.3):")
    for A in [2.5, 3.0, 4.0]:
        for F in [0.2, 0.25, 0.3]:
            ktp = sum(1 for r in tp if r["aspect"] <= A and r["fill_rot"] <= F)
            kfp = sum(1 for r in fp if r["aspect"] <= A and r["fill_rot"] <= F)
            pct = 100 * ktp / max(ktp + kfp, 1)
            print(f"    A<={A:.1f} F<={F:.2f}: TP={ktp} FP={kfp} prec={pct:.0f}%")
    print("fill_rot FP dist:", np.percentile([r["fill_rot"] for r in fp], [10,25,50,75,90]).round(3) if fp else "n/a")
    print("fill_rot TP dist:", np.percentile([r["fill_rot"] for r in tp], [10,25,50,75,90]).round(3) if tp else "n/a")

    print("\n--- GT-incompleteness breakdown of FP-added ---")
    inc = [r for r in fp if r["gt_incomplete"] == 1]
    comp = [r for r in fp if r["gt_incomplete"] == 0]
    print(f"FP-added total={len(fp)}: in GT-incomplete images={len(inc)} ({100*len(inc)//max(len(fp),1)}%), in complete images={len(comp)}")
    print(f"TP-added in GT-incomplete images={sum(1 for r in tp if r['gt_incomplete']==1)}")
    # of the FP-added in complete images, how many have strong full-edge support (likely real box)
    strong = [r for r in comp if r["sup_mean"] >= 0.9 and r["min_side"] >= 30]
    print(f"FP-added in COMPLETE images with strong support & side>=30 (likely genuine FP): {len(strong)}")


if __name__ == "__main__":
    main()