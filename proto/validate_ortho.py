"""LOOP-2 validation: orthogonality rule on real data.

1) Apply the rule to the production hole-detector output (m8) and confirm the
   recall delta is < 0.5% (production boxes are minAreaRect -> already orthogonal).
2) Measure how many LSD raw candidate quads the rule rejects (with vs without).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb
from quad_ortho import is_orthogonal_quad, interior_angles
from lsd_rect import lsd_segments, assemble_rects, DEFAULT, parse_points, load_boxes

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
BENCH = r"D:\src\CodeArts\BRobbDetector\benchmark"
M8 = os.path.join(BENCH, "m8_growbox_v3_final.json")
IOU_THRESHOLD = 0.3


def match_boxes(det, gt, thr=IOU_THRESHOLD):
    used = set(); tp = 0; iou_sum = 0.0
    for db in det:
        best = 0.0; bj = -1
        for j, gb in enumerate(gt):
            if j in used:
                continue
            v = iou_obb(db, gb)
            if v > best:
                best = v; bj = j
        if bj >= 0 and best >= thr:
            used.add(bj); tp += 1; iou_sum += best
    return tp, len(det) - tp, len(gt) - tp, iou_sum


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids)
    return ids[:500]


def part1_production():
    print("=" * 60)
    print("PART 1: orthogonality rule on production (m8) detections")
    print("=" * 60)
    ids = get_ids()
    TP = FP = FN = 0; iou_sum = 0.0; matched = 0
    n_boxes = 0; n_rejected = 0; rejected_nonortho = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        det = load_boxes(OUT, iid)
        gt = load_boxes(LAB, iid)
        kept = []
        for b in det:
            n_boxes += 1
            if is_orthogonal_quad(b, W, H, tol=15.0):
                kept.append(b)
            else:
                n_rejected += 1
                rejected_nonortho += 1
        tp, fp, fn, s = match_boxes(kept, gt)
        TP += tp; FP += fp; FN += fn; iou_sum += s; matched += tp
    # baseline
    with open(M8, encoding="utf-8") as f:
        base = json.load(f)["box_level"]
    recall = TP / (TP + FN) * 100 if (TP + FN) else 0
    fpr = FP / (TP + FP) * 100 if (TP + FP) else 0
    prec = TP / (TP + FP) * 100 if (TP + FP) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    print(f"  boxes checked: {n_boxes}, rejected: {n_rejected}")
    print(f"  baseline m8: TP={base['tp']} FP={base['fp']} FN={base['fn']} R={base['recall']} FPR={base['fpr']} F1={base['f1']}")
    print(f"  filtered   : TP={TP} FP={FP} FN={FN} R={recall:.1f} FPR={fpr:.1f} F1={f1:.1f}")
    drecall = recall - base["recall"]
    print(f"  recall delta = {drecall:+.1f}  -> {'PASS' if drecall >= -0.5 else 'FAIL'} (gate >= -0.5)")
    return drecall


def part2_lsd_rejection(n_images=120):
    print("\n" + "=" * 60)
    print(f"PART 2: LSD raw candidate rejection ({n_images} images)")
    print("=" * 60)
    ids = get_ids()[:n_images]
    p_no = dict(DEFAULT); p_no["use_ortho"] = False
    p_yes = dict(DEFAULT); p_yes["use_ortho"] = True
    tot_raw = tot_pass = 0
    devs = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        segs = lsd_segments(mask, DEFAULT["min_len"])
        raw = assemble_rects(segs, W, H, mask_d, p_no)
        ok = assemble_rects(segs, W, H, mask_d, p_yes)
        tot_raw += len(raw); tot_pass += len(ok)
        for r in raw:
            angs = interior_angles(r)
            devs.append(max(abs(a - 90.0) for a in angs if np.isfinite(a)))
    rej = tot_raw - tot_pass
    print(f"  raw candidates (no ortho filter): {tot_raw}")
    print(f"  passed ortho filter             : {tot_pass}")
    print(f"  rejected by ortho rule          : {rej} ({100*rej/max(tot_raw,1):.1f}%)")
    if devs:
        devs = np.array(devs)
        print(f"  max-angle-deviation percentiles [50,75,90,95,99] deg: "
              f"{np.percentile(devs, [50,75,90,95,99]).round(1)}")
    return tot_raw, tot_pass


def main():
    d = part1_production()
    part2_lsd_rejection()
    print("\n" + "=" * 60)
    print(f"LOOP-2 GATE: production recall delta {d:+.1f} >= -0.5  -> "
          f"{'PASS' if d >= -0.5 else 'FAIL'}")
    print("(Production boxes are minAreaRect rectangles, so the rule is a no-op;")
    print(" it actively prunes non-orthogonal line-assembled quads in PART 2.)")
    print("=" * 60)


if __name__ == "__main__":
    main()