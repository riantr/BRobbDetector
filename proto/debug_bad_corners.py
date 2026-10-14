"""Analyze bad corners: where m8 distance > 10px, is GT on red mask?
Visualize a few examples to understand the discrepancy."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb
from eval_pointmatch import load_boxes, corner_match_distance, PERMS_4

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def best_corner_assignment(det_box, gt_box):
    best_perm = None
    best_max = 1e18
    for perm in PERMS_4:
        dists = [np.linalg.norm(det_box[i] - gt_box[perm[i]]) for i in range(4)]
        if max(dists) < best_max:
            best_max = max(dists)
            best_perm = perm
            best_dists = dists
    return best_perm, best_dists


def main():
    ids = get_ids()
    exp = load_expected()

    # For bad corners (m8 dist > 10px), check if GT is on red mask
    bad_gt_on_red = 0
    bad_gt_off_red = 0
    bad_m8_on_red = 0
    bad_m8_off_red = 0
    bad_dists = []
    bad_examples = []

    # Distance from GT corner to nearest red pixel
    gt_to_red_dists = []

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        det = [d.astype(np.int32) for d in det]
        mask = red_mask(img)
        mask_d2 = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask_d5 = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)

        # Distance transform of non-red mask (distance to nearest red pixel)
        non_red = (mask == 0).astype(np.uint8)
        dist_to_red = cv2.distanceTransform(non_red, cv2.DIST_L2, 3)

        for di, dbox in enumerate(det):
            best_gj = -1
            best_iou = 0
            for gj, gbox in enumerate(gt):
                iou = iou_obb(dbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gj = gj
            if best_gj < 0 or best_iou < 0.3:
                continue

            gbox = gt[best_gj]
            perm, dists = best_corner_assignment(dbox, gbox)

            for i in range(4):
                gt_i = gbox[perm[i]]
                m8_i = dbox[i]
                d = dists[i]

                # Distance from GT corner to nearest red pixel
                gy, gx = int(gt_i[1]), int(gt_i[0])
                if 0 <= gy < dist_to_red.shape[0] and 0 <= gx < dist_to_red.shape[1]:
                    gt_to_red = dist_to_red[gy, gx]
                else:
                    gt_to_red = 999
                gt_to_red_dists.append(gt_to_red)

                if d > 10:
                    bad_dists.append(d)
                    if mask_d5[gy, gx] > 0:
                        bad_gt_on_red += 1
                    else:
                        bad_gt_off_red += 1
                    my, mx = int(m8_i[1]), int(m8_i[0])
                    if mask_d5[my, mx] > 0:
                        bad_m8_on_red += 1
                    else:
                        bad_m8_off_red += 1

                    if len(bad_examples) < 15 and d > 15:
                        bad_examples.append({
                            "iid": iid, "corner": i, "dist": round(d, 1),
                            "gt_to_red": round(gt_to_red, 1),
                            "gt": (int(gt_i[0]), int(gt_i[1])),
                            "m8": (int(m8_i[0]), int(m8_i[1])),
                        })

    gt_to_red_dists = np.array(gt_to_red_dists)

    print("=" * 70)
    print(f"Total corners analyzed: {len(gt_to_red_dists)}")
    print(f"\nGT corner distance to nearest red pixel:")
    print(f"  median={np.median(gt_to_red_dists):.1f}  mean={gt_to_red_dists.mean():.1f}  p75={np.percentile(gt_to_red_dists, 75):.1f}  p90={np.percentile(gt_to_red_dists, 90):.1f}")
    print(f"  0px: {(gt_to_red_dists <= 0).sum()}  <=1px: {(gt_to_red_dists <= 1).sum()}  <=2px: {(gt_to_red_dists <= 2).sum()}  <=3px: {(gt_to_red_dists <= 3).sum()}  <=5px: {(gt_to_red_dists <= 5).sum()}")

    print(f"\nBad corners (m8 dist > 10px): {len(bad_dists)}")
    print(f"  GT on red mask (±5px): {bad_gt_on_red} ({bad_gt_on_red/len(bad_dists)*100:.1f}%)")
    print(f"  GT off red mask:        {bad_gt_off_red} ({bad_gt_off_red/len(bad_dists)*100:.1f}%)")
    print(f"  m8 on red mask (±5px):  {bad_m8_on_red} ({bad_m8_on_red/len(bad_dists)*100:.1f}%)")
    print(f"  m8 off red mask:        {bad_m8_off_red} ({bad_m8_off_red/len(bad_dists)*100:.1f}%)")

    print(f"\nBad corner distance distribution:")
    bd = np.array(bad_dists)
    print(f"  median={np.median(bd):.1f}  mean={bd.mean():.1f}  p25={np.percentile(bd, 25):.1f}  p75={np.percentile(bd, 75):.1f}")

    print(f"\nExamples (dist > 15px):")
    for ex in bad_examples:
        print(f"  {ex['iid']} corner{ex['corner']}: dist={ex['dist']} gt_to_red={ex['gt_to_red']} gt={ex['gt']} m8={ex['m8']}")


if __name__ == "__main__":
    main()