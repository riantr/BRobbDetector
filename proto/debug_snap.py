"""Debug corner snapping: analyze why edge-fit snapping hurts.

For IoU-matched boxes, compare:
1. m8 corner distance to GT
2. Whether GT corners are on the red mask
3. Whether m8 corners are on the red mask
4. Edge-fit snapped corner distance to GT
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask
from eval_pointmatch import (load_boxes, corner_match_distance, match_boxes_point,
                             snap_box_edge_fit, refine_box, PERMS_4)

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def best_corner_assignment(det_box, gt_box):
    """Return the best permutation and per-corner distances."""
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

    # Stats
    m8_dists = []        # per-corner distances (m8 vs GT)
    snap_dists = []      # per-corner distances (snap vs GT)
    gt_on_red = []       # is GT corner on red mask? (within 2px)
    m8_on_red = []       # is m8 corner on red mask?
    improved = 0
    worsened = 0
    same = 0
    examples = []

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        det = [d.astype(np.int32) for d in det]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        # For each det box, find best GT match by IoU
        for di, dbox in enumerate(det):
            best_gj = -1
            best_iou = 0
            for gj, gbox in enumerate(gt):
                from common import iou_obb
                iou = iou_obb(dbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gj = gj
            if best_gj < 0 or best_iou < 0.3:
                continue

            gbox = gt[best_gj]
            snapped = snap_box_edge_fit(dbox, mask, search_width=15)

            # Best corner assignments
            _, m8_d = best_corner_assignment(dbox, gbox)
            _, snap_d = best_corner_assignment(snapped, gbox)

            m8_max = max(m8_d)
            snap_max = max(snap_d)

            m8_dists.extend(m8_d)
            snap_dists.extend(snap_d)

            # Check if corners are on red mask
            for i in range(4):
                x, y = int(gbox[i, 0]), int(gbox[i, 1])
                gt_on_red.append(1 if mask_d[y, x] > 0 else 0)
                x, y = int(dbox[i, 0]), int(dbox[i, 1])
                m8_on_red.append(1 if mask_d[y, x] > 0 else 0)

            if snap_max < m8_max - 0.5:
                improved += 1
            elif snap_max > m8_max + 0.5:
                worsened += 1
            else:
                same += 1

            if len(examples) < 10 and m8_max > 5 and m8_max < 30:
                examples.append({
                    "iid": iid, "m8_max": round(m8_max, 1), "snap_max": round(snap_max, 1),
                    "m8_d": [round(d, 1) for d in m8_d],
                    "snap_d": [round(d, 1) for d in snap_d],
                })

    m8_dists = np.array(m8_dists)
    snap_dists = np.array(snap_dists)

    print("=" * 70)
    print(f"Matched boxes: {len(m8_dists) // 4}")
    print(f"\nPer-corner distance (px):")
    print(f"  m8:   median={np.median(m8_dists):.1f}  mean={m8_dists.mean():.1f}  p25={np.percentile(m8_dists, 25):.1f}  p75={np.percentile(m8_dists, 75):.1f}")
    print(f"  snap: median={np.median(snap_dists):.1f}  mean={snap_dists.mean():.1f}  p25={np.percentile(snap_dists, 25):.1f}  p75={np.percentile(snap_dists, 75):.1f}")

    print(f"\nMax-corner-distance per box:")
    print(f"  improved: {improved}  worsened: {worsened}  same: {same}")

    print(f"\nCorner on red mask (within 2px):")
    print(f"  GT corners: {sum(gt_on_red)}/{len(gt_on_red)} ({sum(gt_on_red)/len(gt_on_red)*100:.1f}%)")
    print(f"  m8 corners: {sum(m8_on_red)}/{len(m8_on_red)} ({sum(m8_on_red)/len(m8_on_red)*100:.1f}%)")

    print(f"\nExample boxes (m8_max 5-30px):")
    for ex in examples:
        print(f"  {ex['iid']}: m8_max={ex['m8_max']} snap_max={ex['snap_max']} m8_d={ex['m8_d']} snap_d={ex['snap_d']}")


if __name__ == "__main__":
    main()