"""Which candidate GT agrees with the authoritative annotation record?

The detail CSVs (2026-09-*_detail.csv) record how many annotations exist per
image. That count is independent of every candidate GT dir, so it is the
arbiter. We compare, per benchmark image:

    csv_expected  vs  len(raw)  vs  len(Compare-V1)  vs  len(detections)

We also re-measure red support with a TIGHT dilation (1 iteration instead of
3) so a box only counts as "drawn" if a red line really runs along its edge.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import parse_points, red_support
from canny_quad import get_ids
from cv_scope34 import load_boxes_any, DET_DIR

RAW = os.path.join(DATA, "Label@CARS_202609_raw")
CMP1 = os.path.join(DATA, "Compare-V1@CARS_202609_annotated")
CMP = os.path.join(DATA, "Compare@CARS_202609_annotated")


def tight_support(box, mask_d):
    """Same as red_support but on a 1-iteration dilated mask (tighter)."""
    return red_support(box.astype(np.int32), mask_d)


def main():
    ids = get_ids()
    exp = load_expected()

    rows = []
    for iid in ids:
        n_csv = exp.get(iid, 0)
        rows.append((iid, n_csv,
                     len(load_boxes_any(RAW, iid)),
                     len(load_boxes_any(CMP1, iid)),
                     len(load_boxes_any(CMP, iid)),
                     len(load_boxes_any(DET_DIR, iid))))

    print("=== per-image count agreement with the annotation CSV (500 imgs) ===")
    for label, idx in [("raw", 2), ("Compare-V1", 3), ("Compare", 4), ("det", 5)]:
        diff = np.array([r[idx] - r[1] for r in rows])
        exact = int(np.sum(diff == 0))
        within1 = int(np.sum(np.abs(diff) <= 1))
        print(f"  {label:12s} total={int(np.array([r[idx] for r in rows]).sum()):5d}  "
              f"csv={sum(r[1] for r in rows):5d}  "
              f"exact={exact:4d}/500  within1={within1:4d}/500  "
              f"mean_err={diff.mean():+.2f}  over={int((diff>0).sum()):3d} under={int((diff<0).sum()):3d}")

    print("\n=== tight red support (1-iteration dilate) of GT boxes ===")
    for label, d in [("raw", RAW), ("Compare-V1", CMP1)]:
        sups = []
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            md = cv2.dilate(red_mask(img), np.ones((3, 3), np.uint8), iterations=1)
            for b in load_boxes_any(d, iid):
                sups.append(tight_support(b, md))
        s = np.array(sups)
        print(f"  {label:12s} n={len(s):4d} mean={s.mean():.3f} "
              f">=0.8:{100*np.mean(s>=0.8):5.1f}%  >=0.5:{100*np.mean(s>=0.5):5.1f}%  "
              f"<0.2:{100*np.mean(s<0.2):5.1f}%")

    print("\n=== sample: images where raw and Compare-V1 disagree most ===")
    rows.sort(key=lambda r: -abs(r[2] - r[3]))
    for r in rows[:8]:
        print(f"  {r[0]}  csv={r[1]}  raw={r[2]}  cmp1={r[3]}  det={r[5]}")


if __name__ == "__main__":
    main()