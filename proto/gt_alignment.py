"""Decisive test: do the candidate GT boxes actually correspond to red annotation
boxes in the benchmark IMAGES?

For each candidate GT dir, for each GT box, measure the red support of the box
edges in the corresponding image (Image@CARS_202609_annotated) using the same
heuristic as eval_pointmatch.red_support. A GT whose boxes do not sit on red
pixels in the image is misaligned (wrong coordinate frame / different image set).

Also compares the raw vs annotated image files for the same id (dimensions +
mean abs pixel diff) to see whether they are the same picture.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask
from eval_pointmatch import parse_points, red_support
from canny_quad import get_ids

CANDIDATES = {
    "Compare-V1@annotated": os.path.join(DATA, "Compare-V1@CARS_202609_annotated"),
    "Compare@annotated":    os.path.join(DATA, "Compare@CARS_202609_annotated"),
    "Label@raw":            os.path.join(DATA, "Label@CARS_202609_raw"),
}
IMG_RAW = os.path.join(DATA, "Image@CARS_202609_raw")


def load_boxes(d, iid):
    p = os.path.join(d, iid + ".json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def main():
    ids = get_ids()

    print("=== image file comparison (raw vs annotated, same id) ===")
    for iid in ids[:8]:
        a = os.path.join(IMG_RAW, iid + ".jpg")
        b = os.path.join(IMG, iid + ".jpg")
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ia, ib = cv2.imread(a), cv2.imread(b)
        if ia is None or ib is None:
            continue
        same_shape = ia.shape == ib.shape
        if same_shape:
            d = float(np.mean(np.abs(ia.astype(float) - ib.astype(float))))
        else:
            d = float('nan')
        print(f"  {iid}: raw{ia.shape[:2]} ann{ib.shape[:2]} meanAbsDiff={d:.2f}")

    print("\n=== GT red-support in Image@CARS_202609_annotated ===")
    for name, d in CANDIDATES.items():
        sups = []
        for iid in ids:
            img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
            if img is None:
                continue
            mask = red_mask(img)
            mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
            for b in load_boxes(d, iid):
                sups.append(red_support(b.astype(np.int32), mask_d))
        sups = np.array(sups)
        if not len(sups):
            continue
        print(f"  {name:22s} n={len(sups):4d}  mean={sups.mean():.3f}  "
              f">=0.30:{100*np.mean(sups>=0.30):5.1f}%  >=0.50:{100*np.mean(sups>=0.50):5.1f}%  "
              f">=0.70:{100*np.mean(sups>=0.70):5.1f}%  <0.10:{100*np.mean(sups<0.10):5.1f}%")


if __name__ == "__main__":
    main()