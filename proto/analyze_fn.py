import os, sys, json, glob
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import load_expected, IMG, LAB, iou_obb, red_mask

OUT_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
IOU_THRESHOLD = 0.3

def parse_points(s):
    pts = []
    for p in s.split(';'):
        x, y = p.split(',')
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)

def load_gt_boxes(image_id):
    path = os.path.join(LAB, image_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]

def load_det_boxes(image_id):
    path = os.path.join(OUT_DIR, image_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]

def box_props(pts):
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    minSide = min(w, h)
    maxSide = max(w, h)
    aspect = maxSide / minSide if minSide > 0 else 0
    area = cv2.contourArea(pts)
    return {"minSide": round(minSide, 1), "maxSide": round(maxSide, 1),
            "aspect": round(aspect, 2), "area": int(area),
            "cx": round(rect[0][0], 1), "cy": round(rect[0][1], 1)}

def main():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:500]

    fn_cases = []
    for k in ids:
        gt_boxes = load_gt_boxes(k)
        det_boxes = load_det_boxes(k)
        if not gt_boxes:
            continue

        used = set()
        for db in det_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_boxes):
                if j in used:
                    continue
                iou = iou_obb(db, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= IOU_THRESHOLD:
                used.add(best_j)

        for j, gb in enumerate(gt_boxes):
            if j in used:
                continue
            best_iou = 0.0
            best_det = -1
            for di, db in enumerate(det_boxes):
                iou = iou_obb(db, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_det = di

            gt_props = box_props(gb)
            img_path = os.path.join(IMG, k + ".jpg")
            img = cv2.imread(img_path)
            if img is None:
                continue

            x0 = gt_props["cx"] - gt_props["maxSide"]
            y0 = gt_props["cy"] - gt_props["maxSide"]
            x1 = gt_props["cx"] + gt_props["maxSide"]
            y1 = gt_props["cy"] + gt_props["maxSide"]
            x0c = max(0, int(x0)); y0c = max(0, int(y0))
            x1c = min(img.shape[1], int(x1)); y1c = min(img.shape[0], int(y1))
            roi = img[y0c:y1c, x0c:x1c] if x1c > x0c and y1c > y0c else None
            red_density = 0.0
            if roi is not None and roi.size > 0:
                rm = red_mask(roi)
                red_density = round(np.count_nonzero(rm) / roi.size * 100, 2)

            det_props = box_props(det_boxes[best_det]) if best_det >= 0 else None
            det_ratio = 0.0
            if det_props:
                det_ratio = round(det_props["minSide"] / gt_props["minSide"], 3)

            fn_cases.append({
                "image": k,
                "gt": gt_props,
                "best_iou": round(best_iou, 3),
                "has_det": best_det >= 0,
                "det": det_props,
                "det_gt_ratio": det_ratio,
                "red_density_around_gt": red_density,
            })

    fn_cases.sort(key=lambda x: x["best_iou"], reverse=True)

    print(f"Total FN cases: {len(fn_cases)}")
    print(f"\n--- IoU distribution ---")
    bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3)]
    for lo, hi in bins:
        count = sum(1 for c in fn_cases if lo <= c["best_iou"] < hi)
        print(f"  IoU [{lo:.2f}, {hi:.2f}): {count}")
    no_det = sum(1 for c in fn_cases if not c["has_det"])
    print(f"  No detection at all: {no_det}")

    print(f"\n--- GT box size distribution ---")
    minSides = [c["gt"]["minSide"] for c in fn_cases]
    print(f"  minSide: min={min(minSides):.1f} median={np.median(minSides):.1f} max={max(minSides):.1f}")
    maxSides = [c["gt"]["maxSide"] for c in fn_cases]
    print(f"  maxSide: min={min(maxSides):.1f} median={np.median(maxSides):.1f} max={max(maxSides):.1f}")
    aspects = [c["gt"]["aspect"] for c in fn_cases]
    print(f"  aspect:  min={min(aspects):.2f} median={np.median(aspects):.2f} max={max(aspects):.2f}")

    print(f"\n--- Det/GT size ratio (when det exists) ---")
    ratios = [c["det_gt_ratio"] for c in fn_cases if c["has_det"]]
    if ratios:
        print(f"  det_gt_ratio: min={min(ratios):.3f} median={np.median(ratios):.3f} max={max(ratios):.3f}")
        print(f"  <0.5 (strip-like): {sum(1 for r in ratios if r < 0.5)}")
        print(f"  0.5-0.8: {sum(1 for r in ratios if 0.5 <= r < 0.8)}")
        print(f"  >=0.8: {sum(1 for r in ratios if r >= 0.8)}")

    print(f"\n--- Red density around GT ---")
    densities = [c["red_density_around_gt"] for c in fn_cases]
    print(f"  min={min(densities):.2f} median={np.median(densities):.2f} max={max(densities):.2f}")
    print(f"  <1%: {sum(1 for d in densities if d < 1)}")
    print(f"  1-5%: {sum(1 for d in densities if 1 <= d < 5)}")
    print(f"  >=5%: {sum(1 for d in densities if d >= 5)}")

    print(f"\n--- Top 20 near-miss FN (highest IoU below threshold) ---")
    for c in fn_cases[:20]:
        det_str = f"det(min={c['det']['minSide']}, ratio={c['det_gt_ratio']})" if c['det'] else "no det"
        print(f"  {c['image'][:25]} iou={c['best_iou']:.3f} gt(min={c['gt']['minSide']}, asp={c['gt']['aspect']}) {det_str} red={c['red_density_around_gt']}%")

    with open(r"D:\src\CodeArts\BRobbDetector\benchmark\fn_detail.json", "w", encoding="utf-8") as f:
        json.dump(fn_cases, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detailed FN analysis to benchmark/fn_detail.json")

if __name__ == "__main__":
    main()