"""Explore hard FN cases (no meaningful detection) with LSD on the red mask."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb

OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
BENCH = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
DBG = r"D:\src\CodeArts\BRobbDetector\benchmark\lsd_debug"


def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (p.split(',') for p in s.split(';'))], np.int32)


def load_boxes(d, iid):
    p = os.path.join(d, iid + ".json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def hard_fn_cases():
    with open(BENCH, encoding="utf-8") as f:
        bench = json.load(f)
    cases = []
    for p in bench["per_image"]:
        iid = p["id"]
        if p["fn"] == 0:
            continue
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        for g in gt:
            best = max((iou_obb(d, g) for d in det), default=0.0)
            if best < 0.05:
                cases.append((iid, g, best, det))
    return cases


def lsd_segments(mask, min_len=10):
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(mask)[0]
    segs = []
    if lines is None:
        return segs
    lines = np.asarray(lines).reshape(-1, 4)
    for l in lines:
        x1, y1, x2, y2 = l
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        segs.append((float(x1), float(y1), float(x2), float(y2), float(length), float(ang)))
    return segs


def seg_near_box(segs, box, margin=20):
    """Count segments whose midpoint is within the box (expanded)."""
    x, y, w, h = cv2.boundingRect(box)
    x0, y0, x1, y1 = x - margin, y - margin, x + w + margin, y + h + margin
    out = []
    for s in segs:
        mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        if x0 <= mx <= x1 and y0 <= my <= y1:
            out.append(s)
    return out


def main():
    os.makedirs(DBG, exist_ok=True)
    cases = hard_fn_cases()
    print(f"Hard FN cases: {len(cases)}\n")

    for iid, g, best, det in cases:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        mask = red_mask(img)
        segs = lsd_segments(mask, min_len=10)
        near = seg_near_box(segs, g, margin=25)

        gx, gy, gw, gh = cv2.boundingRect(g)
        print(f"{iid}: GT rect=({gx},{gy},{gw},{gh}) best_iou={best:.3f} "
              f"det_count={len(det)} total_segs={len(segs)} segs_near_gt={len(near)}")

        vis = img.copy()
        cv2.polylines(vis, [g], True, (0, 255, 0), 2)
        for d in det:
            cv2.polylines(vis, [d], True, (255, 0, 0), 1)
        for s in near:
            cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(DBG, iid + "_vis.jpg"), vis)
        cv2.imwrite(os.path.join(DBG, iid + "_mask.jpg"), mask)

    print(f"\nDebug images in {DBG}")


if __name__ == "__main__":
    main()