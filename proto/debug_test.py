import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import red_mask
from detectors import remove_text, line_open, detect_holes, dedup_contained, nms

TEST = os.path.join(os.path.dirname(__file__), "..", "data", "Image@test")
for name in ["20260905_055321_325", "20260905_055520_184"]:
    img = cv2.imread(os.path.join(TEST, name + ".jpg"))
    if img is None:
        print(f"{name}: cannot read")
        continue
    mask = red_mask(img)
    px = np.count_nonzero(mask)
    m1 = remove_text(mask)
    px1 = np.count_nonzero(m1)
    m2 = line_open(m1, length=15, step=15)
    px2 = np.count_nonzero(m2)
    rects = detect_holes(m2, min_area=300, min_side=12, min_fill=0.50, min_support=0.50)
    rects = nms(rects, 0.4)
    rects = dedup_contained(rects)
    print(f"{name}: mask={px} noText={px1} lineOpen={px2} holes={len(rects)}")
    for r in rects:
        c = cv2.boxPoints(r).astype(int)
        print(f"  rect: {c.tolist()}")

    cv2.imwrite(f"debug_{name}_mask.png", mask)
    cv2.imwrite(f"debug_{name}_notext.png", m1)
    cv2.imwrite(f"debug_{name}_lineopen.png", m2)