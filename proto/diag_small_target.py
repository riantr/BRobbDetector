"""LOOP-32: Diagnostic — why is the right-side small target in
20260905_062030_276.jpg not detected by pm_v5_2?

Traces the image through all 6 pipeline stages, reporting what happens
in the RIGHT half of the image (where the small target is), and tests
alternative params (esp. min_area) to find the blocking stage.

Usage: python proto/diag_small_target.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, iou_obb
from canny_quad import get_ids, denoise, to_json
from cv_hybrid import outer_contours, _pass, dedup_keep_order, ROOT, FIT_DEFAULT
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params

IID = "20260905_062030_276"
IMG_PATH = os.path.join(IMG, IID + ".jpg")


def v52_params():
    p = pm5_params()
    p["min_area"] = 1500.0
    p["smin"] = 80
    p["vmin"] = 110
    p["ortho_tol"] = 20.0
    p["boundary_band"] = 20.0
    p["dedup_iou"] = 0.75
    return p


def comp_stats(mask, label):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    print(f"  {label}: {n-1} components")
    rows = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        rows.append((a, x, y, w, h, cent[i]))
    rows.sort(reverse=True)
    for a, x, y, w, h, c in rows[:8]:
        print(f"    area={a:7d}  bbox=({x},{y},{w}x{h})  center=({c[0]:.0f},{c[1]:.0f})")
    return rows


def main():
    img = cv2.imread(IMG_PATH)
    H, W = img.shape[:2]
    print(f"Image: {IID}  size={W}x{H}")
    print(f"Pipeline detection output: 1 box (left) only\n")

    p = v52_params()

    # ---- Stage 1: red_mask ----
    print("="*72)
    print("STAGE 1: red_mask (smin=80, vmin=110)")
    print("="*72)
    mask = red_mask(img, p["smin"], p["vmin"])
    print(f"  red pixels total: {np.count_nonzero(mask)}")
    # right half
    right = mask[:, W//2:]
    print(f"  red pixels in RIGHT half: {np.count_nonzero(right)}")
    comp_stats(mask, "all components")

    # ---- Stage 2: denoise ----
    print("\n" + "="*72)
    print("STAGE 2: denoise (denoise_min_area=40)")
    print("="*72)
    den = denoise(mask, 40)
    print(f"  red pixels after denoise: {np.count_nonzero(den)}")
    comp_stats(den, "components after denoise")

    # ---- Stage 3: outer_contours at various min_area ----
    print("\n" + "="*72)
    print("STAGE 3: outer_contours — min_area sweep")
    print("="*72)
    for ma in [0, 400, 800, 1000, 1200, 1500, 2000, 3000]:
        p2 = dict(p); p2["min_area"] = float(ma)
        cnts = outer_contours(den, p2)
        # count contours in right half
        right_cnts = 0
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"]/M["m00"]
                if cx > W//2:
                    right_cnts += 1
        print(f"  min_area={ma:5d}: {len(cnts):3d} contours total, "
              f"{right_cnts} in right half")

    # Now trace the small right-side contour at min_area=1500 (pm_v5_2)
    print("\n" + "="*72)
    print("TRACE: right-half contours at min_area=1500 (pm_v5_2)")
    print("="*72)
    cnts = outer_contours(den, p)
    for ci, c in enumerate(cnts):
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        cx = M["m10"]/M["m00"] if M["m00"] > 0 else 0
        cy = M["m01"]/M["m00"] if M["m00"] > 0 else 0
        if cx > W//2:
            x, y, w, h = cv2.boundingRect(c)
            print(f"  RIGHT contour #{ci}: area={area:.0f} bbox=({x},{y},{w}x{h}) "
                  f"center=({cx:.0f},{cy:.0f}) n_pts={len(c)}")

    # ---- Stage 3b: find ALL red contours (no denoise) in right half ----
    print("\n" + "="*72)
    print("STAGE 3b: raw red contours in RIGHT half (no denoise, no min_area)")
    print("="*72)
    raw_cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in raw_cnts:
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = M["m10"]/M["m00"]
        if cx <= W//2:
            continue
        area = cv2.contourArea(c)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(c)
        print(f"  raw right contour: area={area:.0f} bbox=({x},{y},{w}x{h}) "
              f"center=({cx:.0f},{M['m01']/M['m00']:.0f}) n_pts={len(c)}")

    # ---- Stage 1b: check HSV in right half ----
    print("\n" + "="*72)
    print("STAGE 1b: HSV analysis of right-half red-ish pixels")
    print("="*72)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # sample the right-bottom-ish region where target likely is
    # find bright/reddish pixels in right half
    rh = hsv[:, W//2:]
    # hue near 0 or 180 (red)
    redish = ((rh[:,:,0] < 15) | (rh[:,:,0] > 165))
    print(f"  right-half pixels with hue near red: {np.count_nonzero(redish)}")
    if np.count_nonzero(redish) > 0:
        s = rh[:,:,1][redish]; v = rh[:,:,2][redish]
        print(f"  their S: min={s.min()} med={np.median(s):.0f} max={s.max()}")
        print(f"  their V: min={v.min()} med={np.median(v):.0f} max={v.max()}")
    # relax threshold test
    for smin, vmin in [(80,110), (40,60), (20,40), (10,20)]:
        m2 = red_mask(img, smin, vmin)
        rc = np.count_nonzero(m2[:, W//2:])
        print(f"  smin={smin:3d} vmin={vmin:3d}: right-half red pixels = {rc}")

    # ---- Test full pipeline at various min_area ----
    print("\n" + "="*72)
    print("FULL PIPELINE: det count at various min_area")
    print("="*72)
    for ma in [400, 800, 1000, 1200, 1500, 2000]:
        p2 = dict(p); p2["min_area"] = float(ma)
        cnts = outer_contours(den, p2)
        quads = []
        for c in cnts:
            base = _base_quads(c, p2)
            for m in HY_ORDER:
                q = _pass(base[m], W, H, p2)
                if q is not None:
                    quads.append(q)
        final = dedup_keep_order(quads, p2["dedup_iou"])
        # check if any box center in right half
        print(f"  min_area={ma:5d}: {len(final)} detections")
        for q in final:
            cx = q[:,0].mean()
            side = "RIGHT" if cx > W//2 else "left "
            print(f"      {side} box center=({cx:.0f},{q[:,1].mean():.0f}) "
                  f"pts={' '.join(f'{int(x)},{int(y)}' for x,y in q)}")


if __name__ == "__main__":
    main()