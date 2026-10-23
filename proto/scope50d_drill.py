"""LOOP-50d: numeric + visual drill-down for one FN case."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, _holes_of, _hole_quad
from scope43_fn_forensics import RAW

iid = sys.argv[1]
gi = int(sys.argv[2])
CFG = dict(HOLE_DEFAULT)

gt = load_boxes_any(RAW, iid)
img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
m = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
m = denoise(m, P["denoise_min_area"])
holes = _holes_of(m, CFG)
g = np.asarray(gt[gi], np.int32)
print("GT", g.tolist(), "area", cv2.contourArea(g))

cnt = np.zeros(m.shape, np.uint8)
cv2.fillPoly(cnt, [g], 1)
print("mask area inside GT:", int(cnt.sum()), "of", m.shape)

for i, h in enumerate(holes):
    pts = h.reshape(-1, 2).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, m.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, m.shape[0] - 1)
    f = float(cnt[pts[:, 1], pts[:, 0]].mean())
    if f <= 0.001:
        continue
    q = _hole_quad(h, m, CFG)
    rect = cv2.minAreaRect(h)
    fill = cv2.contourArea(h) / max(rect[1][0] * rect[1][1], 1e-6)
    print(f"hole {i}: area={cv2.contourArea(h):8.1f} frac_in_gt={f:.3f} "
          f"fill={fill:.2f} bbox={cv2.boundingRect(h)} quad="
          f"{None if q is None else q.astype(int).tolist()}")

x0, y0 = g.min(axis=0)
x1, y1 = g.max(axis=0)
pad = 30
X0, Y0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
X1, Y1 = min(int(x1) + pad, img.shape[1]), min(int(y1) + pad, img.shape[0])
vis = img[Y0:Y1, X0:X1].copy()
vis[m[Y0:Y1, X0:X1] > 0] = (0, 0, 255)
for h in holes:
    cv2.drawContours(vis, [h - np.array([X0, Y0])], -1, (0, 255, 255), 1)
cv2.polylines(vis, [g - np.array([X0, Y0])], True, (0, 255, 0), 1)
out = os.path.join(r"D:\src\CodeArts\BRobbDetector\tmp_fn50",
                   f"d_{iid}_{gi}.png")
cv2.imwrite(out, cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print("saved", out)