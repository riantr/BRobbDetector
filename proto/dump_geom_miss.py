"""Print the GT / detection corner coordinates of the geometric near-misses."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, RAW, detect
from eval_pointmatch import corner_match_distance
from quad_ortho import quad_on_boundary

ROOT = r"D:\src\CodeArts\BRobbDetector"

ids = get_ids()
cases = []
for iid in ids:
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    if img is None:
        continue
    H, W = img.shape[:2]
    dets = [np.asarray(q, np.int32) for q in
            detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
    gts = [np.asarray(g, np.int32) for g in load_boxes_any(RAW, iid)
           if not quad_on_boundary(g, W, H, 0.0)]
    for gi, g in enumerate(gts):
        if not dets:
            continue
        ds = [corner_match_distance(d, g) for d in dets]
        k = int(np.argmin(ds))
        if 5 < ds[k] <= 60:
            cases.append((ds[k], iid, gi, g, dets[k], len(gts), len(dets)))
cases.sort(key=lambda t: -t[0])
for cd, iid, gi, g, d, ng, nd in cases[:25]:
    print("%-28s gt%d/%d  cd=%.1f" % (iid, gi, ng, cd))
    print("   gt  %s" % np.asarray(g).tolist())
    print("   det %s" % np.asarray(d).tolist())