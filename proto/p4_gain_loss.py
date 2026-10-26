"""Is there ANY upside in the P4 outward-completion move set?

`force` lost 15 TP at D=5.  That net figure hides gains and losses.  If the
move set never produces a TP that the shipped config misses, the direction is
dead outright.  If it does, a selector would have to separate them.
"""
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto"))
from common import IMG
from canny_quad import get_ids
from cv_scope34 import load_boxes_any
from eval_pointmatch import match_boxes_point
from quad_ortho import quad_on_boundary
import p4_outward_complete as P4

RAW = r"E:\Xray\CARS\data\Label@CARS_202609_raw"


def main():
    ids = get_ids()
    gt = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = (img.shape[:2] if img is not None else (0, 0))
        gt[iid] = [g for g in load_boxes_any(RAW, iid)
                   if not quad_on_boundary(g, W, H, 0.0)]

    dets = {}
    for mode in ("base", "force"):
        d = {}
        with Pool(12) as pool:
            for iid, _m, det in pool.imap_unordered(
                    P4._work, [(i, mode) for i in ids], chunksize=3):
                d[iid] = det
        dets[mode] = d
        print("ran", mode)

    def tp_ids(mode, D=5.0):
        out = {}
        for iid in ids:
            t, unmatched = match_boxes_point(dets[mode][iid], gt[iid], D)
            used = set(range(len(dets[mode][iid]))) - set(unmatched)
            out[iid] = used
        return out

    a = tp_ids("base")
    b = tp_ids("force")
    # match by quad identity: a TP is a GT box index; identify TPs by the GT box they matched
    def matched_gt(det, gtq, D=5.0):
        import itertools
        res = []
        for gi, g in enumerate(gtq):
            best = None
            for i in range(len(det)):
                a = np.asarray(det[i], float)[:, None, :]
                b = np.asarray(g, float)[None, :, :]
                dm = np.linalg.norm(a - b, axis=2)          # (4,4)
                worst = min(max(dm[p[0], 0], dm[p[1], 1], dm[p[2], 2], dm[p[3], 3])
                            for p in itertools.permutations(range(4)))
                if worst <= D and (best is None or worst < best):
                    best = worst
            if best is not None:
                res.append(gi)
        return set(res)

    base_gt, force_gt = set(), set()
    gain, loss = 0, 0
    for iid in ids:
        g = gt[iid]
        sb = matched_gt(dets["base"][iid], g)
        sf = matched_gt(dets["force"][iid], g)
        base_gt |= {(iid, x) for x in sb}
        force_gt |= {(iid, x) for x in sf}
        gain += len(sf - sb)
        loss += len(sb - sf)
    print("\nD=5 TP boxes: base=%d  force=%d" % (len(base_gt), len(force_gt)))
    print("  force gains (TP not in base): %d" % gain)
    print("  force losses (base TP not in force): %d" % loss)
    print("  => best possible per-image oracle over {base,force}: %d TP"
          % (len(base_gt | force_gt)))


if __name__ == "__main__":
    main()