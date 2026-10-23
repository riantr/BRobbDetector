"""LOOP-45 diagnostic: the red stroke profile perpendicular to each GT side.

For every GT box side we sample the red mask at perpendicular offsets
s = -15..+15 px and print the red fraction.  This tells us exactly where
the GT side line sits relative to the hand-drawn stroke:
    inner edge / stroke centre / outer edge
so the hole expansion can be calibrated correctly.

Usage: python proto/scope45_profile.py [--n 12]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, red_mask
from canny_quad import get_ids, denoise
from cv_scope34 import load_boxes_any
from scope45_holes import P, _side_geom
from scope43_fn_forensics import RAW

ROOT = r"D:\src\CodeArts\BRobbDetector"


def profile(mask, gt, side):
    a, u, n, L = _side_geom(np.asarray(gt, float), side)
    ns = max(int(L), 10)
    ts = np.linspace(0.1, 0.9, ns)
    ss = np.arange(-15.0, 15.5, 1.0)
    H, W = mask.shape
    base = a[None, :] + ts[:, None] * (u * L)[None, :]
    pts = base[:, None, :] + ss[None, :, None] * n[None, None, :]
    xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
    return (mask[ys, xs] > 0).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--mask", default="prod", choices=["prod", "loose"])
    args = ap.parse_args()

    ids = get_ids()
    shown = 0
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        gt = load_boxes_any(RAW, iid)
        if not gt:
            continue
        if args.mask == "prod":
            mask = red_mask(img, P["sat_min"], P["val_min"], P["hue_max"])
        else:
            mask = red_mask(img, 60, 80, 8)
        mask = denoise(mask, P["denoise_min_area"])
        print(f"\n=== {iid}  ({len(gt)} GT) ===")
        for gi, g in enumerate(gt[:4]):
            print(f"  GT#{gi}:")
            for s in range(4):
                fr = profile(mask, g, s)
                bar = "".join("." if f < 0.15 else ("-" if f < 0.5 else "#")
                              for f in fr)
                # index of the peak
                pk = int(np.argmax(fr))
                print(f"    side{s}: {bar}   peak@s={int(np.arange(-15,16)[pk]):+d}")
            shown += 1
            if shown >= args.n:
                return


if __name__ == "__main__":
    main()