"""Render the emulation boxes that the detector missed, with the GT (blue) and
the detection (green), next to the red mask.

Usage: python proto/viz_emul_fn.py [--img-dir ...] [--lab-dir ...] [--out ...]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pointmatch import corner_match_distance
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect
from canny_quad import denoise
from gen_emulation import primary_mask

ROOT = r"D:\src\CodeArts\BRobbDetector"


def side_support(mask, g):
    H, W = mask.shape
    sups = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        L = float(np.linalg.norm(b - a))
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        sups.append(float((mask[ys, xs] > 0).mean()))
    return sups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=os.path.join(ROOT, "data", "Image@emulcal"))
    ap.add_argument("--lab-dir", default=os.path.join(ROOT, "data", "Label@emulcal"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp_emul_fn.png"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--imgs", type=int, default=120)
    ap.add_argument("--color", default="红")
    ap.add_argument("--lo", type=float, default=0.9)
    ap.add_argument("--hi", type=float, default=1.01)
    args = ap.parse_args()

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(args.lab_dir, "*.json")))[:args.imgs]
    tiles = []
    for stem in stems:
        img = cv2.imread(os.path.join(args.img_dir, stem + ".jpg"))
        if img is None:
            continue
        with open(os.path.join(args.lab_dir, stem + ".json"), encoding="utf-8") as f:
            lab = json.load(f)
        boxes = []
        for info in lab.get("infos", []):
            if info.get("type") != args.color:
                continue
            pts = np.array([[int(x), int(y)] for x, y in
                            (p.split(",") for p in info["points"].split(";"))], np.int32)
            boxes.append(pts)
        if not boxes:
            continue
        mask = primary_mask(img, args.color)
        mask = denoise(mask, P["denoise_min_area"])
        dets = [np.asarray(d, np.int32) for d in detect(img, P, HOLE_DEFAULT,
                                                        PIPE_DEFAULT, mask)]
        for g in boxes:
            sups = side_support(mask, g)
            if not (args.lo <= min(sups) < args.hi):
                continue
            best, bd = 1e9, None
            for d in dets:
                dd = corner_match_distance(d, g)
                if dd < best:
                    best, bd = dd, d
            if best <= 5.0:
                continue
            c = g.mean(axis=0)
            rad = int(max(60, np.linalg.norm(g - c[None, :], axis=1).max() + 40))
            px0, py0 = max(0, int(c[0]) - rad), max(0, int(c[1]) - rad)
            px1 = min(img.shape[1], int(c[0]) + rad)
            py1 = min(img.shape[0], int(c[1]) + rad)
            crop = img[py0:py1, px0:px1].copy()
            cv2.polylines(crop, [g - [px0, py0]], True, (255, 0, 0), 1)
            if bd is not None:
                cv2.polylines(crop, [bd - [px0, py0]], True, (0, 255, 0), 1)
            s = 2 if max(crop.shape[:2]) < 320 else 1
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            lbl = "%s sup=%.2f d=%.1f" % (stem[-6:], min(sups), best)
            cv2.putText(crop, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
            if len(tiles) >= args.n:
                break
        if len(tiles) >= args.n:
            break
    if not tiles:
        print("no tiles")
        return 1
    hh = max(t.shape[0] for t in tiles)
    ww = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), 4):
        row = []
        for t in tiles[i:i + 4]:
            pad = np.zeros((hh, ww, 3), np.uint8)
            pad[:t.shape[0], :t.shape[1]] = t
            row.append(pad)
        while len(row) < 4:
            row.append(np.zeros((hh, ww, 3), np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(args.out, grid)
    print("wrote %s (%d tiles)" % (args.out, len(tiles)))
    return 0


if __name__ == "__main__":
    sys.exit(main())