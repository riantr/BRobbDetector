"""Quality check of the emulation dataset (data/Image@emulation + data/Label@emulation).

For every label box we build a mask of its own stroke colour and measure how
much of the rectangle's own sides are actually painted - the same "side support"
metric used for the real data (proto/scope73_ceiling.py::side_support).

A correct emulation box has support ~1.0 on every side.  The script reports the
support distribution, the per-colour breakdown and any broken records.

Usage: python proto/verify_emulation.py [--img-dir ...] [--lab-dir ...]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "Image@emulation")
LAB_DIR = os.path.join(ROOT, "data", "Label@emulation")

COLORS = {
    "红": (0, 0, 255),
    "绿": (0, 255, 0),
    "蓝": (255, 0, 0),
}


def color_mask(img, bgr, tol=150):
    """Mask of pixels that are (approximately) the given stroke colour.

    The tolerance has to be loose: the stroke is antialiased and the frame is
    JPEG compressed, so a pixel on the stroke centre is usually a blend of the
    pure colour and the content underneath it.
    """
    d = np.abs(img.astype(np.int16) - np.array(bgr, np.int16)).sum(axis=2)
    return (d <= tol).astype(np.uint8) * 255


def side_support(mask, g):
    """Per-side coverage of the rectangle's own sides (0..1 each)."""
    H, W = mask.shape
    sups = []
    for i in range(4):
        a = np.asarray(g[i], float)
        b = np.asarray(g[(i + 1) % 4], float)
        L = float(np.linalg.norm(b - a))
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = a[None] + ts[:, None] * (b - a)[None]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        sups.append(float((mask[ys, xs] > 0).mean()))
    return sups


def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in
                     (p.split(",") for p in s.split(";"))], np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--lab-dir", default=LAB_DIR)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
    labs = sorted(glob.glob(os.path.join(args.lab_dir, "*.json")))
    if args.limit:
        imgs = imgs[:args.limit]
        labs = labs[:args.limit]
    print("images: %d   labels: %d" % (len(imgs), len(labs)))
    if len(imgs) != len(labs):
        print("!! image/label count mismatch")
    lab_stems = {os.path.splitext(os.path.basename(p))[0] for p in labs}
    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    print("labels without image: %d   images without label: %d"
          % (len(lab_stems - img_stems), len(img_stems - lab_stems)))

    all_sups = []
    per_color = {}
    n_box = 0
    bad = []
    n_img_no_box = 0
    for p in labs:
        stem = os.path.splitext(os.path.basename(p))[0]
        ip = os.path.join(args.img_dir, stem + ".jpg")
        if not os.path.exists(ip):
            continue
        with open(p, encoding="utf-8") as f:
            lab = json.load(f)
        if lab.get("img_name") != stem + ".jpg":
            bad.append((stem, "img_name mismatch: %s" % lab.get("img_name")))
        infos = lab.get("infos", [])
        if not infos:
            n_img_no_box += 1
        img = cv2.imread(ip)
        for info in infos:
            n_box += 1
            g = parse_points(info["points"])
            cname = info.get("type", "")
            if cname not in COLORS:
                bad.append((stem, "unknown type %r" % cname))
                continue
            m = color_mask(img, COLORS[cname])
            sups = side_support(m, g)
            mn = min(sups)
            all_sups.append(mn)
            per_color.setdefault(cname, []).append(mn)
            if mn < 0.9:
                bad.append((stem, "support %.2f on %s" % (mn, cname)))

    a = np.array(all_sups) if all_sups else np.zeros(1)
    print("\nboxes: %d   images without box: %d" % (n_box, n_img_no_box))
    print("side support  min %.3f  p1 %.3f  p5 %.3f  mean %.4f"
          % (a.min(), np.percentile(a, 1), np.percentile(a, 5), a.mean()))
    print("support >= 0.95: %d/%d (%.1f%%)"
          % ((a >= 0.95).sum(), len(a), 100.0 * (a >= 0.95).mean()))
    for c, v in sorted(per_color.items()):
        v = np.array(v)
        print("  %s: n=%4d  min %.3f  mean %.4f" % (c, len(v), v.min(), v.mean()))
    if bad:
        print("\nPROBLEMS (%d):" % len(bad))
        for b in bad[:20]:
            print("  ", b)
    else:
        print("\nno problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())