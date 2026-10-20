"""Run pm_v5_2 detection on all images in the real data directory.
Skips images that already have output (resume from C# CLI partial run).

Usage: python proto/run_v52_full.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import DATA
from canny_quad import get_ids, to_json, denoise
from cv_hybrid import outer_contours, _pass, dedup_keep_order, ROOT
from cv_scope4 import _base_quads, HY_ORDER
from cv_scope7 import pm5_params
from common import red_mask

IMG_DIR = os.path.join(DATA, "Image@CARS_202609_annotated")
OUT_DIR = os.path.join(DATA, "Label@CARS_202609_annotated")


def v52_params():
    p = pm5_params()
    p["min_area"] = 1500.0
    p["smin"] = 80
    p["vmin"] = 110
    p["ortho_tol"] = 20.0
    p["boundary_band"] = 20.0
    p["dedup_iou"] = 0.75
    return p


def detect(img, p):
    H, W = img.shape[:2]
    mask = red_mask(img, p["smin"], p["vmin"])
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    cnts = outer_contours(mask, p)
    all_quads = []
    for c in cnts:
        base = _base_quads(c, p)
        for m in HY_ORDER:
            q = _pass(base[m], W, H, p)
            if q is not None:
                all_quads.append(q)
    return dedup_keep_order(all_quads, p["dedup_iou"])


def main():
    p = v52_params()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_imgs = sorted(f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    existing = set(f[:-5] for f in os.listdir(OUT_DIR) if f.endswith(".json"))
    todo = [iid for iid in all_imgs if iid not in existing]

    print(f"pm_v5_2 full run: {len(all_imgs)} total, {len(existing)} done, "
          f"{len(todo)} todo")

    t0 = time.time()
    n_box = 0
    for n, iid in enumerate(todo):
        img = cv2.imread(os.path.join(IMG_DIR, iid + ".jpg"))
        if img is None:
            continue
        det = detect(img, p)
        n_box += len(det)
        out = {"img_name": iid + ".jpg",
               "infos": [{"points": ";".join(f"{int(x)},{int(y)}" for x, y in q),
                          "type": ""} for q in det]}
        with open(os.path.join(OUT_DIR, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        if (n + 1) % 5000 == 0:
            dt = time.time() - t0
            print(f"  [{n+1}/{len(todo)}] boxes={n_box} "
                  f"({dt:.0f}s, {dt/(n+1)*1000:.1f}ms/img)", flush=True)

    dt = time.time() - t0
    total = len(existing) + len(todo)
    print(f"\nDone: {len(todo)} processed, {n_box} boxes in {dt:.0f}s")
    print(f"Total: {total} images, output at {OUT_DIR}")


if __name__ == "__main__":
    main()