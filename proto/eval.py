import os, sys, csv, time
import numpy as np
import cv2
from common import load_expected, red_mask, rect_corners, nms, IMG, LAB
from detectors import (
    detect_baseline, remove_text, line_open, detect_holes, dedup_contained,
    detect_twopass, detect_twopass_components,
)

HA, HS, HF, HSUP = 300, 12, 0.50, 0.50

CONFIGS = [
    ("baseline",       {"mode": "baseline"}),
    ("single",         {"mode": "single"}),
    ("2p_pixel",       {"mode": "2p_pixel"}),
    ("2p_components",  {"mode": "2p_comp"}),
    ("2p_comp_s20",    {"mode": "2p_comp", "max_size": 20}),
    ("2p_comp_s40",    {"mode": "2p_comp", "max_size": 40}),
    ("2p_comp_a300",   {"mode": "2p_comp", "max_area": 300}),
]


def run_pipeline(mask, cfg):
    mode = cfg["mode"]
    if mode == "baseline":
        return len(detect_baseline(mask))
    if mode == "single":
        m = remove_text(mask)
        m = line_open(m, length=15, step=15)
        rects = detect_holes(m, min_area=HA, min_side=HS, min_fill=HF, min_support=HSUP)
        rects = nms(rects, thr=0.4)
        rects = dedup_contained(rects)
        return len(rects)
    if mode == "2p_pixel":
        return len(detect_twopass(mask, min_area=HA, min_side=HS, min_fill=HF, min_support=HSUP))
    if mode == "2p_comp":
        return len(detect_twopass_components(mask, min_area=HA, min_side=HS, min_fill=HF, min_support=HSUP,
                                             max_comp_size=cfg.get("max_size", 30),
                                             max_comp_area=cfg.get("max_area", 600)))
    return 0


def evaluate(n_images=800, seed=0, verbose=False):
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    ids = ids[:n_images]

    diffs = {name: [] for name, _ in CONFIGS}
    t0 = time.time()
    for idx, k in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, k + ".jpg"))
        if img is None:
            continue
        mask = red_mask(img)
        e = exp[k]
        for name, cfg in CONFIGS:
            n = run_pipeline(mask, cfg)
            diffs[name].append(n - e)
        if (idx + 1) % 200 == 0:
            print(f"  ...{idx+1}/{len(ids)}  {time.time()-t0:.0f}s")

    print(f"\nimages={len(diffs['baseline'])}")
    for name, _ in CONFIGS:
        d = np.array(diffs[name])
        exact = np.count_nonzero(d == 0) / len(d) * 100
        over = np.count_nonzero(d > 0)
        under = np.count_nonzero(d < 0)
        print(f"  {name:16s}: exact={exact:5.1f}%  over={over:4d}  under={under:4d}  mean={d.mean():+.2f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    evaluate(n_images=n, verbose=True)
