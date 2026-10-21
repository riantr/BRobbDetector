"""Characterize candidate GT directories for the 500-image benchmark.

For each candidate we report: id coverage, box count, orthogonality deviation
profile, and whether the boxes look hand-drawn (orthogonal) or machine-generated
(skewed). Used to decide which GT to re-baseline against.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import DATA, IMG
from eval_pointmatch import parse_points
from canny_quad import get_ids

CANDIDATES = {
    "Compare-V1@annotated": os.path.join(DATA, "Compare-V1@CARS_202609_annotated"),
    "Compare@annotated":    os.path.join(DATA, "Compare@CARS_202609_annotated"),
    "Label@annotated":      os.path.join(DATA, "Label@CARS_202609_annotated"),
    "Label@raw":            os.path.join(DATA, "Label@CARS_202609_raw"),
}
IMG_RAW = os.path.join(DATA, "Image@CARS_202609_raw")


def load_boxes(d, iid):
    p = os.path.join(d, iid + ".json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def interior_angles(q):
    q = np.asarray(q, float)
    angs = []
    for i in range(4):
        p0, p1, p2 = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angs.append(np.degrees(np.arccos(c)))
    return angs


def main():
    ids = get_ids()
    print(f"benchmark ids: {len(ids)}\n")
    print(f"{'candidate':22s} {'ids':>5s} {'boxes':>6s} {'b/img':>6s} "
          f"{'devMed':>7s} {'devP90':>7s} {'devMax':>7s} {'>10deg%':>8s} {'types'}")
    for name, d in CANDIDATES.items():
        if not os.path.isdir(d):
            print(f"{name:22s}  <missing dir>")
            continue
        n_ids = 0
        devs = []
        types = set()
        for iid in ids:
            bs = load_boxes(d, iid)
            if bs:
                n_ids += 1
            for b in bs:
                for a in interior_angles(b):
                    devs.append(abs(a - 90.0))
                p = os.path.join(d, iid + ".json")
                with open(p, encoding="utf-8-sig") as f:
                    for i in json.load(f).get("infos", []):
                        types.add(i.get("type", ""))
        devs = np.array(devs) if devs else np.array([0.0])
        tot = sum(len(load_boxes(d, i)) for i in ids)
        print(f"{name:22s} {n_ids:5d} {tot:6d} {tot/max(n_ids,1):6.2f} "
              f"{np.median(devs):7.2f} {np.percentile(devs,90):7.2f} {devs.max():7.2f} "
              f"{100*np.mean(devs>10):7.1f}%  {sorted(t for t in types if t)[:4]}")

    # how many benchmark ids exist in the raw image set?
    raw_ids = [i for i in ids if os.path.exists(os.path.join(IMG_RAW, i + ".jpg"))]
    print(f"\nbenchmark ids present in Image@CARS_202609_raw : {len(raw_ids)} / {len(ids)}")
    ann_ids = [i for i in ids if os.path.exists(os.path.join(IMG, i + ".jpg"))]
    print(f"benchmark ids present in Image@CARS_202609_annotated : {len(ann_ids)} / {len(ids)}")
    # are raw and annotated images the same files?
    same = 0
    for i in raw_ids[:50]:
        a = os.path.join(IMG_RAW, i + ".jpg")
        b = os.path.join(IMG, i + ".jpg")
        if os.path.exists(b) and os.path.getsize(a) == os.path.getsize(b):
            same += 1
    print(f"raw/annotated same file size (first {min(50,len(raw_ids))} ids): {same}")


if __name__ == "__main__":
    main()