"""LOOP-1 prototype: LSD line-segment -> rectangle assembly.

Assembles rotated rectangles from LSD segments by pairing roughly-parallel
opposite edges and roughly-perpendicular edge pairs, then validating the
rectangle (convex, side lengths, orthogonality, endpoint proximity).
"""
import sys, os, json, itertools, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, iou_obb
from quad_ortho import is_orthogonal_quad


# ---------------- geometry helpers ----------------

def line_intersection(p1, d1, p2, d2):
    """Intersection of two lines given point+dir. Returns point or None."""
    A = np.array([d1, -d2]).T
    b = p2 - p1
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    t, s = np.linalg.solve(A, b)
    return p1 + t * d1


def angle_diff_mod180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def seg_line_distance(seg, other):
    """Perp distance from seg midpoint to other segment's infinite line."""
    p = seg["mid"]
    o1, d = other["p1"], other["dir"]
    return abs(cross2(d, p - o1))


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def along_overlap(s1, s2):
    """Overlap of two segments projected on s1's direction."""
    u = s1["dir"]
    a1, a2 = np.dot(s1["p1"], u), np.dot(s1["p2"], u)
    b1, b2 = np.dot(s2["p1"], u), np.dot(s2["p2"], u)
    lo = max(min(a1, a2), min(b1, b2))
    hi = min(max(a1, a2), max(b1, b2))
    return max(0.0, hi - lo)


# ---------------- LSD ----------------

def lsd_segments(mask, min_len):
    lsd = cv2.createLineSegmentDetector()
    res = lsd.detect(mask)
    lines = res[0]
    if lines is None:
        return []
    lines = np.asarray(lines).reshape(-1, 4)
    segs = []
    for x1, y1, x2, y2 in lines:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        d = np.array([x2 - x1, y2 - y1], float) / length
        segs.append({
            "p1": np.array([x1, y1], float),
            "p2": np.array([x2, y2], float),
            "mid": np.array([(x1 + x2) / 2, (y1 + y2) / 2], float),
            "dir": d, "ang": ang, "length": length,
        })
    return segs


# ---------------- orthogonality filter (LOOP-2 prereq) ----------------

def check_orthogonal(corners, img_w, img_h, ortho_tol=20.0, boundary_band=6.0):
    """Thin wrapper delegating to the shared rule in quad_ortho (LOOP-2)."""
    return is_orthogonal_quad(corners, img_w, img_h, tol=ortho_tol,
                              boundary_band=boundary_band)


# ---------------- rectangle assembly ----------------

def edge_supports(mask_d, corners, band=3):
    """Fraction of pixels along each rectangle edge that are red (dilated mask)."""
    sup = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        L = np.linalg.norm(q - p)
        ns = max(int(L), 2)
        ts = np.linspace(0, 1, ns)
        pts = p[None, :] + ts[:, None] * (q - p)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, mask_d.shape[1] - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, mask_d.shape[0] - 1)
        red = np.count_nonzero(mask_d[ys, xs])
        sup.append(red / ns)
    return sup


def seg_coverage(seg, cA, cB):
    """Fraction of the corner-to-corner edge [cA,cB] covered by seg's projection."""
    u = cB - cA
    L = np.linalg.norm(u)
    if L < 1e-6:
        return 0.0
    u = u / L
    a0, a1 = 0.0, L
    s0, s1 = np.dot(seg["p1"] - cA, u), np.dot(seg["p2"] - cA, u)
    lo, hi = min(s0, s1), max(s0, s1)
    return max(0.0, min(hi, a1) - max(lo, a0)) / L


def build_rect_from_two_pairs(a1, a2, b1, b2, min_side, max_side):
    c1 = line_intersection(a1["p1"], a1["dir"], b1["p1"], b1["dir"])
    c2 = line_intersection(a1["p1"], a1["dir"], b2["p1"], b2["dir"])
    c3 = line_intersection(a2["p1"], a2["dir"], b2["p1"], b2["dir"])
    c4 = line_intersection(a2["p1"], a2["dir"], b1["p1"], b1["dir"])
    if any(c is None for c in (c1, c2, c3, c4)):
        return None
    corners = np.array([c1, c2, c3, c4], float)

    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    if min(sides) < min_side or max(sides) > max_side:
        return None

    signs = []
    for i in range(4):
        v1 = corners[(i + 1) % 4] - corners[i]
        v2 = corners[(i + 2) % 4] - corners[(i + 1) % 4]
        signs.append(cross2(v1, v2))
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return None

    cov = [
        seg_coverage(a1, c1, c2),
        seg_coverage(b2, c2, c3),
        seg_coverage(a2, c3, c4),
        seg_coverage(b1, c4, c1),
    ]
    return corners, cov


def assemble_rects(segs, img_w, img_h, mask_d, p):
    angle_tol = p["angle_tol"]
    min_side = p["min_side"]
    max_side = p["max_side"]
    min_overlap_frac = p["min_overlap_frac"]
    edge_sup_thr = p["edge_sup_thr"]
    min_edges = p["min_edges"]
    cover_thr = p["cover_thr"]
    use_ortho = p.get("use_ortho", True)

    parallel_pairs = []
    n = len(segs)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = segs[i], segs[j]
            if angle_diff_mod180(si["ang"], sj["ang"]) > angle_tol:
                continue
            d = seg_line_distance(si, sj)
            if d < min_side or d > max_side:
                continue
            ov = along_overlap(si, sj)
            if ov < min_overlap_frac * min(si["length"], sj["length"]):
                continue
            bx0 = min(si["p1"][0], si["p2"][0], sj["p1"][0], sj["p2"][0])
            by0 = min(si["p1"][1], si["p2"][1], sj["p1"][1], sj["p2"][1])
            bx1 = max(si["p1"][0], si["p2"][0], sj["p1"][0], sj["p2"][0])
            by1 = max(si["p1"][1], si["p2"][1], sj["p1"][1], sj["p2"][1])
            parallel_pairs.append((i, j, d, (si["ang"] + sj["ang"]) / 2, (bx0, by0, bx1, by1)))

    cands = []
    seen = set()
    m = len(parallel_pairs)
    for ka in range(m):
        ia1, ia2, da, anga, bba = parallel_pairs[ka]
        for kb in range(ka + 1, m):
            ib1, ib2, db, angb, bbb = parallel_pairs[kb]
            if abs(angle_diff_mod180(anga, angb) - 90.0) > angle_tol:
                continue
            if bba[2] < bbb[0] or bbb[2] < bba[0] or bba[3] < bbb[1] or bbb[3] < bba[1]:
                continue
            ids = tuple(sorted([ia1, ia2, ib1, ib2]))
            if ids in seen:
                continue
            seen.add(ids)
            res = build_rect_from_two_pairs(
                segs[ia1], segs[ia2], segs[ib1], segs[ib2],
                min_side, max_side)
            if res is None:
                continue
            rect, cov = res
            if np.mean(cov) < cover_thr:
                continue
            if use_ortho and not check_orthogonal(rect, img_w, img_h, ortho_tol=p["ortho_tol"]):
                continue
            sup = edge_supports(mask_d, rect)
            if sum(1 for s in sup if s >= edge_sup_thr) < min_edges:
                continue
            cands.append(rect)
    return cands


def dedup_rects(rects, iou_thr=0.3):
    rects = sorted(rects, key=lambda r: -cv2.contourArea(r.astype(np.float32)))
    keep = []
    for r in rects:
        ri = r.astype(np.int32)
        if all(iou_obb(ri, k.astype(np.int32)) < iou_thr for k in keep):
            keep.append(r.astype(np.int32))
    return keep


DEFAULT = {
    "angle_tol": 10.0,
    "min_len": 12.0,
    "min_side": 12.0,
    "max_side": 700.0,
    "min_overlap_frac": 0.5,
    "edge_sup_thr": 0.30,
    "min_edges": 3,
    "cover_thr": 0.55,
    "ortho_tol": 20.0,
    "use_ortho": True,
    "post_aspect_max": 1e9,
    "post_side_min": 0.0,
}


def detect_rects(mask, img_w, img_h, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    segs = lsd_segments(mask, p["min_len"])
    mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    cands = assemble_rects(segs, img_w, img_h, mask_d, p)
    cands = dedup_rects(cands)
    out = []
    for r in cands:
        sides = [np.linalg.norm(r[(i + 1) % 4] - r[i]) for i in range(4)]
        mn, mx = min(sides), max(sides)
        asp = mx / max(mn, 1e-6)
        if asp <= p["post_aspect_max"] and mn >= p["post_side_min"]:
            out.append(r)
    return out


# ---------------- eval on 13 hard cases ----------------

def parse_points(s):
    return np.array([[int(x), int(y)] for x, y in (pt.split(',') for pt in s.split(';'))], np.int32)


def load_boxes(d, iid):
    path = os.path.join(d, iid + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [parse_points(i["points"]) for i in json.load(f).get("infos", [])]


def main():
    BENCH = r"D:\src\CodeArts\BRobbDetector\benchmark\m8_growbox_v3_final.json"
    OUT = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"
    with open(BENCH, encoding="utf-8") as f:
        bench = json.load(f)

    cases = []
    for pi in bench["per_image"]:
        iid = pi["id"]
        if pi["fn"] == 0:
            continue
        gt = load_boxes(LAB, iid)
        det = load_boxes(OUT, iid)
        for g in gt:
            best = max((iou_obb(d, g) for d in det), default=0.0)
            if best < 0.05:
                cases.append((iid, g, det))

    print(f"Hard FN cases: {len(cases)}  params={DEFAULT}\n")
    recovered = 0
    total_spurious = 0
    for iid, g, det in cases:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        H, W = img.shape[:2]
        mask = red_mask(img)
        rects = detect_rects(mask, W, H)
        new = [r for r in rects if all(iou_obb(r, d.astype(np.int32)) < 0.3 for d in det)]
        hit = max((iou_obb(r.astype(np.int32), g) for r in rects), default=0.0)
        if hit >= 0.3:
            recovered += 1
        total_spurious += len(new)
        print(f"{iid}: all_rects={len(rects)} new={len(new)} best_iou_gt={hit:.3f} "
              f"{'<-- RECOVERED' if hit >= 0.3 else ''}")
    print(f"\nRecovered {recovered}/{len(cases)} hard FN; spurious new rects total={total_spurious}")


if __name__ == "__main__":
    main()