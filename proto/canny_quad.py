"""Canny-line quad detector (new detection approach).

Pipeline (per user's spec):
  1. Canny line search     : red mask -> Canny -> HoughLinesP segments
  2. Close lines to shapes : draw segments, bridge gaps (morph close + endpoint
                             linking), so intersecting lines close into blobs
  3. Select near-quad      : findContours -> approxPolyDP -> keep 4-gons
  4. Adjacent-edge ortho   : keep only quads whose adjacent edges are ~90 deg

Usage:
  python proto/canny_quad.py --viz <image_id>          # visualize one image
  python proto/canny_quad.py --run                     # detect 500 ids -> tmp_canny_out
  python proto/canny_quad.py --run --set key=value ... # override params
"""
import sys, os, json, argparse, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import IMG, LAB, red_mask, iou_obb, load_expected
from quad_ortho import is_orthogonal_quad
from lsd_rect import (assemble_rects, dedup_rects, DEFAULT as LR_DEFAULT,
                      angle_diff_mod180)

OUT_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_canny_out"
VIZ_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_canny_viz"

DEFAULT = {
    "denoise_min_area": 40,    # drop red components smaller than this before Canny
    "canny_lo": 50,
    "canny_hi": 150,
    "dilate_mask": 1,          # dilate red mask before Canny (bridge hand-drawn gaps)
    "hough_thr": 20,
    "hough_min_len": 15,
    "hough_max_gap": 15,
    "line_thick": 3,
    "close_len": 9,            # morph close kernel (bridges corner gaps)
    "link_dist": 18,           # max endpoint distance to link two segments
    "approx_eps": 0.03,        # approxPolyDP epsilon as fraction of perimeter
    "approx_adaptive": 1,      # try increasing eps until exactly 4 vertices
    "approx_eps_list": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18],
    "use_centerline": 1,       # average outer+inner ring contours -> stroke centerline
    "min_area": 900.0,
    "min_side": 15.0,
    "max_side": 800.0,
    "max_aspect": 8.0,
    "ortho_tol": 20.0,
    "boundary_band": 6.0,
    "dedup_iou": 0.30,
    "merge_collinear": 1,
    "merge_ang_tol": 4.0,
    "merge_line_tol": 3.0,
    "merge_gap_tol": 20.0,
}


# ---------------- step 0: denoise ----------------

def denoise(mask, min_area):
    """Keep only red connected components with area >= min_area."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    keep = np.zeros(n, np.uint8)
    keep[1:] = np.where(stats[1:, 4] >= min_area, 255, 0)
    return keep[lab]


# ---------------- step 1: Canny line search ----------------

def canny_segments(mask, p):
    m = mask
    if p["dilate_mask"] > 0:
        k = np.ones((3, 3), np.uint8)
        m = cv2.dilate(mask, k, iterations=p["dilate_mask"])
    edges = cv2.Canny(m, p["canny_lo"], p["canny_hi"])
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=p["hough_thr"],
        minLineLength=p["hough_min_len"], maxLineGap=p["hough_max_gap"])
    segs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            segs.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return edges, segs


# ---------------- step 2: close lines into shapes ----------------

def _link_endpoints(segs, link_dist):
    """Merge nearby endpoints, returning segments with snapped/shared endpoints."""
    if not segs:
        return segs
    pts = []
    for a, b in segs:
        pts.append((a, 0, 0))
        pts.append((b, 0, 1))
    n = len(pts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    coords = [np.array(p[0], float) for p in pts]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) <= link_dist:
                union(i, j)
    # cluster centroids
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    centroid = {}
    for root, members in groups.items():
        centroid[root] = np.mean([coords[m] for m in members], axis=0)

    out = []
    for s in range(0, n - 1, 2):
        pa = centroid[find(s)].astype(int)
        pb = centroid[find(s + 1)].astype(int)
        if np.linalg.norm(pa - pb) >= 2:
            out.append((tuple(pa), tuple(pb)))
    return out


def close_into_shapes(segs, shape, p):
    """Draw (endpoint-linked) segments and morph-close gaps to form blobs."""
    canvas = np.zeros(shape[:2], np.uint8)
    linked = _link_endpoints(segs, p["link_dist"])
    for a, b in linked:
        cv2.line(canvas, a, b, 255, p["line_thick"])
    ksz = p["close_len"]
    if ksz >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k)
    return canvas, linked


# ---------------- step 3 + 4: quad selection + ortho filter ----------------

def _order_quad(pts):
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def _align_cycle(o, i):
    """Align inner quad i to outer quad o by best cyclic shift; return i."""
    best, best_d = i, 1e18
    for s in range(4):
        ii = np.roll(i, s, axis=0)
        d = float(np.sum(np.linalg.norm(o - ii, axis=1)))
        if d < best_d:
            best_d, best = d, ii
    return best


def _quad_from_contour(c, eps_frac):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps_frac * peri, True)
    if len(approx) != 4:
        return None
    return approx.reshape(4, 2).astype(float)


def _quad_adaptive(c, eps_list):
    """Try increasing epsilon values until approxPolyDP yields exactly 4 vertices."""
    peri = cv2.arcLength(c, True)
    for e in eps_list:
        ap = cv2.approxPolyDP(c, e * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(float)
    return None


def _side_ok(pts, p):
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    mn, mx = min(sides), max(sides)
    if mn < p["min_side"] or mx > p["max_side"]:
        return False
    if mx / max(mn, 1e-6) > p["max_aspect"]:
        return False
    return True


def quads_from_shapes(closed, mask, img_w, img_h, p):
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    if hierarchy is None:
        return [], mask_d
    hierarchy = hierarchy[0]

    quads = []
    for idx, c in enumerate(contours):
        if cv2.contourArea(c) < p["min_area"]:
            continue
        parent = hierarchy[idx][3]
        if parent != -1:
            continue  # only consider outer (parentless) contours
        outer = _quad_from_contour(c, p["approx_eps"])
        if outer is None:
            continue
        child = hierarchy[idx][2]
        quad = outer
        if p["use_centerline"] and child != -1:
            inner = _quad_from_contour(contours[child], p["approx_eps"])
            if inner is not None:
                o = _order_quad(outer)
                i = _align_cycle(o, _order_quad(inner))
                quad = (o + i) / 2.0
        if not _side_ok(quad, p):
            continue
        if not is_orthogonal_quad(quad, img_w, img_h, tol=p["ortho_tol"],
                                  boundary_band=p["boundary_band"]):
            continue
        quads.append(quad.astype(np.int32))
    return quads, mask_d


def dedup(quads, iou_thr):
    quads = sorted(quads, key=lambda r: -cv2.contourArea(r.astype(np.float32)))
    keep = []
    for r in quads:
        if all(iou_obb(r, k) < iou_thr for k in keep):
            keep.append(r)
    return keep


# ---------------- approach C: line-pair intersection assembly ----------------

def hough_segments(mask, p):
    """Canny + HoughLinesP, returned as lsd_rect-style segment dicts."""
    edges, segs = canny_segments(mask, p)
    out = []
    for (x1, y1), (x2, y2) in segs:
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L < 1:
            continue
        out.append({
            "p1": np.array([x1, y1], float), "p2": np.array([x2, y2], float),
            "mid": np.array([(x1 + x2) / 2, (y1 + y2) / 2], float),
            "dir": np.array([x2 - x1, y2 - y1], float) / L,
            "ang": float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0),
            "length": L,
        })
    return out


def merge_collinear(segs, ang_tol=4.0, line_tol=3.0, gap_tol=20.0):
    """Merge near-collinear, near-coincident overlapping segments."""
    used = [False] * len(segs)
    merged = []
    for i in range(len(segs)):
        if used[i]:
            continue
        si = segs[i]
        chain = [si]
        used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            sj = segs[j]
            if angle_diff_mod180(si["ang"], sj["ang"]) > ang_tol:
                continue
            # perp distance of sj midpoint to si line
            d = abs(si["dir"][0] * (sj["mid"][1] - si["p1"][1]) -
                    si["dir"][1] * (sj["mid"][0] - si["p1"][0]))
            if d > line_tol:
                continue
            # along-axis gap
            u = si["dir"]
            proj = [np.dot(chain[0]["p1"], u), np.dot(chain[0]["p2"], u)]
            a0, a1 = min(proj), max(proj)
            b0, b1 = sorted([np.dot(sj["p1"], u), np.dot(sj["p2"], u)])
            if b0 > a1 + gap_tol or a0 > b1 + gap_tol:
                continue
            chain.append(sj)
            used[j] = True
        if len(chain) > 1:
            pts = np.array([pt for s in chain for pt in (s["p1"], s["p2"])])
            c = pts.mean(axis=0)
            u = si["dir"]
            t = (pts - c) @ u
            lo, hi = t.argmin(), t.argmax()
            p1, p2 = pts[lo], pts[hi]
            L = float(np.linalg.norm(p2 - p1))
            if L < 2:
                merged.append(si)
                continue
            merged.append({
                "p1": p1, "p2": p2, "mid": (p1 + p2) / 2,
                "dir": (p2 - p1) / L,
                "ang": float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0),
                "length": L,
            })
        else:
            merged.append(si)
    return merged


def detect_assemble(img, p):
    """Approach C: Canny -> Hough -> merge -> parallel-pair rectangle assembly."""
    H, W = img.shape[:2]
    mask = red_mask(img)
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    segs = hough_segments(mask, p)
    if p.get("merge_collinear", 1):
        segs = merge_collinear(segs, ang_tol=p.get("merge_ang_tol", 4.0),
                               line_tol=p.get("merge_line_tol", 3.0),
                               gap_tol=p.get("merge_gap_tol", 20.0))
    lr = dict(LR_DEFAULT)
    lr.update({
        "min_len": p["hough_min_len"],
        "min_side": p["min_side"],
        "max_side": p["max_side"],
        "ortho_tol": p["ortho_tol"],
    })
    try:
        cands = assemble_rects(segs, W, H, mask_d, lr)
    except Exception:
        cands = []
    rects = dedup_rects(cands, iou_thr=p["dedup_iou"])
    out = []
    for r in rects:
        if _side_ok(r.astype(float), p):
            out.append(r.astype(np.int32))
    return out


# ---------------- approach B: direct contour on denoised mask ----------------

def detect_contour(img, p):
    """Approach B: denoise -> findContours -> approxPolyDP quad -> ortho filter.

    Uses the OUTER contour (no centerline averaging); GT corners align with the
    outer boundary of the red stroke."""
    H, W = img.shape[:2]
    mask = red_mask(img)
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cnts):
        if cv2.contourArea(c) < p["min_area"]:
            continue
        if hier[i][3] != -1:
            continue  # outer contours only
        if p.get("approx_adaptive", 1):
            q = _quad_adaptive(c, p.get("approx_eps_list",
                                         [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]))
        else:
            q = _quad_from_contour(c, p["approx_eps"])
        if q is None:
            continue
        if not _side_ok(q, p):
            continue
        if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                                  boundary_band=p["boundary_band"]):
            continue
        out.append(q.astype(np.int32))
    return dedup(out, p["dedup_iou"])


def detect(img, params=None, debug=False, method="assemble"):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    if method == "contour":
        quads = detect_contour(img, p)
        if debug:
            mask = red_mask(img)
            if p["denoise_min_area"] > 0:
                mask = denoise(mask, p["denoise_min_area"])
            return quads, {"mask": mask}
        return quads
    if method == "assemble":
        quads = detect_assemble(img, p)
        if debug:
            mask = red_mask(img)
            if p["denoise_min_area"] > 0:
                mask = denoise(mask, p["denoise_min_area"])
            edges, segs = canny_segments(mask, p)
            return quads, {"mask": mask, "edges": edges, "segs": segs}
        return quads

    H, W = img.shape[:2]
    mask = red_mask(img)
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    edges, segs = canny_segments(mask, p)
    closed, linked = close_into_shapes(segs, mask.shape, p)
    quads, mask_d = quads_from_shapes(closed, mask, W, H, p)
    quads = dedup(quads, p["dedup_iou"])
    if debug:
        return quads, {"mask": mask, "edges": edges, "closed": closed, "linked": linked}
    return quads


# ---------------- runner / eval ----------------

def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def to_json(iid, quads):
    infos = []
    for q in quads:
        pts = ";".join(f"{int(x)},{int(y)}" for x, y in q)
        infos.append({"points": pts, "type": ""})
    return {"img_name": iid + ".jpg", "infos": infos}


def run_all(params, out_dir=OUT_DIR, method="assemble"):
    os.makedirs(out_dir, exist_ok=True)
    ids = get_ids()
    total = 0
    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        quads = detect(img, params, method=method)
        total += len(quads)
        with open(os.path.join(out_dir, iid + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_json(iid, quads), f, ensure_ascii=False)
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(ids)}] boxes={total}")
    print(f"Done: {len(ids)} images, {total} boxes -> {out_dir}")


def viz(iid, params, method="assemble"):
    os.makedirs(VIZ_DIR, exist_ok=True)
    img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
    quads, dbg = detect(img, params, debug=True, method=method)
    vis = img.copy()
    for q in quads:
        cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(VIZ_DIR, iid + "_det.jpg"), vis)
    cv2.imwrite(os.path.join(VIZ_DIR, iid + "_edges.jpg"), dbg["edges"])
    cv2.imwrite(os.path.join(VIZ_DIR, iid + "_closed.jpg"), dbg["closed"])

    # draw GT in blue for comparison
    gt_path = os.path.join(LAB, iid + ".json")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8-sig") as f:
            gt = json.load(f)
        for info in gt.get("infos", []):
            pts = np.array([[int(x), int(y)] for x, y in
                            (pt.split(",") for pt in info["points"].split(";"))], np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(VIZ_DIR, iid + "_det_gt.jpg"), vis)

    print(f"{iid}: {len(quads)} quads, {len(dbg['linked'])} linked segments")
    print(f"  viz -> {VIZ_DIR}")


def parse_overrides(kvs):
    p = {}
    for kv in kvs or []:
        k, v = kv.split("=", 1)
        p[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--method", default="assemble", choices=["assemble", "shapes", "contour"])
    ap.add_argument("--set", nargs="*", default=None, help="param overrides k=v")
    args = ap.parse_args()
    params = parse_overrides(args.set)
    if args.viz:
        viz(args.viz, params, method=args.method)
    elif args.run:
        run_all(params, args.out, method=args.method)
    else:
        ap.error("specify --viz <id> or --run")


if __name__ == "__main__":
    main()