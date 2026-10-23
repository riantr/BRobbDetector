"""LOOP-45/46: hole-based box recovery + tunable pipeline.

LOOP-43/44 showed that essentially every false negative comes from a red
component that is too big: one connected blob holds two or more GT boxes
(or one box plus junk), and the pipeline can only emit ONE quad per blob.

Key observation: a hand-drawn red rectangle encloses a HOLE - the region
inside the stroke.  The interior of a box is background, so `findContours`
with RETR_CCOMP reports it as a child contour of the box's component.
Two boxes that merely touch keep their interiors separate, so the HOLES
give us the boxes directly - no component splitting needed.

The hole boundary is the inner edge of the stroke; the GT side line runs
through the stroke CENTRE (measured in scope45_profile.py: red-fraction
peak at s = -1..0, band ~[-3,+3]).  So each hole side is refined by a
total-least-squares fit over the red band just outside the hole.

Usage:
  python proto/scope45_holes.py --strategy both
  python proto/scope46_tune.py --axis hole_band
"""
import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import DATA, IMG, red_mask, load_expected
from eval_pointmatch import match_boxes_point, is_real_unlabeled
from canny_quad import DEFAULT as CQ_DEFAULT, get_ids, denoise, _side_ok, _order_quad
from cv_hybrid import (FIT_DEFAULT, MASK_DEFAULT, PROD_DEFAULT, outer_contours,
                       _edgefit_corners, dedup_keep_order)
from quad_ortho import is_orthogonal_quad, quad_on_boundary
from cv_scope34 import load_boxes_any, ortho_edge, ortho_blend

ROOT = r"D:\src\CodeArts\BRobbDetector"
RAW = os.path.join(DATA, "Label@CARS_202609_raw")
DIST_VALUES = [3, 5, 8, 12]

P = dict(CQ_DEFAULT)
P.update(FIT_DEFAULT)
P.update(MASK_DEFAULT)
P.update(PROD_DEFAULT)

HOLE_DEFAULT = {
    "hole_scan": 16.0,      # max stroke half-width to scan outside a hole side
    "hole_fill": 0.55,      # min contourArea/rectArea for a hole to be rectangular
    "hole_min_side": 14.0,  # min hole side (the box is slightly larger)
    "hole_min_comp_area": 60.0,
    "hole_expand": 2.7,     # fallback stroke half-width (measured median)
    "band_frac": 0.5,       # fraction of the side that must be red
    "hole_band": 3.0,       # band OUTSIDE the hole side that holds the stroke
    "hole_pts": 12,         # min red points on a side to fit its line
    "hole_fit_pass": 2,     # extra robust passes of the side fit
    "hole_fit_tol": 1.5,    # residual (px) that counts as an outlier
    "hole_overshoot": 0.06,  # how far past the hole corner to sample
    "min_red_support": 0.5,  # min fraction of the quad perimeter that is red
    "hole_group_fill": 0.60,  # min fill ratio when merging holes of one box
    "hole_group_mode": "off",  # off | replace | extra
    "hole_quad_mode": "v1",  # v1 = red band fit, v2 = contour fit
    "hole_off": 2.0,         # v2: outward offset of the contour line (stroke half width)
    "hole_contour_frac": 0.08,  # v2: fraction of the side dropped at the corners
    "hole_contour_pts": 8,   # v2: min contour points per side
    "hole_refine": 1,        # 1 = re-fit the 4 sides on the red stroke
    "hole_refine_band": 3.0,  # half width of the symmetric refine band
    "hole_refine_pts": 8,    # min red points on a side of the refine band
    "hole_refine_tol": 3.0,  # max 90th percentile residual of the refine band
}

PIPE_DEFAULT = {
    "strategy": "both",     # holes | edgefit | both
    "ortho": "ortho_edge",  # none | ortho_edge | ortho_blend
    "gate_first": 1,        # apply the orthogonality gate before enforcement
    "dedup_iou": 0.40,
}


# ----------------------------------------------------------------- helpers

def _line_isect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], float)
    b = np.asarray(p2, float) - np.asarray(p1, float)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(A, b)
    return np.asarray(p1, float) + t[0] * d1


def _side_geom(pts, i):
    """(anchor, unit dir, outward normal, length) of side i of a convex quad."""
    a = np.asarray(pts[i], float)
    b = np.asarray(pts[(i + 1) % 4], float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    c = np.asarray(pts, float).mean(axis=0)
    if float(np.dot(c - a, n)) > 0:
        n = -n
    return a, u, n, L


def _fit_side_line(mask, a, u, n, L, cfg):
    """TLS line through the red stroke just OUTSIDE a hole side.

    The band can contain red pixels of ANOTHER stroke (a crossing stroke, a
    neighbouring box, red text).  Those outliers drag the total-least-squares
    fit onto the wrong line, so the fit is repeated with the outliers removed.
    """
    ns = max(int(L * 1.5), 20)
    ts = np.linspace(-cfg["hole_overshoot"], 1.0 + cfg["hole_overshoot"], ns)
    ss = np.arange(0.0, cfg["hole_band"] + 0.5, 1.0)
    H, W = mask.shape
    base = a[None, :] + ts[:, None] * (u * L)[None, :]
    pts = base[:, None, :] + ss[None, :, None] * n[None, None, :]
    xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
    sel = mask[ys, xs] > 0
    if int(sel.sum()) < cfg["hole_pts"]:
        return None
    P = pts[sel].astype(float)
    mu = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
    d = vt[0]
    if float(np.dot(d, u)) < 0:
        d = -d
    for _ in range(int(cfg.get("hole_fit_pass", 0))):
        nn = np.array([-d[1], d[0]])
        resid = np.abs((P - mu) @ nn)
        keep = resid <= cfg.get("hole_fit_tol", 1.5)
        if int(keep.sum()) < cfg["hole_pts"]:
            break
        P = P[keep]
        mu = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        d = vt[0]
        if float(np.dot(d, u)) < 0:
            d = -d
    return mu, d


def _hole_quad(hole, mask, cfg):
    """Recover the box that bounds a hole of the red mask."""
    area = cv2.contourArea(hole)
    if area < 20:
        return None
    rect = cv2.minAreaRect(hole)
    (cx, cy), (rw, rh), _ = rect
    if rw < cfg["hole_min_side"] or rh < cfg["hole_min_side"]:
        return None
    if area / max(rw * rh, 1e-6) < cfg["hole_fill"]:
        return None                      # not a rectangle -> not a box interior
    corners = _order_quad(cv2.boxPoints(rect).astype(float))
    lines = []
    for i in range(4):
        g = _side_geom(corners, i)
        if g is None:
            return None
        a, u, n, L = g
        fit = _fit_side_line(mask, a, u, n, L, cfg)
        if fit is None:
            lines.append((a + cfg["hole_expand"] * n, u))
        else:
            mu, d = fit
            if float(np.dot(d, u)) < 0:
                d = -d
            lines.append((mu, d))
    q = []
    for i in range(4):
        x = _line_isect(*lines[i - 1], *lines[i])
        if x is None:
            return None
        q.append(x)
    q = np.array(q, float)
    if not np.isfinite(q).all():
        return None
    return q


def _contour_sides(hole, corners, frac=0.08):
    """Group the hole's contour points by side (corner regions dropped)."""
    pts = hole.reshape(-1, 2).astype(np.float64)
    c = np.asarray(corners, np.float64)
    frames = []
    for i in range(4):
        a = c[i]
        b = c[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return None
        u = d / L
        n = np.array([-u[1], u[0]])
        frames.append((a, u, n, L))
    dist = np.empty((len(pts), 4))
    proj = np.empty((len(pts), 4))
    for i, (a, u, n, L) in enumerate(frames):
        rel = pts - a[None, :]
        proj[:, i] = rel @ u
        dist[:, i] = rel @ n
    owner = np.argmin(np.abs(dist), axis=1)
    groups = [[] for _ in range(4)]
    for i, (a, u, n, L) in enumerate(frames):
        sel = owner == i
        t = proj[sel, i]
        keep = (t > frac * L) & (t < (1.0 - frac) * L)
        groups[i].append(pts[sel][keep])
    return groups, frames


def _hole_quad_v2(hole, mask, cfg):
    """Hole quad from the CONTOUR of the hole instead of the red band.

    The red band just outside the hole edge can contain other strokes (a
    crossing stroke, a neighbouring box, red text) and then the total-least-
    squares fit of that side is pulled onto the wrong line.  The hole's own
    contour is unambiguous: it is exactly the inner edge of the box's stroke.
    Each side is fitted on its contour points and then moved outwards by
    half the stroke width, because the GT side line runs through the stroke
    CENTRE.
    """
    area = cv2.contourArea(hole)
    if area < 20:
        return None
    rect = cv2.minAreaRect(hole)
    (cx, cy), (rw, rh), _ = rect
    if rw < cfg["hole_min_side"] or rh < cfg["hole_min_side"]:
        return None
    if area / max(rw * rh, 1e-6) < cfg["hole_fill"]:
        return None
    corners = _order_quad(cv2.boxPoints(rect).astype(float))
    gr = _contour_sides(hole, corners, cfg.get("hole_contour_frac", 0.08))
    if gr is None:
        return None
    groups, frames = gr
    ctr = np.asarray(hole, np.float64).reshape(-1, 2).mean(axis=0)
    lines = []
    for i, (a, u, n, L) in enumerate(frames):
        P = groups[i][0]
        if len(P) < cfg.get("hole_contour_pts", 8):
            return None
        mu = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        d = vt[0]
        if float(np.dot(d, u)) < 0:
            d = -d
        nn = np.array([-d[1], d[0]])
        if float(np.dot(nn, ctr - mu)) > 0:      # must point AWAY from the hole
            nn = -nn
        fit = _fit_side_line(mask, mu, d, nn, L, cfg)
        if fit is None:
            lines.append((mu + cfg.get("hole_off", 2.0) * nn, d))
        else:
            fm, fd = fit
            if float(np.dot(fd, d)) < 0:
                fd = -fd
            lines.append((fm, fd))
    q = []
    for i in range(4):
        x = _line_isect(*lines[i - 1], *lines[i])
        if x is None:
            return None
        q.append(x)
    q = np.array(q, float)
    if not np.isfinite(q).all():
        return None
    return q


def _quad_of(hole, mask, cfg):
    """Quad of a hole - the contour based fit (v2) or the red band fit (v1)."""
    if cfg.get("hole_quad_mode", "v1") == "v2":
        return _hole_quad_v2(hole, mask, cfg)
    return _hole_quad(hole, mask, cfg)



def _holes_of(mask, cfg):
    """All hole contours of the red mask (parent component area filtered).

    OpenCV's hierarchy is [next, previous, first_child, parent], so the
    siblings of a hole are chained through hier[i][0] - walking hier[i][2]
    would only ever yield the FIRST hole of a component.
    """
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        if cv2.contourArea(c) < cfg["hole_min_comp_area"]:
            continue
        child = hier[i][2]
        while child != -1:
            out.append(cnts[child])
            child = hier[child][0]
    return out


# ----------------------------------------------------------------- detectors

def _refine_sides(mask, q, band=4.0, min_pts=8, tol=2.0):
    """Re-fit the 4 sides of a quad on the red stroke centred on each side.

    The hole quad is built from the hole's minAreaRect: its orientation and
    its side offsets inherit the error of the minAreaRect fit.  A second pass
    that fits each side on the red pixels in a symmetric band around the side
    line and re-intersects the sides removes most of that error.
    """
    H, W = mask.shape
    lines = []
    for i in range(4):
        a = np.asarray(q[i], float)
        b = np.asarray(q[(i + 1) % 4], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return None
        u = d / L
        n = np.array([-u[1], u[0]])
        ts = np.linspace(0.0, 1.0, max(int(L), 8))
        ss = np.arange(-band, band + 0.5, 1.0)
        base = a[None, :] + ts[:, None] * d[None, :]
        pts = base[:, None, :] + ss[None, :, None] * n[None, None, :]
        xs = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
        sel = mask[ys, xs] > 0
        if int(sel.sum()) < min_pts:
            return None
        P = pts[sel].astype(float)
        mu = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        dd = vt[0]
        if float(np.dot(dd, u)) < 0:
            dd = -dd
        # reject the fit if the band is not a single stroke
        resid = np.abs((P - mu) @ n)
        if float(np.percentile(resid, 90)) > tol:
            return None
        lines.append((mu, dd))
    out = []
    for i in range(4):
        x = _line_isect(*lines[i - 1], *lines[i])
        if x is None:
            return None
        out.append(x)
    res = np.array(out, float)
    if not np.isfinite(res).all():
        return None
    return res


def _enforce(q, mode):
    if mode == "none":
        return np.asarray(q, float)
    if mode == "ortho_edge":
        return ortho_edge(np.asarray(q, float)).astype(float)
    if mode == "ortho_blend":
        return ortho_blend(np.asarray(q, float), 0.5).astype(float)
    return np.asarray(q, float)


def _gate(q, W, H, p):
    if q is None or not _side_ok(q, p):
        return None
    if not is_orthogonal_quad(q, W, H, tol=p["ortho_tol"],
                              edge_band=p.get("edge_band", 0.0),
                              waive_boundary=False):
        return None
    return q


def _on_edge(q, W, H, p):
    """True when the REPORTED rectangle has a vertex on the image edge.

    The rule (2026-10-21) applies to the rectangle that is emitted, i.e. to
    the rounded corners, not only to the float quad that entered the gate.
    """
    return quad_on_boundary(q, W, H, p.get("edge_band", 0.0))


def detect_edgefit(img, p, cfg, pipe, mask=None):
    """pm_v7 baseline: one edgefit quad per outer contour."""
    H, W = img.shape[:2]
    if mask is None:
        mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
    out = []
    for c in outer_contours(mask, p):
        q = _edgefit_corners(c, p)
        if pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        else:
            if q is None or not _side_ok(q, p):
                continue
        q = _enforce(q, pipe["ortho"])
        if _on_edge(q, W, H, p):
            continue
        if not pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        out.append(np.asarray(q, np.int32))
    return out


def _red_support(mask, quad, step=2):
    """Fraction of the quad perimeter that is red in the mask."""
    q = np.asarray(quad, float)
    H, W = mask.shape
    tot = 0
    red = 0
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        n = max(int(L / step), 2)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        red += int(np.count_nonzero(mask[ys, xs] > 0))
        tot += n
    return red / max(tot, 1)


def _hole_side_lines(hole, mask, cfg):
    """The 4 fitted side lines of a hole: [(point, dir), ...] in quad order."""
    rect = cv2.minAreaRect(hole)
    corners = _order_quad(cv2.boxPoints(rect).astype(float))
    lines = []
    for i in range(4):
        g = _side_geom(corners, i)
        if g is None:
            return None
        a, u, n, L = g
        fit = _fit_side_line(mask, a, u, n, L, cfg)
        if fit is None:
            lines.append((a + cfg["hole_expand"] * n, u))
        else:
            mu, d = fit
            if float(np.dot(d, u)) < 0:
                d = -d
            lines.append((mu, d))
    return corners, lines


def _line_angle(d):
    a = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    return a


def _ang_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _same_line(p1, d1, p2, d2, ang_tol=4.0, dist_tol=5.0):
    """True iff the two infinite lines are (nearly) the same line."""
    if _ang_diff(_line_angle(d1), _line_angle(d2)) > ang_tol:
        return False
    n = np.array([-d1[1], d1[0]])
    return abs(float(np.dot(p2 - p1, n))) <= dist_tol


def _shared_lines(lines_a, lines_b):
    """Number of (nearly) collinear line pairs between two holes."""
    n = 0
    for p1, d1 in lines_a:
        for p2, d2 in lines_b:
            if _same_line(p1, d1, p2, d2):
                n += 1
                break
    return n


def _group_quad(corners_list, lines_list, areas, mask, cfg):
    """Quad for a group of holes that belong to the SAME box.

    When a crossing stroke splits a box interior into several pieces, the
    pieces keep the box's side lines.  The box is then the intersection of
    the OUTERMOST parallel lines of the group - the crossing strip's own
    edges lie strictly inside and are ignored.

    Two DIFFERENT boxes must never be merged, so the merged rectangle has to
    be filled by the pieces: a box split by a thin strip is ~85% filled,
    two boxes glued side by side are far below that.
    """
    ref = lines_list[0]
    d0 = ref[0][1]
    d1 = ref[1][1]
    n0 = np.array([-d0[1], d0[0]])
    n1 = np.array([-d1[1], d1[0]])
    c = np.mean([c.mean(axis=0) for c in corners_list], axis=0)

    cls0, cls1 = [], []
    for lines in lines_list:
        for p, d in lines:
            off = float(np.dot(p - c, n1))
            if _ang_diff(_line_angle(d), _line_angle(d0)) <= 45.0:
                cls0.append((off, p, d))
            else:
                cls1.append((float(np.dot(p - c, n0)), p, d))
    if len(cls0) < 2 or len(cls1) < 2:
        return None
    cls0.sort(key=lambda t: t[0])
    cls1.sort(key=lambda t: t[0])
    a0, b0 = cls0[0], cls0[-1]
    a1, b1 = cls1[0], cls1[-1]
    q = []
    for (_, pa, da), (_, pb, db) in ((a0, a1), (a1, b0), (b0, b1), (b1, a0)):
        x = _line_isect(pa, da, pb, db)
        if x is None:
            return None
        q.append(x)
    q = np.array(q, float)
    fill = sum(areas) / max(abs(cv2.contourArea(q.astype(np.float32))), 1e-6)
    if fill < cfg.get("hole_group_fill", 0.0):
        return None
    return q


def detect_holes(img, p, cfg, pipe, mask=None):
    """One quad per hole of the red mask (holes of the same box are merged)."""
    H, W = img.shape[:2]
    if mask is None:
        mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
    holes = _holes_of(mask, cfg)
    infos = []
    for h in holes:
        r = _hole_side_lines(h, mask, cfg)
        if r is not None:
            infos.append((h, r[0], r[1]))
    if not infos:
        return []

    # group holes that share side lines -> they are pieces of the same box
    n = len(infos)
    mode = cfg.get("hole_group_mode", "off")
    groups = collections.OrderedDict()
    if mode != "off":
        # O(n^2) union-find: only worth it when the groups are actually used
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if _shared_lines(infos[i][2], infos[j][2]) >= 2:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
    else:
        groups = {i: [i] for i in range(n)}

    out = []
    cands = []
    for gidx in groups.values():
        if len(gidx) == 1:
            cands.append(_quad_of(infos[gidx[0]][0], mask, cfg))
            continue
        gq = _group_quad([infos[k][1] for k in gidx],
                         [infos[k][2] for k in gidx],
                         [cv2.contourArea(infos[k][0]) for k in gidx],
                         mask, cfg)
        if mode == "replace":
            cands.append(gq)
        else:                       # "extra": keep the pieces AND the merge
            for k in gidx:
                cands.append(_quad_of(infos[k][0], mask, cfg))
            cands.append(gq)
    for q in cands:
        if q is None:
            continue
        if cfg.get("hole_refine", 0):
            r = _refine_sides(mask, q, band=cfg.get("hole_refine_band", 4.0),
                              min_pts=int(cfg.get("hole_refine_pts", 8)),
                              tol=cfg.get("hole_refine_tol", 2.0))
            if r is not None:
                q = r
        if pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        else:
            if not _side_ok(q, p):
                continue
        q = _enforce(q, pipe["ortho"])
        if _on_edge(q, W, H, p):
            continue
        if not pipe.get("gate_first", 1):
            q = _gate(q, W, H, p)
            if q is None:
                continue
        if cfg.get("min_red_support", 0.0) > 0:
            if _red_support(mask, q) < cfg["min_red_support"]:
                continue
        out.append(np.asarray(q, np.int32))
    return out


def detect(img, p, cfg, pipe, mask=None):
    """Full pipeline.  `mask` overrides the red mask (colour-agnostic mode).

    When a mask is supplied it is used as-is (the caller owns the denoise);
    otherwise the red mask is built and denoised once here and shared by both
    families, which is what the C# port does.
    """
    if mask is None:
        mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
        if p["denoise_min_area"] > 0:
            mask = denoise(mask, p["denoise_min_area"])
    strat = pipe["strategy"]
    raw = []
    if strat in ("holes", "both"):
        raw += detect_holes(img, p, cfg, pipe, mask)
    if strat in ("edgefit", "both"):
        raw += detect_edgefit(img, p, cfg, pipe, mask)
    return dedup_keep_order(raw, pipe.get("dedup_iou", p["dedup_iou"]))


# ----------------------------------------------------------------- evaluation

mask_cache = {}


def evaluate(ids, gt_map, det_map, exp):
    res = {d: {"tp": 0, "raw_fp": 0, "true_fp": 0, "real_unl": 0} for d in DIST_VALUES}
    ndet = 0
    for iid in ids:
        det = [np.asarray(d, np.int32) for d in det_map.get(iid, [])]
        gt = gt_map[iid]
        ndet += len(det)
        mask_d = mask_cache.get(iid)
        csv_n = exp.get(iid, 0)
        for D in DIST_VALUES:
            tp, unmatched = match_boxes_point(det, gt, float(D))
            res[D]["tp"] += tp
            res[D]["raw_fp"] += len(unmatched)
            for di in unmatched:
                if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                    res[D]["real_unl"] += 1
                else:
                    res[D]["true_fp"] += 1
    total_gt = sum(len(v) for v in gt_map.values())
    out = {}
    for D in DIST_VALUES:
        r = res[D]
        rec = r["tp"] / max(total_gt, 1) * 100
        prec = r["tp"] / max(r["tp"] + r["true_fp"], 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[str(D)] = dict(r, recall=round(rec, 1), precision=round(prec, 1),
                           corrected_f1=round(f1, 1))
    return ndet, total_gt, out


def load_all():
    ids = get_ids()
    gt_map = {iid: load_boxes_any(RAW, iid) for iid in ids}
    exp = load_expected()
    return ids, gt_map, exp


def run(ids, gt_map, exp, cfg, pipe, p=None):
    p = p or P
    det_map = {}
    t0 = time.time()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            det_map[iid] = []
            continue
        if iid not in mask_cache:
            mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                          np.ones((3, 3), np.uint8), iterations=3) > 0
        det_map[iid] = detect(img, p, cfg, pipe)
    ndet, total_gt, per_D = evaluate(ids, gt_map, det_map, exp)
    return ndet, total_gt, per_D, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="both", choices=["holes", "edgefit", "both"])
    ap.add_argument("--ortho", default="ortho_edge",
                    choices=["none", "ortho_edge", "ortho_blend"])
    ap.add_argument("--cfg", default="")
    args = ap.parse_args()

    cfg = dict(HOLE_DEFAULT)
    for kv in filter(None, args.cfg.split(",")):
        k, v = kv.split("=")
        try:
            cfg[k] = float(v)
        except ValueError:
            cfg[k] = v
    pipe = dict(PIPE_DEFAULT)
    pipe["strategy"] = args.strategy
    pipe["ortho"] = args.ortho

    ids, gt_map, exp = load_all()
    print(f"LOOP-45 strategy={args.strategy} ortho={args.ortho} on {len(ids)} images, "
          f"{sum(len(v) for v in gt_map.values())} GT boxes")
    ndet, total_gt, per_D, dt = run(ids, gt_map, exp, cfg, pipe)
    print(f"\ndet={ndet}  ({dt:.0f}s)")
    for D in DIST_VALUES:
        r = per_D[str(D)]
        print(f"  D={D:2d}  TP={r['tp']:4d}/{total_gt}  trueFP={r['true_fp']:4d} "
              f"real_unl={r['real_unl']:3d}  corr_f1={r['corrected_f1']:.1f}")

    out = os.path.join(ROOT, "benchmark", f"scope45_{args.strategy}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"strategy": args.strategy, "pipe": pipe, "hole_cfg": cfg,
                   "params": {k: P[k] for k in ("hue_max", "sat_min", "val_min",
                                                "denoise_min_area", "min_area",
                                                "min_side", "max_side", "max_aspect",
                                                "ortho_tol", "boundary_band",
                                                "dedup_iou", "fit_band",
                                                "fit_min_pts", "fit_jump")},
                   "det": ndet, "total_gt": total_gt, "per_D": per_D},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()