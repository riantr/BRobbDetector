import os, sys, time
import numpy as np
import cv2
from common import red_mask, rect_corners, nms, IMG, LAB


# ---------------- baseline (current C# algorithm) ----------------
def detect_baseline(mask, min_area=600, min_span=25, max_fill=0.35, max_frac=0.5, eps=0.02, close_it=2):
    H, W = mask.shape
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    any_kept = False
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or w < min_span or h < min_span:
            continue
        if a / (w * h) > max_fill:
            continue
        if w * h > max_frac * W * H:
            continue
        keep[lab == i] = 255
        any_kept = True
    if not any_kept:
        return []
    closed = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_it)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area * 2:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        pts = approx.reshape(-1, 2)
        if len(pts) > 4:
            approx = cv2.approxPolyDP(hull, eps * 1.8 * peri, True)
            pts = approx.reshape(-1, 2)
        if len(pts) > 4:
            pts = cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
        if len(pts) != 4:
            continue
        out.append(pts)
    return out


# ---------------- text removal (vectorized) ----------------
def remove_text(mask, glyph_max=28, glyph_area=450, min_cluster=2, join_x=21, join_y=7):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    glyph_ids = [i for i in range(1, n)
                 if max(stats[i][2], stats[i][3]) <= glyph_max and stats[i][4] <= glyph_area]
    if not glyph_ids:
        return mask.copy()

    glyph_mask = np.isin(lab, glyph_ids).astype(np.uint8) * 255
    glyph_set = np.zeros(n, bool)
    glyph_set[glyph_ids] = True

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (join_x, join_y))
    clusters = cv2.dilate(glyph_mask, k)
    nc, clab, cstats, ccent = cv2.connectedComponentsWithStats(clusters, 8)

    remove = np.zeros(n, bool)
    for c in range(1, nc):
        sel = (clab == c) & (lab > 0)
        if not sel.any():
            continue
        ids_in = np.unique(lab[sel])
        ids_in = ids_in[(ids_in >= 0) & (ids_in < n)]
        members = ids_in[glyph_set[ids_in]]
        if len(members) >= min_cluster:
            remove[members] = True

    if not remove.any():
        return mask.copy()

    out = mask.copy()
    out[np.isin(lab, np.nonzero(remove)[0])] = 0
    return out


# ---------------- line opening (keep long straight strokes) ----------------
def line_kernel(length, angle_deg):
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    half = length // 2
    p1 = (int(round(c - dx * half)), int(round(c - dy * half)))
    p2 = (int(round(c + dx * half)), int(round(c + dy * half)))
    cv2.line(k, p1, p2, 1, 1)
    return k


def line_open(mask, length=15, step=15):
    out = np.zeros_like(mask)
    for ang in range(0, 180, step):
        k = line_kernel(length, ang)
        o = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        out = cv2.bitwise_or(out, o)
    return out


# ---------------- new detector: holes -> OBB ----------------
def rect_corners_f(rect):
    return cv2.boxPoints(rect)


def perimeter_support(rect, evidence, dilate=2):
    """Fraction of the rect outline covered by evidence (dilated line mask)."""
    ev = cv2.dilate(evidence, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    h, w = evidence.shape
    pts = cv2.boxPoints(rect)
    # perimeter sampling
    n = max(24, int(max(rect[1]) / 2))
    total = 0
    hit = 0
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        for t in np.linspace(0, 1, n):
            x = int(round(p1[0] + (p2[0] - p1[0]) * t))
            y = int(round(p1[1] + (p2[1] - p1[1]) * t))
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if ev[y, x]:
                    hit += 1
    return hit / total if total else 0.0


def detect_holes(cleaned, min_area=600, min_side=20, stroke=3, close_it=2,
                 max_frac=0.5, min_fill=0.65, min_support=0.65):
    H, W = cleaned.shape
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_it)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    if hierarchy is None:
        return rects
    h = hierarchy[0]
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area or area > max_frac * W * H:
            continue
        if h[i][3] == -1:      # outer contour, not a hole
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < min_side:
            continue
        # hole must be rectangular (clean interior)
        rect_area = rw * rh
        if rect_area <= 0 or area / rect_area < min_fill:
            continue
        # expand inner hole by stroke width -> drawn centerline rect
        rect2 = ((cx, cy), (rw + stroke, rh + stroke), ang)
        if perimeter_support(rect2, cleaned) < min_support:
            continue
        rects.append(rect2)
    return rects


def interior_density(rect, evidence, inset=6):
    """Fraction of line pixels strictly inside the (rotated) rect; ~0 for a real box."""
    h, w = evidence.shape
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 255)
    if inset > 0:
        m = cv2.erode(m, np.ones((2 * inset + 1, 2 * inset + 1), np.uint8))
    area = np.count_nonzero(m)
    if area == 0:
        return 0.0
    return np.count_nonzero(cv2.bitwise_and(m, evidence)) / area


def detect_outer_rings(cleaned, min_area=300, min_side=12, stroke=3, close_it=2,
                       max_frac=0.5, max_interior=0.06, min_hole_area=200):
    """Single-hole components -> rect from OUTER contour (robust to irregular holes).
    Components with != 1 hole are returned separately for Hough processing."""
    H, W = cleaned.shape
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_it)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    merged = []
    if hierarchy is None:
        return rects, merged
    h = hierarchy[0]

    # count children per outer contour
    children = {}
    for i in range(len(contours)):
        p = h[i][3]
        if p != -1:
            children.setdefault(p, []).append(i)

    for i, c in enumerate(contours):
        if h[i][3] != -1:      # skip holes here
            continue
        kids = [x for x in children.get(i, []) if cv2.contourArea(contours[x]) >= min_hole_area]
        area = cv2.contourArea(c)
        if area < min_area or area > max_frac * W * H:
            if kids:
                merged.append(i)
            continue
        if len(kids) != 1:
            merged.append(i)
            continue
        hole_area = cv2.contourArea(contours[kids[0]])
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        rw -= stroke
        rh -= stroke
        if min(rw, rh) < min_side:
            continue
        rect = ((cx, cy), (rw, rh), ang)
        if interior_density(rect, cleaned) > max_interior:
            continue
        rects.append(rect)
    return rects, merged


def dedup_contained(rects, contain_thr=0.85):

    def corners(r):
        return cv2.boxPoints(r)

    def inside_frac(inner, outer):
        ic = np.round(corners(inner)).astype(np.int32)
        oc = np.round(corners(outer)).astype(np.int32)
        x, y, w, h = cv2.boundingRect(oc)
        off = np.array([x - 1, y - 1], np.int32)
        m = np.zeros((h + 2, w + 2), np.uint8)
        cv2.fillPoly(m, [oc - off], 255)
        mi = np.zeros_like(m)
        cv2.fillPoly(mi, [ic - off], 255)
        inter = np.count_nonzero(cv2.bitwise_and(m, mi))
        ia = np.count_nonzero(mi)
        return inter / ia if ia else 0.0

    rects = sorted(rects, key=lambda r: r[1][0] * r[1][1], reverse=True)
    keep = []
    for r in rects:
        if any(inside_frac(r, k) >= contain_thr for k in keep):
            continue
        keep.append(r)
    return keep


# ---------------- Canny-based text filter ----------------
def canny_straight_mask(mask, canny_low=50, canny_high=150,
                        hough_thr=12, min_len=25, max_gap=6,
                        stroke_width=5):
    """Keep only mask pixels on straight line segments (box edges).

    Canny edges -> HoughLinesP -> dilate detected lines -> intersect with mask.
    Text glyphs produce short/curved Canny edges that don't form long straight
    lines, so they are excluded.
    """
    edges = cv2.Canny(mask, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thr,
                            minLineLength=min_len, maxLineGap=max_gap)
    straight = np.zeros_like(mask)
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            cv2.line(straight, (x1, y1), (x2, y2), 255, 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (stroke_width, stroke_width))
    straight = cv2.dilate(straight, k)
    return cv2.bitwise_and(mask, straight)


def hough_straight_mask(mask, hough_thr=15, min_len=25, max_gap=6,
                        stroke_width=5):
    """Variant: HoughLinesP directly on the binary mask (skip Canny).

    The mask is already binary; HoughLinesP accumulates votes in rho-theta
    space.  Short text strokes rarely reach the threshold.
    """
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=hough_thr,
                            minLineLength=min_len, maxLineGap=max_gap)
    straight = np.zeros_like(mask)
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            cv2.line(straight, (x1, y1), (x2, y2), 255, 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (stroke_width, stroke_width))
    straight = cv2.dilate(straight, k)
    return cv2.bitwise_and(mask, straight)


def component_straightness_filter(mask, min_len=25, max_residual=2.0):
    """Per-component straightness test using Canny + line fit.

    For each connected component:
      1. Extract Canny edges of the component.
      2. Fit a line (PCA / least-squares) to the edge points.
      3. Compute mean perpendicular residual.
      4. Keep only if the component is long enough AND fits a straight line.
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        length = np.hypot(w, h)
        if length < min_len:
            continue
        comp = (lab == i).astype(np.uint8) * 255
        edges = cv2.Canny(comp, 50, 150)
        ys, xs = np.nonzero(edges)
        if len(xs) < 10:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        cx, cy = pts.mean(axis=0)
        centered = pts - [cx, cy]
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        residuals = np.abs(centered @ normal)
        if residuals.mean() > max_residual:
            continue
        out[lab == i] = 255
    return out


# ---------------- combined detector ----------------
def canny_validate(mask, canny_low=50, canny_high=150,
                   hough_thr=12, min_len=25, max_gap=6,
                   min_overlap=0.08):
    """Gentle Canny validation: keep components with straight-line support.

    Unlike canny_straight_mask which intersects pixel-by-pixel, this keeps
    entire connected components that have >= min_overlap fraction of their
    pixels near a detected straight line segment.
    """
    edges = cv2.Canny(mask, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thr,
                            minLineLength=min_len, maxLineGap=max_gap)
    line_mask = np.zeros_like(mask)
    if lines is not None:
        for l in lines.reshape(-1, 4):
            cv2.line(line_mask, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 255, 5)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = lab == i
        overlap = np.count_nonzero(comp & (line_mask > 0))
        if overlap / stats[i][4] >= min_overlap:
            out[comp] = 255
    return out


def detect_combined(mask, use_text_filter=True, use_canny=True,
                    min_area=300, min_side=12, min_fill=0.65,
                    min_support=0.65, stroke=3, close_it=2,
                    nms_thr=0.4):
    """Full pipeline: text filter -> Canny straight filter -> holes + Hough -> NMS -> dedup."""
    m = mask
    if use_text_filter:
        m = remove_text(m)
    if use_canny:
        m = canny_straight_mask(m)
    hole_rects = detect_holes(m, min_area=min_area, min_side=min_side,
                              stroke=stroke, close_it=close_it,
                              min_fill=min_fill, min_support=min_support)
    hough_rects = detect_hough_rects(m, min_side=min_side)
    all_rects = hole_rects + hough_rects
    all_rects = nms(all_rects, thr=nms_thr)
    all_rects = dedup_contained(all_rects)
    return all_rects


# ---------------- Hough-based OBB assembly (Option A) ----------------
def _seg_angle(s):
    return np.arctan2(s[3] - s[1], s[2] - s[0])


def _angdiff(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def _perp_dist(seg, pt):
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L == 0:
        return np.hypot(pt[0] - x1, pt[1] - y1)
    return abs((pt[0] - x1) * dy - (pt[1] - y1) * dx) / L


def _merge_collinear(segs, ang_tol=4.0, dist_tol=4.0):
    order = sorted(range(len(segs)), key=lambda i: -np.hypot(segs[i][2] - segs[i][0], segs[i][3] - segs[i][1]))
    used = [False] * len(segs)
    merged = []
    for oi in order:
        if used[oi]:
            continue
        s = segs[oi]
        used[oi] = True
        ang = _seg_angle(s)
        group = [s]
        for j in range(len(segs)):
            if used[j]:
                continue
            t = segs[j]
            if _angdiff(ang, _seg_angle(t)) > np.deg2rad(ang_tol):
                continue
            mid = ((t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0)
            if _perp_dist(s, mid) > dist_tol:
                continue
            group.append(t)
            used[j] = True
        u = (np.cos(ang), np.sin(ang))
        origin = (s[0], s[1])
        ts = []
        for g in group:
            for p in [(g[0], g[1]), (g[2], g[3])]:
                ts.append((p[0] - origin[0]) * u[0] + (p[1] - origin[1]) * u[1])
        lo, hi = min(ts), max(ts)
        merged.append(((origin[0] + lo * u[0], origin[1] + lo * u[1]),
                       (origin[0] + hi * u[0], origin[1] + hi * u[1]), ang))
    return merged


def detect_hough_rects(cleaned, min_side=15, max_side=600, angle_tol_deg=8.0, min_overlap=15,
                       min_support=0.7, hough_min_len=18, hough_gap=8, hough_thr=18):
    lines = cv2.HoughLinesP(cleaned, 1, np.pi / 360, threshold=hough_thr,
                            minLineLength=hough_min_len, maxLineGap=hough_gap)
    if lines is None:
        return []
    segs = [tuple(int(v) for v in l) for l in lines.reshape(-1, 4)]
    merged = _merge_collinear(segs)
    rects = []
    n = len(merged)
    for i in range(n):
        p1, p2, a1 = merged[i]
        for j in range(i + 1, n):
            q1, q2, a2 = merged[j]
            if _angdiff(a1, a2) > np.deg2rad(angle_tol_deg):
                continue
            d = _perp_dist((p1[0], p1[1], p2[0], p2[1]), q1)
            if d < min_side or d > max_side:
                continue
            ang = (a1 + a2) / 2.0
            u = (np.cos(ang), np.sin(ang))
            nvec = (-u[1], u[0])
            origin = p1

            def proj(p):
                return (p[0] - origin[0]) * u[0] + (p[1] - origin[1]) * u[1]

            lo = max(min(proj(p1), proj(p2)), min(proj(q1), proj(q2)))
            hi = min(max(proj(p1), proj(p2)), max(proj(q1), proj(q2)))
            if hi - lo < min_overlap:
                continue
            off2 = (q1[0] - origin[0]) * nvec[0] + (q1[1] - origin[1]) * nvec[1]
            corners = []
            for t, off in [(lo, 0.0), (hi, 0.0), (hi, off2), (lo, off2)]:
                corners.append([origin[0] + t * u[0] + off * nvec[0],
                                origin[1] + t * u[1] + off * nvec[1]])
            rect = cv2.minAreaRect(np.array(corners, np.float32))
            if min(rect[1]) < min_side:
                continue
            if perimeter_support(rect, cleaned, dilate=1) < min_support:
                continue
            rects.append(rect)
    return rects

# ---------------- OBB-guided text removal (two-pass) ----------------
def obb_remove_text(mask, rects, edge_band=5, shrink=2):
    """Remove text inside detected OBBs by keeping only the edge band.

    For each OBB:
      - Fill the OBB polygon (interior).
      - Draw the OBB outline with thickness 2*edge_band+1 (edge band).
      - Erode the interior by `shrink` to avoid touching the stroke.
      - Remove mask pixels that are in the eroded interior but NOT in the edge band.
    """
    h, w = mask.shape
    remove = np.zeros_like(mask)
    for rect in rects:
        pts = cv2.boxPoints(rect).astype(np.int32)
        interior = np.zeros((h, w), np.uint8)
        cv2.fillPoly(interior, [pts], 255)
        if shrink > 0:
            interior = cv2.erode(interior, np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8))
        edge = np.zeros((h, w), np.uint8)
        cv2.polylines(edge, [pts], True, 255, 2 * edge_band + 1)
        text_region = cv2.bitwise_and(interior, cv2.bitwise_not(edge))
        remove = cv2.bitwise_or(remove, text_region)
    return cv2.bitwise_and(mask, cv2.bitwise_not(remove))


def detect_twopass(mask, min_area=300, min_side=12, min_fill=0.50, min_support=0.50,
                   stroke=3, close_it=2, nms_thr=0.4, line_len=15, line_step=15,
                   edge_band=5, shrink=2):
    """Two-pass detector: initial detection -> OBB-guided text removal -> re-detect."""
    m = remove_text(mask)
    m = line_open(m, length=line_len, step=line_step)
    rects = detect_holes(m, min_area=min_area, min_side=min_side, stroke=stroke,
                         close_it=close_it, min_fill=min_fill, min_support=min_support)
    rects = nms(rects, thr=nms_thr)
    rects = dedup_contained(rects)
    if not rects:
        return rects

    cleaned = obb_remove_text(mask, rects, edge_band=edge_band, shrink=shrink)
    m2 = remove_text(cleaned)
    m2 = line_open(m2, length=line_len, step=line_step)
    rects2 = detect_holes(m2, min_area=min_area, min_side=min_side, stroke=stroke,
                          close_it=close_it, min_fill=min_fill, min_support=min_support)
    rects2 = nms(rects2, thr=nms_thr)
    rects2 = dedup_contained(rects2)
    return rects2

def obb_remove_text_components(mask, rects, max_comp_size=30, max_comp_area=600,
                               edge_band=5, shrink=2):
    """Gentler OBB text removal: only remove small connected components inside OBBs.

    Instead of clearing all interior pixels, find connected components in the
    mask that are inside an OBB interior and small enough to be text glyphs.
    """
    h, w = mask.shape
    interior_mask = np.zeros((h, w), np.uint8)
    for rect in rects:
        pts = cv2.boxPoints(rect).astype(np.int32)
        fill = np.zeros((h, w), np.uint8)
        cv2.fillPoly(fill, [pts], 255)
        if shrink > 0:
            fill = cv2.erode(fill, np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8))
        edge = np.zeros((h, w), np.uint8)
        cv2.polylines(edge, [pts], True, 255, 2 * edge_band + 1)
        fill = cv2.bitwise_and(fill, cv2.bitwise_not(edge))
        interior_mask = cv2.bitwise_or(interior_mask, fill)

    inside = cv2.bitwise_and(mask, interior_mask)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(inside, 8)
    remove_ids = [i for i in range(1, n)
                  if max(stats[i][2], stats[i][3]) <= max_comp_size
                  and stats[i][4] <= max_comp_area]
    if not remove_ids:
        return mask.copy()
    out = mask.copy()
    out[np.isin(lab, remove_ids)] = 0
    return out


def detect_twopass_components(mask, min_area=300, min_side=12, min_fill=0.50, min_support=0.50,
                              stroke=3, close_it=2, nms_thr=0.4, line_len=15, line_step=15,
                              edge_band=5, shrink=2, max_comp_size=30, max_comp_area=600):
    """Two-pass with gentler component-level text removal inside OBBs."""
    m = remove_text(mask)
    m = line_open(m, length=line_len, step=line_step)
    rects = detect_holes(m, min_area=min_area, min_side=min_side, stroke=stroke,
                         close_it=close_it, min_fill=min_fill, min_support=min_support)
    rects = nms(rects, thr=nms_thr)
    rects = dedup_contained(rects)
    if not rects:
        return rects

    cleaned = obb_remove_text_components(mask, rects, max_comp_size=max_comp_size,
                                         max_comp_area=max_comp_area,
                                         edge_band=edge_band, shrink=shrink)
    m2 = remove_text(cleaned)
    m2 = line_open(m2, length=line_len, step=line_step)
    rects2 = detect_holes(m2, min_area=min_area, min_side=min_side, stroke=stroke,
                          close_it=close_it, min_fill=min_fill, min_support=min_support)
    rects2 = nms(rects2, thr=nms_thr)
    rects2 = dedup_contained(rects2)
    return rects2
