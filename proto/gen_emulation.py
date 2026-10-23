"""Generate the emulation dataset: synthetic OBB annotations on real X-ray backgrounds.

Source backgrounds : E:\\Xray\\CARailwaySciences\\Image@Xray.CARS.202505  (2564 jpgs)
Output images      : data/Image@emulation/emul_XXXXXX.jpg   (JPEG)
Output labels      : data/Label@emulation/emul_XXXXXX.json  (BRobbDetector label format)
Manifest           : benchmark/emulation_manifest.json

Every output image is a source X-ray frame with 1-4 random OBB outlines drawn on
top of the *content* (non-background) region:

  * side lengths   : uniform in [100, 400] px
  * angle          : uniform in [0, 90) deg
  * stroke colour  : one of the three additive primaries - red / green / blue
  * stroke width   : 2-4 px, antialiased

"Non-background" is resolved per pixel: the background of these frames is the
near-white paper, the content is the scanned cargo.  A foreground mask is built
with Otsu on the gray channel, cleaned morphologically, and a box is only kept
when its centre sits on the eroded content core and its interior is covered by
content for at least `--min-coverage` of its area.

The label records the drawn corners as "x,y;x,y;x,y;x,y" in the same corner
order as the hand-written GT (TL, TR, BR, BL) and the stroke colour in "type".
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DEFAULT = r"E:\Xray\CARailwaySciences\Image@Xray.CARS.202505"
OUT_IMG = os.path.join(ROOT, "data", "Image@emulation")
OUT_LAB = os.path.join(ROOT, "data", "Label@emulation")
MANIFEST = os.path.join(ROOT, "benchmark", "emulation_manifest.json")

# (type name, BGR) - the three additive primaries.
#
# The tones are the ones the REAL annotation uses: the hand drawn line is an
# opaque, desaturated red (measured on Image@CARS_202609_annotated: the core
# of the band is constantly BGR (59,60,235) whatever the background is under
# it, i.e. RGB (235,60,59) - a "salmon" pen, not a pure 255,0,0).
#
# The tone matters: with a pure red the anti aliased edge of the stroke still
# passes the production mask (hue<=4, sat>=100, val>=140), with the salmon
# tone it does not, so the band the detector sees is narrower - exactly as in
# the real data (measured band width p50 4.0 px).
COLORS = [
    ("红", (59, 60, 235)),
    ("绿", (60, 235, 60)),
    ("蓝", (235, 60, 60)),
]

# boxes per image: n -> weight
BOX_COUNT_DIST = {1: 0.35, 2: 0.35, 3: 0.20, 4: 0.10}


def obb_corners(cx, cy, w, h, ang_deg):
    """4 corners of a rotated rectangle, ordered TL, TR, BR, BL (clockwise)."""
    a = np.radians(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    ax, ay = (w / 2.0) * ca, (w / 2.0) * sa
    bx, by = -(h / 2.0) * sa, (h / 2.0) * ca
    return np.array([
        [cx - ax - bx, cy - ay - by],   # TL
        [cx + ax - bx, cy + ay - by],   # TR
        [cx + ax + bx, cy + ay + by],   # BR
        [cx - ax + bx, cy - ay + by],   # BL
    ], dtype=np.float64)


def foreground_mask(img):
    """Binary mask (255 = content) of the scanned cargo."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape
    min_area = max(300, int(0.0015 * H * W))
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def rect_coverage(fg, corners):
    """Fraction of the box interior that falls on foreground content."""
    H, W = fg.shape
    x0 = max(0, int(np.floor(corners[:, 0].min())) - 1)
    x1 = min(W, int(np.ceil(corners[:, 0].max())) + 2)
    y0 = max(0, int(np.floor(corners[:, 1].min())) - 1)
    y1 = min(H, int(np.ceil(corners[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = fg[y0:y1, x0:x1]
    m = np.zeros(roi.shape, np.uint8)
    cv2.fillPoly(m, [(corners - [x0, y0]).astype(np.int32)], 255)
    tot = int(np.count_nonzero(m))
    if tot == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(roi, m))) / tot


def poly_iou(a, b):
    """IoU of two convex 4-point polygons."""
    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    inter = abs(float(inter))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def place_boxes(rng, fg, core_xy, n_want, size_min, size_max, margin=3,
                min_coverage=0.5, max_iou=0.05, tries_per_box=400, pool=None,
                stroke_min=2, stroke_max=5, min_gap=8.0):
    """Place up to n_want OBBs on the content region. Returns list of dicts.

    `pool` (optional) is a list of (w, h) pairs; when given, the side lengths
    are drawn from it instead of a uniform range, so that the emulation can
    reproduce the size / aspect distribution of the real annotations.

    `min_gap` keeps the boxes apart by that many pixels: a stroke drawn later
    must never paint over the side of an earlier box, otherwise the label of
    the covered box no longer matches the image.
    """
    H, W = fg.shape
    placed = []
    # dilated foreground: translucent content edges still count as content
    fg_dil = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    if len(core_xy) == 0:
        return placed

    for _ in range(n_want):
        got = False
        if pool:
            w, h = pool[rng.randint(len(pool))]
            w = float(w)
            h = float(h)
        else:
            w = rng.uniform(size_min, size_max)
            h = rng.uniform(size_min, size_max)
        fit_fail = 0
        for _ in range(tries_per_box):
            a, b = w / 2.0, h / 2.0
            ang = rng.uniform(0.0, 90.0)
            rad = np.radians(ang)
            ex = a * abs(np.cos(rad)) + b * abs(np.sin(rad))
            ey = a * abs(np.sin(rad)) + b * abs(np.cos(rad))
            if 2 * ex > W - 1 - 2 * margin or 2 * ey > H - 1 - 2 * margin:
                # this angle does not fit -> retry another angle, shrink only
                # after several consecutive failures
                fit_fail += 1
                if fit_fail % 8 == 0:
                    w = max(size_min, w * 0.9)
                    h = max(size_min, h * 0.9)
                continue
            fit_fail = 0
            lo_x, hi_x = ex + margin, W - 1 - ex - margin
            lo_y, hi_y = ey + margin, H - 1 - ey - margin
            if lo_x > hi_x or lo_y > hi_y:
                continue
            idx = rng.randint(0, len(core_xy))
            cx, cy = float(core_xy[idx, 0]), float(core_xy[idx, 1])
            if not (lo_x <= cx <= hi_x and lo_y <= cy <= hi_y):
                continue
            corners = obb_corners(cx, cy, w, h, ang)
            if rect_coverage(fg_dil, corners) < min_coverage:
                continue
            c_int = np.rint(corners).astype(np.int32)
            if any(poly_iou(c_int, p["corners"]) > max_iou for p in placed):
                continue
            # keep a gap so that no stroke can paint over another box
            r = 0.5 * math.hypot(w, h)
            too_close = False
            for p in placed:
                rp = 0.5 * math.hypot(p["w"], p["h"])
                if math.hypot(cx - p["cx"], cy - p["cy"]) < r + rp + min_gap:
                    too_close = True
                    break
            if too_close:
                continue
            placed.append({
                "corners": c_int,
                "w": w, "h": h, "angle": ang,
                "cx": cx, "cy": cy,
            })
            got = True
            break
        if not got:
            break

    # assign colours + stroke widths
    for p in placed:
        name, bgr = COLORS[rng.randint(len(COLORS))]
        p["type"] = name
        p["color"] = bgr
        p["stroke_w"] = int(rng.randint(stroke_min, stroke_max))
    return placed


def draw_box(img, corners, color, stroke_w, support=None, spans=None,
             overshoot=0.0):
    """Draw the rectangle outline.

    `support` (4 values in [0, 1]) gives the fraction of each side that is
    actually painted - the real hand-drawn annotation is often only partially
    rendered in the image, and that is what limits the detector.  The label
    keeps the ideal corners; the stroke covers a contiguous part of each side
    (`spans` gives its position as a fraction of the side) plus `overshoot` px
    past both ends, like a real pen stroke.
    """
    c = np.asarray(corners, float)
    for i in range(4):
        a = c[i]
        b = c[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        s = 1.0 if support is None else float(np.clip(support[i], 0.0, 1.0))
        if s <= 0.0:
            continue
        span = s * L
        if spans is None:
            t0 = 0.0
        else:
            t0 = spans[i] * (L - span)
        p0 = a + (b - a) * (t0 / L)
        p1 = a + (b - a) * ((t0 + span) / L)
        if overshoot > 0:
            u = (b - a) / L
            p0 = p0 - u * overshoot
            p1 = p1 + u * overshoot
        cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), color, stroke_w,
                 cv2.LINE_AA)


# --------------------------------------------------------- annotation text
#
# The real annotation is not only the rectangle: every object also carries a
# red class label ("液体 0.95") and a leader arrow that points from the label
# to the object.  Both are red, both are connected with the box, and the
# arrowhead that lands on a box corner turns the hole of the red mask into a
# spiky polygon - the single largest source of false positives on the real
# data (proto/why_fp.py, proto/viz_real_fp.py).
#
# Measured on the 500 real images (proto/measure_annotation_furniture.py):
#   * label glyphs are ~11 x 14 px, area p50 87
#   * ~10 red components per image
#   * 26% of the holes of the red mask are not rectangular
#
# The glyphs are drawn as random strokes inside a 11 x 14 cell: the detector
# only ever sees the red mask, so the shape statistics are what matters.


def draw_glyph_run(img, origin, n_chars, color, rng, cw=11, ch=14, gap=2):
    """Draw a run of `n_chars` label glyphs; returns the run's bbox."""
    x0, y0 = int(origin[0]), int(origin[1])
    for k in range(n_chars):
        gx = x0 + k * (cw + gap)
        n_stroke = rng.randint(2, 5)
        for _ in range(n_stroke):
            ax = gx + rng.uniform(1.0, cw - 1.0)
            ay = y0 + rng.uniform(1.0, ch - 1.0)
            bx = gx + rng.uniform(1.0, cw - 1.0)
            by = y0 + rng.uniform(1.0, ch - 1.0)
            cv2.line(img, (int(round(ax)), int(round(ay))),
                     (int(round(bx)), int(round(by))), color, 2, cv2.LINE_AA)
    return (x0, y0, x0 + n_chars * (cw + gap) - gap, y0 + ch)


def draw_leader_arrow(img, p_from, p_to, color, rng, width=2, head=10.0,
                      spread=26.0):
    """A leader arrow: shaft + arrow head, as the annotation tool draws it."""
    a = np.asarray(p_from, float)
    b = np.asarray(p_to, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return
    u = d / L
    cv2.line(img, (int(round(a[0])), int(round(a[1]))),
             (int(round(b[0]) - u[0] * head * 0.4),
              int(round(b[1]) - u[1] * head * 0.4)), color, width, cv2.LINE_AA)
    base = b - u * head
    for s in (1.0, -1.0):
        ang = np.radians(90.0 + s * spread)
        v = np.array([np.cos(ang), np.sin(ang)])
        p = base + v * head
        cv2.line(img, (int(round(base[0])), int(round(base[1]))),
                 (int(round(p[0])), int(round(p[1]))), color, width, cv2.LINE_AA)


def draw_annotation_furniture(img, corners, color, rng, label_prob=0.7,
                               arrow_prob=0.6, cw=11, ch=14, gap=2,
                               n_chars=(3, 8), corner_prob=0.5):
    """Draw the class label (and optionally the leader arrow) of one box."""
    H, W = img.shape[:2]
    if rng.rand() > label_prob:
        return
    c = np.asarray(corners, float)
    # the label sits outside the box, next to one of its corners
    ci = int(rng.randint(4))
    corner = c[ci]
    centre = c.mean(axis=0)
    out = corner - centre
    n = float(np.linalg.norm(out))
    if n < 1e-6:
        return
    out = out / n
    nc = int(rng.randint(n_chars[0], n_chars[1] + 1))
    w_run = nc * (cw + gap) - gap
    dist = rng.uniform(26.0, 46.0)
    base = corner + out * dist
    # keep the run inside the frame
    x0 = int(round(base[0] - w_run / 2.0))
    y0 = int(round(base[1] - ch / 2.0))
    x0 = int(np.clip(x0, 1, max(1, W - w_run - 2)))
    y0 = int(np.clip(y0, 1, max(1, H - ch - 2)))
    bb = draw_glyph_run(img, (x0, y0), nc, color, rng, cw, ch, gap)
    if rng.rand() > arrow_prob:
        return
    # the arrow points from the label to the box: mostly at a corner, but
    # sometimes at a random point of a side
    if rng.rand() < corner_prob:
        target = corner
    else:
        si = int(rng.randint(4))
        t = rng.uniform(0.25, 0.75)
        target = c[si] + (c[(si + 1) % 4] - c[si]) * t
    frm = np.array([bb[0] + w_run * 0.5, bb[1] + ch * 0.5], float)
    draw_leader_arrow(img, frm, target, color, rng)


# --------------------------------------------------------------- wobbly pen
#
# The real annotation is a hand drawn line, and the GT rectangle is its
# idealisation.  proto/measure_real_line.py measured, on the production red
# mask of the real data:
#
#   * the red band is ~4 px wide (p10 3.0, p50 4.0, p90 7.5)
#   * on 83% of the sides the band sits ON the ideal line (coverage 0.97)
#   * on the remaining sides the band is DISPLACED by a few px, so the ideal
#     line is only covered on part of the side - but the band is continuous,
#     the shape still closes and the detector still finds the box.
#
# The missing part of a side is therefore NOT a gap in the outline (that would
# open the shape and the box would be lost) but a *wobbling* line.  The
# emulation reproduces that: every side is drawn as a smooth, zero-mean
# wandering polyline whose amplitude is solved so that the coverage of the
# ideal side line matches the requested target.

WOBBLE_NHARM = 3


def wobble_profile(rng, nharm=1, fmin=0.3, fmax=1.2):
    """The wander of the pen around the ideal edge, as a function of the
    position along the side.

    Returns (t, g): t is the position along the side (0..1), g the offset in
    units of the amplitude.  g is normalised so that max|g| = 1, i.e. `amp`
    is the largest distance the pen strays from the ideal edge.

    The profile is a slow *bow* that starts and ends on the ideal edge (the
    corners of the hand drawn rectangle are where the pen started and
    stopped, they are correct) and bulges away in between.  That is what the
    real data shows (proto/measure_real_wobble.py, 12 bins per side):
    e.g. "+3.5 +3.0 +2.0 +2.0 +1.0 +0.5 +0.0 -3.0 -4.0 -2.0 +0.5 +3.0" -
    one smooth excursion, not a zigzag.
    """
    t = np.linspace(0.0, 1.0, 96)
    env = np.sin(np.pi * t)          # 0 at both ends: the corners stay closed
    g = np.zeros_like(t)
    for j in range(nharm):
        f = rng.uniform(fmin, fmax)
        ph = rng.uniform(0.0, 2.0 * math.pi)
        g += (1.0 / (j + 1.0)) * np.sin(2.0 * math.pi * f * t + ph)
    g *= env
    mx = float(np.max(np.abs(g)))
    if mx > 1e-9:
        g /= mx
    return t, g


def _side_frame(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    return a, u, n, L


# HSV windows of the three additive primaries (the same windows the
# benchmark uses, proto/eval_emulation.py::color_mask).
COLOR_HUE = {
    "红": (0, 4),
    "绿": (60, 6),
    "蓝": (120, 6),
}
MASK_SAT_MIN = 100
MASK_VAL_MIN = 140


def primary_mask(img, color):
    """Mask of the pixels of the given primary colour (production windows)."""
    centre, w = COLOR_HUE[color]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = MASK_SAT_MIN, MASK_VAL_MIN
    if color == "红":
        m1 = cv2.inRange(hsv, (0, s, v), (w, 255, 255))
        m2 = cv2.inRange(hsv, (170, s, v), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    lo = max(0, centre - w)
    hi = min(179, centre + w)
    return cv2.inRange(hsv, (lo, s, v), (hi, 255, 255))


def _side_points(a, u, n, L, profile, amp, shift, t0, t1, drift=0.0):
    ts = np.linspace(t0, t1, max(8, int(L * (t1 - t0) / 2.0) + 1))
    off = np.interp(ts, profile[0], profile[1]) * amp + shift
    if drift:
        # the pen does not start and stop on the ideal corners: the line
        # drifts from one end of the side to the other, so the drawn
        # rectangle is a trapezoid and the corners are off by `drift`.
        off = off + drift * (2.0 * ts - 1.0)
    return a[None, :] + ts[:, None] * (u * L)[None, :] + off[:, None] * n[None, :]


def render_coverage(img, a, b, profile, amp, shift, stroke_w, color,
                    color_name, t0=0.0, t1=1.0, pad=8, drift=0.0):
    """Coverage of the IDEAL side line when the band is rendered on the real
    background of `img`.

    The background matters: on the textured cargo the anti aliased edges of
    the stroke blend with the content and leave the pure colour core only, so
    the effective width of the band is much smaller than the drawn width.  The
    coverage is therefore measured on the image itself, not modelled.

    `shift` moves the whole side parallel to the ideal line (the pen did not
    follow the ideal edge), `amp` is the wobble around that position and
    `drift` tilts the side (the pen did not start and stop on the corners).
    """
    fr = _side_frame(a, b)
    if fr is None:
        return 0.0
    a, u, n, L = fr
    H, W = img.shape[:2]
    x0 = max(0, int(min(a[0], b[0]) - pad))
    x1 = min(W, int(max(a[0], b[0]) + pad + 1))
    y0 = max(0, int(min(a[1], b[1]) - pad))
    y1 = min(H, int(max(a[1], b[1]) + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = img[y0:y1, x0:x1].copy()
    pts = _side_points(a, u, n, L, profile, amp, shift, t0, t1,
                       drift=drift) - np.array([x0, y0])
    cv2.polylines(roi, [np.round(pts).astype(np.int32)], False, color, stroke_w,
                  cv2.LINE_AA)
    m = primary_mask(roi, color_name)
    ns = max(int(L), 2)
    ts = np.linspace(0.0, 1.0, ns)
    spts = a[None, :] + ts[:, None] * (u * L)[None, :]
    q = np.round(spts - np.array([x0, y0])).astype(int)
    q[:, 0] = np.clip(q[:, 0], 0, roi.shape[1] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, roi.shape[0] - 1)
    hit = m[q[:, 1], q[:, 0]] > 0
    hit &= (ts >= t0) & (ts <= t1)
    return float(hit.mean())


def solve_amp_render(img, a, b, profile, target, stroke_w, color, color_name,
                     t0=0.0, t1=1.0, amp_max=8.0, iters=14, shift=0.0, drift=0.0):
    """Wobble amplitude that makes the rendered coverage equal `target`.

    The real pen does not draw a line *beside* the ideal edge - it draws a
    line that weaves around it.  proto/measure_real_wobble.py measured, per
    side, the offset of the red band in 12 bins along the side:

        coverage [0.9,1.0]  mean offset ~0.0 px
        coverage [0.3,0.5]  mean offset ~0.4 px   peak |offset| 1.4 px
        coverage [0.1,0.3]  mean offset ~0.4 px   peak |offset| 2.5 px
        coverage [0.0,0.1]  mean offset ~0.0 px   peak |offset| 5.7 px

    i.e. the line stays centred on the ideal edge and only *excursions* away
    from it destroy the coverage.  The amplitude is therefore the free
    parameter and the shift is 0 (a shift would translate the whole side and
    move the corners of the box - the real data shows a corner error of only
    1-2 px, see benchmark/real_degrade_full.json).
    """
    if target >= 0.999:
        return 0.0
    lo, hi = 0.0, amp_max
    if render_coverage(img, a, b, profile, hi, shift, stroke_w, color, color_name,
                       t0, t1, drift=drift) > target:
        return hi                        # cannot reach the target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if render_coverage(img, a, b, profile, mid, shift, stroke_w, color,
                           color_name, t0, t1, drift=drift) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def draw_wobbly_box(img, corners, color, stroke_w, recipe, rng, color_name=None,
                     overshoot=3.0, wobble_min=0.8, wobble_max=2.5,
                     shift_max=6.0, amp_max=8.0, shift_bias=0.8, shift_min=2.0,
                     drift_max=0.0, broken_amp_max=24.0, nharm=1):
    """Draw a hand-drawn looking rectangle.

    `recipe` - one entry per side: (coverage, missing, band_cover).
      coverage   - the target coverage of the ideal side line
      missing    - 1 when the red is absent over a stretch of the side
      band_cover - the fraction of the side that carries red (missing sides)

    Two mechanisms, both measured on the real data:

      * the side is complete - the pen weaves around the ideal edge; the
        wobble amplitude is solved on the real background so that the
        rendered coverage of the ideal line hits the target;
      * the side is broken - the pen did not follow the ideal edge: the line
        DRIFTS from one end of the side to the other (a trapezoid, the
        corners are off by `drift`) and weaves around it.  The wobble
        amplitude is solved so that the rendered coverage of the ideal line
        hits the target.

    Why the drift matters: on the real data a box without a broken side is
    found in 96.9% of the cases, a box with one broken side only in 42.9%
    (benchmark/real_degrade_full.json).  The missed ones are not "off by a
    bit" - the nearest detection of a missed box is ~36 px away, i.e. the
    pipeline locks onto the *drawn* (distorted) rectangle instead of the
    idealised GT one.  So the drawn line has to leave the ideal edge far
    enough to move the corners out of the D=5 gate.
    """
    c = np.asarray(corners, float)
    if color_name is None:
        color_name = {v: k for k, v in COLORS}.get(tuple(color), "红")
    for i in range(4):
        a = c[i]
        b = c[(i + 1) % 4]
        fr = _side_frame(a, b)
        if fr is None:
            continue
        a, u, n, L = fr
        cov, missing, band = recipe[i]
        target = float(np.clip(cov, 0.0, 1.0))
        profile = wobble_profile(rng, nharm=nharm)
        if missing > 0.5:
            # the line leaves the ideal edge: it drifts from one corner to
            # the other, so the drawn side is tilted and the corners of the
            # drawn rectangle are `drift` px away from the ideal ones
            shift = 0.0
            drift = rng.uniform(0.0, max(drift_max, 0.0))
            if rng.rand() < 0.5:
                drift = -drift
            amp = solve_amp_render(img, a, b, profile, target, stroke_w, color,
                                   color_name, 0.0, 1.0,
                                   amp_max=broken_amp_max, drift=drift)
        else:
            # the pen follows the edge with a small bias (the real data: a
            # mean offset of 0.4-1.5 px) and bows away from it in the middle
            shift = rng.uniform(-shift_bias, shift_bias)
            drift = 0.0
            amp = solve_amp_render(img, a, b, profile, target, stroke_w, color,
                                   color_name, 0.0, 1.0, amp_max=amp_max)
        pts = _side_points(a, u, n, L, profile, amp, shift, 0.0, 1.0,
                           drift=drift)
        if overshoot > 0:
            pts[0] -= u * overshoot
            pts[-1] += u * overshoot
        ipts = np.round(pts).astype(np.int32)
        cv2.polylines(img, [ipts], False, color, stroke_w, cv2.LINE_AA)



def pts_str(corners):
    return ";".join("%d,%d" % (int(p[0]), int(p[1])) for p in corners)


def source_core_size(path):
    """Size (px) of the eroded content core of a source frame (0 = unusable)."""
    img = cv2.imread(path)
    if img is None:
        return 0
    fg = foreground_mask(img)
    core = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return int(np.count_nonzero(core))


def usable_sources(files, min_core=2000, workers=10, cache=None):
    """Filter out frames that are (almost) empty - they cannot host a box."""
    sizes = {}
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            sizes = {k: int(v) for k, v in json.load(f).items()}
    todo = [f for f in files if os.path.basename(f) not in sizes]
    if todo:
        with Pool(workers) as pool:
            vals = pool.map(source_core_size, todo, chunksize=8)
        for p, n in zip(todo, vals):
            sizes[os.path.basename(p)] = int(n)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(sizes, f, ensure_ascii=False)
    return [f for f in files if sizes.get(os.path.basename(f), 0) >= min_core]


def process(task):
    """Worker: build one emulation image + label. Returns a stats dict."""
    (idx, src_path, out_name, seed, n_want, size_min, size_max,
     min_coverage, jpeg_quality, out_img, out_lab, pool, stroke_min,
     stroke_max, min_gap, support_pool, overshoot, stroke_model,
     recipe_pool, draw_params, furniture) = task
    rng = np.random.RandomState(seed)
    img = cv2.imread(src_path)
    if img is None:
        return {"ok": False, "idx": idx}
    H, W = img.shape[:2]
    fg = foreground_mask(img)
    core = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    ys, xs = np.nonzero(core)
    core_xy = np.stack([xs, ys], axis=1).astype(np.int32) if len(xs) else np.zeros((0, 2), np.int32)

    boxes = place_boxes(rng, fg, core_xy, n_want, size_min, size_max,
                        min_coverage=min_coverage, pool=pool,
                        stroke_min=stroke_min, stroke_max=stroke_max,
                        min_gap=min_gap)
    for b in boxes:
        sup = None
        spans = None
        if support_pool:
            sup = support_pool[rng.randint(len(support_pool))]
            spans = [rng.uniform(0.0, 1.0) for _ in range(4)]
        if stroke_model == "wobbly":
            # the pen wanders around the ideal edge; the recipe holds, per
            # side, the target coverage of the ideal line and whether the red
            # is missing over a stretch of the side
            recipe = (recipe_pool[rng.randint(len(recipe_pool))]
                      if recipe_pool else [[1.0, 0.0, 1.0]] * 4)
            draw_wobbly_box(img, b["corners"], b["color"], b["stroke_w"],
                            recipe, rng, color_name=b["type"],
                            overshoot=overshoot, **draw_params)
        else:
            draw_box(img, b["corners"], b["color"], b["stroke_w"], sup, spans,
                     overshoot)
        if furniture:
            draw_annotation_furniture(img, b["corners"], b["color"], rng,
                                      label_prob=furniture["label_prob"],
                                      arrow_prob=furniture["arrow_prob"],
                                      corner_prob=furniture["corner_prob"])

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)
    img_path = os.path.join(out_img, out_name)
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    label = {
        "img_name": out_name,
        "infos": [{"points": pts_str(b["corners"]), "type": b["type"]} for b in boxes],
    }
    with open(os.path.join(out_lab, os.path.splitext(out_name)[0] + ".json"),
              "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False)

    return {
        "ok": True, "idx": idx, "src": os.path.basename(src_path),
        "out": out_name, "n_boxes": len(boxes),
        "sizes": [(round(b["w"], 1), round(b["h"], 1)) for b in boxes],
        "angles": [round(b["angle"], 1) for b in boxes],
        "at_floor": sum(1 for b in boxes
                        if b["w"] <= size_min + 1e-6 or b["h"] <= size_min + 1e-6),
        "colors": [b["type"] for b in boxes],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--size-min", type=float, default=100.0)
    ap.add_argument("--size-max", type=float, default=400.0)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--min-core-px", type=int, default=2000,
                    help="drop source frames whose content core is smaller than this")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-img", default=OUT_IMG)
    ap.add_argument("--out-lab", default=OUT_LAB)
    ap.add_argument("--limit-src", type=int, default=0, help="debug: use only N source images")
    ap.add_argument("--size-pool", default="",
                    help="JSON file with a list of [w, h] pairs; the side "
                         "lengths are drawn from it instead of --size-min/max")
    ap.add_argument("--stroke-min", type=int, default=2)
    ap.add_argument("--stroke-max", type=int, default=5)
    ap.add_argument("--support-pool", default="",
                    help="JSON list of 4 per-side values.  With "
                         "--stroke-model wobbly they are the target coverage "
                         "of the ideal side line; with the default model the "
                         "stroke is only painted on that fraction of the side")
    ap.add_argument("--recipe-pool", default="",
                    help="JSON list of per box recipes [[cov, missing, "
                         "band_cover] x 4]; used by --stroke-model wobbly")
    ap.add_argument("--stroke-model", default="perfect",
                    choices=["perfect", "wobbly"],
                    help="perfect = straight stroke; wobbly = hand drawn line "
                         "that wanders around the ideal edge (the real data)")
    ap.add_argument("--overshoot", type=float, default=0.0,
                    help="px the stroke runs past the corner")
    ap.add_argument("--shift-min", type=float, default=2.0,
                    help="min displacement (px) of a broken side from the ideal edge")
    ap.add_argument("--shift-max", type=float, default=6.0,
                    help="max displacement (px) of a broken side from the ideal edge")
    ap.add_argument("--wobble-min", type=float, default=0.8,
                    help="min wobble amplitude (px) of a broken side")
    ap.add_argument("--wobble-max", type=float, default=2.5,
                    help="max wobble amplitude (px) of a broken side")
    ap.add_argument("--amp-max", type=float, default=8.0,
                    help="max wobble amplitude (px) of a complete side")
    ap.add_argument("--shift-bias", type=float, default=0.8,
                    help="max parallel bias (px) of a complete side")
    ap.add_argument("--drift-max", type=float, default=0.0,
                    help="max drift (px) of a broken side from one corner to "
                         "the other - the drawn side is then tilted and the "
                         "corners of the drawn rectangle are off by that much")
    ap.add_argument("--broken-amp-max", type=float, default=24.0,
                    help="max wobble amplitude (px) of a broken side")
    ap.add_argument("--nharm", type=int, default=1,
                    help="harmonics of the wobble profile (1 = a single slow "
                         "bow, which is what the real pen does)")
    ap.add_argument("--min-gap", type=float, default=8.0,
                    help="minimum distance between two boxes (px)")
    ap.add_argument("--label-prob", type=float, default=0.0,
                    help="probability that a box gets a red class label "
                         "(the real annotation draws one next to the object)")
    ap.add_argument("--arrow-prob", type=float, default=0.0,
                    help="probability that the label gets a leader arrow that "
                         "points at a corner of the box")
    ap.add_argument("--arrow-corner-prob", type=float, default=0.5,
                    help="probability that the leader arrow lands on a corner "
                         "of the box (else on a random point of a side)")
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.jpg")))
    if not files:
        print("ERROR: no source images in %s" % args.src, flush=True)
        return 1
    if args.limit_src:
        files = files[:args.limit_src]
    n_all = len(files)
    files = usable_sources(files, min_core=args.min_core_px, workers=args.workers,
                           cache=os.path.join(os.path.dirname(MANIFEST),
                                              "emulation_sources.json"))
    print("source images: %d usable / %d" % (len(files), n_all), flush=True)
    if len(files) < 2:
        print("ERROR: not enough usable source images", flush=True)
        return 1

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(files))
    # cycle through the shuffled list so every source is used
    picks = [files[i % len(files)] for i in order[np.arange(args.n) % len(order)]]

    counts = np.array([BOX_COUNT_DIST[k] for k in sorted(BOX_COUNT_DIST)])
    ks = sorted(BOX_COUNT_DIST)
    counts = counts / counts.sum()

    recipe_pool = None
    if args.recipe_pool:
        with open(args.recipe_pool, encoding="utf-8") as f:
            recipe_pool = json.load(f)
        print("recipe pool: %d boxes" % len(recipe_pool), flush=True)

    support_pool = None
    if args.support_pool:
        with open(args.support_pool, encoding="utf-8") as f:
            support_pool = [[float(x) for x in row] for row in json.load(f)]
        print("support pool: %d boxes" % len(support_pool), flush=True)

    pool = None
    if args.size_pool:
        with open(args.size_pool, encoding="utf-8") as f:
            pool = [(float(a), float(b)) for a, b in json.load(f)]
        print("size pool: %d (w,h) pairs" % len(pool), flush=True)

    tasks = []
    furniture = {"label_prob": args.label_prob,
                 "arrow_prob": args.arrow_prob,
                 "corner_prob": args.arrow_corner_prob}
    draw_params = {
        "shift_min": args.shift_min, "shift_max": args.shift_max,
        "wobble_min": args.wobble_min, "wobble_max": args.wobble_max,
        "amp_max": args.amp_max, "shift_bias": args.shift_bias,
        "drift_max": args.drift_max, "broken_amp_max": args.broken_amp_max,
        "nharm": args.nharm,
    }
    for i in range(args.n):
        src = picks[i]
        n_want = int(rng.choice(ks, p=counts))
        tasks.append((i, src, "emul_%06d.jpg" % (i + 1),
                      int(rng.randint(0, 2 ** 31 - 1)), n_want,
                      args.size_min, args.size_max, args.min_coverage,
                      args.jpeg_quality, args.out_img, args.out_lab, pool,
                      args.stroke_min, args.stroke_max, args.min_gap,
                      support_pool, args.overshoot, args.stroke_model,
                      recipe_pool, draw_params, furniture))

    def run_tasks(task_list):
        out = []
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for k, r in enumerate(pool.imap_unordered(process, task_list, chunksize=16)):
                    out.append(r)
                    if (k + 1) % 1000 == 0:
                        print("  %d/%d  %.1fs" % (k + 1, len(task_list), time.time() - t0),
                              flush=True)
        else:
            for k, t in enumerate(task_list):
                out.append(process(t))
        return out

    t0 = time.time()
    results = run_tasks(tasks)
    by_idx = {r["idx"]: r for r in results}

    # a frame whose content cannot host a box is replaced by another frame
    for attempt in range(4):
        empty = [i for i, r in by_idx.items() if r.get("ok") and r["n_boxes"] == 0]
        if not empty:
            break
        print("  %d frames without a box -> re-assigning sources (try %d)"
              % (len(empty), attempt + 1), flush=True)
        retry = []
        for i in empty:
            src = files[(i + attempt + 1) % len(files)]
            n_want = int(rng.choice(ks, p=counts))
            retry.append((i, src, "emul_%06d.jpg" % (i + 1),
                          int(rng.randint(0, 2 ** 31 - 1)), n_want,
                          args.size_min, args.size_max, args.min_coverage,
                          args.jpeg_quality, args.out_img, args.out_lab, pool,
                          args.stroke_min, args.stroke_max, args.min_gap,
                          support_pool, args.overshoot, args.stroke_model,
                          recipe_pool, draw_params, furniture))
        for r in run_tasks(retry):
            by_idx[r["idx"]] = r
    dt = time.time() - t0

    ok = [by_idx[i] for i in sorted(by_idx) if by_idx[i].get("ok")]
    n_boxes = sum(r["n_boxes"] for r in ok)
    per_color = {}
    for r in ok:
        for c in r["colors"]:
            per_color[c] = per_color.get(c, 0) + 1
    per_count = {}
    for r in ok:
        per_count[r["n_boxes"]] = per_count.get(r["n_boxes"], 0) + 1
    sizes = [s for r in ok for s in r["sizes"]]
    n_floor = sum(r.get("at_floor", 0) for r in ok)

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": args.src,
        "n_source_images": len(files),
        "n_images": len(ok),
        "n_boxes": n_boxes,
        "boxes_per_image": {str(k): v for k, v in sorted(per_count.items())},
        "boxes_per_color": per_color,
        "side_min": min(s[0] for s in sizes) if sizes else None,
        "side_max": max(s[0] for s in sizes) if sizes else None,
        "side_mean": round(float(np.mean([s[0] for s in sizes])), 1) if sizes else None,
        "boxes_at_size_floor": n_floor,
        "size_range_requested": [args.size_min, args.size_max],
        "size_pool": args.size_pool,
        "support_pool": args.support_pool,
        "stroke_model": args.stroke_model,
        "recipe_pool": args.recipe_pool,
        "overshoot": args.overshoot,
        "label_prob": args.label_prob,
        "arrow_prob": args.arrow_prob,
        "min_coverage": args.min_coverage,
        "min_gap": args.min_gap,
        "jpeg_quality": args.jpeg_quality,
        "seed": args.seed,
        "elapsed_sec": round(dt, 1),
        "out_img": args.out_img,
        "out_lab": args.out_lab,
    }
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    # per-image index (output -> source + drawn boxes)
    with open(os.path.join(os.path.dirname(args.manifest), "emulation_index.jsonl"),
              "w", encoding="utf-8") as f:
        for r in sorted(ok, key=lambda r: r["idx"]):
            f.write(json.dumps({
                "out": r["out"], "src": r["src"], "n_boxes": r["n_boxes"],
                "sizes": r["sizes"], "angles": r["angles"], "colors": r["colors"],
            }, ensure_ascii=False) + "\n")

    print("\n=== emulation dataset ===", flush=True)
    print("  images      : %d  (%.1fs, %.2fs/img)" % (len(ok), dt, dt / max(1, len(ok))), flush=True)
    print("  boxes       : %d  (%.2f /img)" % (n_boxes, n_boxes / max(1, len(ok))), flush=True)
    print("  per image   : %s" % manifest["boxes_per_image"], flush=True)
    print("  per color   : %s" % per_color, flush=True)
    print("  side range  : %.1f .. %.1f (mean %.1f)" % (
        manifest["side_min"], manifest["side_max"], manifest["side_mean"]), flush=True)
    print("  manifest    : %s" % MANIFEST, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())