"""Reusable orthogonality check for detected quads (LOOP-2).

Project rule (2026-10-21): a detected rectangle is only valid when

  1. it has exactly four edges,
  2. every corner is orthogonal (90 deg) to its adjacent edges, and
  3. NONE of its four vertices lies on the image edge.

Rule 3 replaced the former behaviour, which *waived* the orthogonality check
at a corner that lay on the image boundary (the box may be clipped).  A
rectangle that touches the image edge is now rejected outright: the human
annotation was clipped by the image frame, so the target is not a valid
rectangle.

`edge_band` is the half-width (px) of the border ring that counts as "the
image edge": 0 = the outermost pixel ring (the literal rule), larger values
are more forgiving.  `waive_boundary` keeps the legacy behaviour of exempting
corners that are close to (but not on) the edge from the orthogonality check;
it is OFF by default because a corner near the edge was only ever waived to
tolerate the clipped boxes that the new rule rejects outright.

`boundary_band` is kept for API compatibility with the legacy callers (it is
only used when `waive_boundary` is on).

This module is the single source of truth for the rule; line/corner-based
detectors (LSD, Hough) must call `is_orthogonal_quad` as a post-grouping filter.
"""
import math
import numpy as np

EPS = 1e-6


def interior_angles(corners):
    """Interior angle (deg) at each corner of a polygon. NaN for degenerate corners."""
    c = np.asarray(corners, float)
    n = len(c)
    out = []
    for i in range(n):
        cur = c[i]
        v1 = c[(i - 1) % n] - cur
        v2 = c[(i + 1) % n] - cur
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < EPS or n2 < EPS:
            out.append(float("nan"))
            continue
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        out.append(math.degrees(math.acos(cosv)))
    return out


def corner_on_boundary(pt, img_w, img_h, band):
    """True when the corner lies in the border ring of half-width `band`.

    The test is symmetric: the outermost pixel ring is x == 0 / x == img_w - 1
    (and likewise for y).  `band` widens the ring inwards on all four sides.
    """
    x, y = pt[0], pt[1]
    return (x <= band or x >= (img_w - 1 - band)
            or y <= band or y >= (img_h - 1 - band))


def quad_on_boundary(corners, img_w, img_h, band):
    """True when ANY vertex of the quad lies in the border ring."""
    c = np.asarray(corners, float)
    for i in range(len(c)):
        if corner_on_boundary(c[i], img_w, img_h, band):
            return True
    return False


def is_orthogonal_quad(corners, img_w, img_h, tol=15.0, boundary_band=6.0,
                       edge_band=0.0, waive_boundary=False, return_details=False):
    """Return True iff the quad has exactly 4 edges, every corner is orthogonal
    (within `tol` degrees of 90) and no corner lies on the image edge.

    `edge_band` (default 0) defines the border ring that invalidates the quad.
    `waive_boundary` (default off) keeps the legacy behaviour of exempting
    corners within `boundary_band` of the edge from the orthogonality check (a
    corner that is ON the edge always invalidates the quad, waived or not).
    """
    c = np.asarray(corners, float)
    if c.ndim != 2 or c.shape[0] != 4 or c.shape[1] != 2:
        if return_details:
            return False, {"reason": "not_a_quad", "n": int(c.shape[0]) if c.ndim == 2 else 0}
        return False

    angles = interior_angles(c)
    checked = []
    waived = []
    ok = True
    reason = ""
    for i in range(4):
        if corner_on_boundary(c[i], img_w, img_h, edge_band):
            ok = False
            reason = "vertex_on_edge"
            checked.append((i, angles[i], False))
            break
        if waive_boundary and corner_on_boundary(c[i], img_w, img_h, boundary_band):
            waived.append(i)
            continue
        a = angles[i]
        if not math.isfinite(a):
            ok = False
            checked.append((i, a, False))
            continue
        good = abs(a - 90.0) <= tol
        ok = ok and good
        checked.append((i, a, good))

    if not checked:
        ok = True  # every corner was waived -> nothing left to check

    if return_details:
        return ok, {
            "angles": angles,
            "checked": checked,
            "waived": waived,
            "reason": reason,
            "max_dev": max((abs(a - 90.0) for _, a, _ in checked if math.isfinite(a)),
                           default=0.0),
        }
    return ok