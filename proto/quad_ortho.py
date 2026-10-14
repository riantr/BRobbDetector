"""Reusable orthogonality check for detected quads (LOOP-2).

Project rule: every detected target must have exactly four edges, and each edge
must be orthogonal (90 deg) to its adjacent edges. The check is waived at a
corner iff that corner lies on the image boundary (the box may be clipped).

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
    x, y = pt[0], pt[1]
    return (x <= band or x >= img_w - band or y <= band or y >= img_h - band)


def is_orthogonal_quad(corners, img_w, img_h, tol=15.0, boundary_band=6.0,
                       return_details=False):
    """Return True iff the quad has exactly 4 edges and every non-boundary corner
    is orthogonal (within `tol` degrees of 90).

    A corner lying on the image boundary is waived. If every corner is on the
    boundary the quad cannot be judged and is accepted.
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
    for i in range(4):
        if corner_on_boundary(c[i], img_w, img_h, boundary_band):
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
        ok = True  # all corners on boundary -> waived entirely

    if return_details:
        return ok, {
            "angles": angles,
            "checked": checked,
            "waived": waived,
            "max_dev": max((abs(a - 90.0) for _, a, _ in checked if math.isfinite(a)),
                           default=0.0),
        }
    return ok