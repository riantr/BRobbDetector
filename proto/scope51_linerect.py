"""LOOP-51: line-based rectangle detection on the red mask.

Why
---
The hole pipeline (scope45) reaches 926/1121 GT boxes at D=5, but every
remaining FN is a box whose INTERIOR is not a single clean hole:

  * a stroke crosses the interior  -> the hole is split into pieces
  * the interior merges with a neighbouring box's interior through a gap in
    the annotation -> the hole is too big and not rectangular any more

In both cases the 4 red strokes of the box ARE present in the mask - only the
enclosed REGION is wrong.  So detect the rectangle from the strokes.

Algorithm
  1. Radon-style projection of the red pixels on 180 orientations; the band
     centres are candidate stroke centrelines.  Only the most "peaky"
     orientations are swept (a box's own orientation dominates the score);
     the perpendicular of every swept angle is swept as well, because a
     stroke perpendicular to the projection axis projects to a single spike
     and is invisible to the collinearity test of that angle.
  2. keep the bands whose pixels are collinear (real strokes) and use their
     own total-least-squares direction.
  3. group the strokes by direction; for every perpendicular pair of
     direction families build all (2 x 2) rectangles.  The corners are the
     intersections of the actual fitted lines.
  4. score every candidate: side lengths, perimeter red support, interior red
     fraction.

Everything is vectorised: the candidate rectangles of one family pair are
scored in a single numpy pass.
"""
import os, sys, math, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import red_mask
from canny_quad import denoise

ROOT = r"D:\src\CodeArts\BRobbDetector"

LR_DEFAULT = {
    "lr_angle_step": 1.0,     # degrees between projection angles
    "lr_keep_angles": 4,      # orientations kept after the peakiness score
    "lr_min_pixels": 6,       # min red pixels in a stroke band
    "lr_band": 2.5,           # half width of the band around a band centre
    "lr_collinear_tol": 3.2,  # max RMS residual of the band (px)
    "lr_ang_tol": 6.0,        # max |fitted angle - projection angle| (deg)
    "lr_max_lines": 14,       # cap the strokes per direction family
    "lr_dup_ang": 3.0,        # two strokes closer than this are the same
    "lr_dup_dist": 3.0,       # ... and this close (px)
    "lr_fam_tol": 4.0,        # direction family tolerance (deg)
    "lr_min_side": 12.0,
    "lr_max_side": 800.0,
    "lr_max_aspect": 12.0,
    "lr_perim": 0.50,         # min fraction of the perimeter that is red
    "lr_interior": 0.40,      # max red fraction inside the rectangle
    "lr_samples": 16,         # samples per side for the perimeter support
}


def _perp(v):
    return np.array([-v[1], v[0]])


def _band_centres(hist, min_pixels, band):
    """Band centres of a projection histogram (flat-topped stroke bands)."""
    n = len(hist)
    if n == 0:
        return []
    w = 2 * int(band) + 1
    sm = np.convolve(hist.astype(np.float64), np.ones(w), mode="same")
    idx = np.nonzero((sm[1:-1] >= sm[:-2]) & (sm[1:-1] > sm[2:]))[0] + 1
    idx = idx[hist[idx] >= min_pixels]
    if len(idx) == 0:
        return []
    order = idx[np.argsort(-hist[idx])]
    keep = []
    r = int(band) + 3
    for i in order:
        if all(abs(i - k) > r for k in keep):
            keep.append(i)
    out = []
    for k in sorted(keep):
        lo = max(k - int(band), 0)
        hi = min(k + int(band) + 1, n)
        seg = hist[lo:hi].astype(np.float64)
        if seg.sum() <= 0:
            continue
        c = (np.arange(lo, hi) * seg).sum() / seg.sum()
        out.append((c, int(seg.sum())))
    return out


def _tls(pts):
    """Total least squares line fit -> (point, direction, rms residual)."""
    mu = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    d = vt[0]
    resid = np.abs((pts - mu) @ _perp(d))
    return mu, d, float(np.sqrt((resid ** 2).mean()))


def _angles(pts, cfg):
    """Projection angles ranked by peakiness (sum of squared bin counts)."""
    step = cfg["lr_angle_step"]
    n_ang = int(round(180.0 / step))
    scores = []
    for ai in range(n_ang):
        ang = ai * step
        th = math.radians(ang)
        e = np.array([math.cos(th), math.sin(th)])
        c = pts @ _perp(e)
        lo = c.min()
        hist = np.bincount(np.round(c - lo).astype(np.int32))
        scores.append(float((hist.astype(np.float64) ** 2).sum()))
    order = np.argsort(scores)[::-1]
    keep = []
    for ai in order:
        ang = float(ai * step)
        if keep and min(abs(ang - k) for k in keep) < 4.0:
            continue
        keep.append(ang)
        if len(keep) >= cfg["lr_keep_angles"]:
            break
    out = []
    for ang in keep:
        out.append(ang)
        perp = (ang + 90.0) % 180.0
        if all(min(abs(perp - k), 180.0 - abs(perp - k)) > 0.5 for k in out):
            out.append(perp)
    return out


def _strokes_at(pts, ang, cfg):
    """Stroke centrelines whose direction is close to `ang`."""
    th = math.radians(ang)
    e = np.array([math.cos(th), math.sin(th)])
    nvec = _perp(e)
    c = pts @ nvec
    lo = c.min()
    hist = np.bincount(np.round(c - lo).astype(np.int32))
    out = []
    for centre, _cnt in _band_centres(hist, cfg["lr_min_pixels"], cfg["lr_band"]):
        off = centre + lo
        sel = np.abs(c - off) <= cfg["lr_band"]
        if int(sel.sum()) < cfg["lr_min_pixels"]:
            continue
        sub = pts[sel]
        mu, dd, rms = _tls(sub)
        fang = math.degrees(math.atan2(dd[1], dd[0])) % 180.0
        diff = abs(fang - ang)
        diff = min(diff, 180.0 - diff)
        if diff > cfg["lr_ang_tol"] or rms > cfg["lr_collinear_tol"]:
            continue
        if fang > 90.0:               # canonical sign
            dd = -dd
            fang -= 180.0
        out.append({"ang": fang, "dir": dd, "point": mu,
                    "count": int(sel.sum()), "rms": rms})
    return out


def _dedup(strokes, cfg):
    """Drop strokes that were found by several projection angles."""
    keep = []
    for S in sorted(strokes, key=lambda x: -x["count"]):
        dup = False
        for T in keep:
            d = abs(S["ang"] - T["ang"])
            d = min(d, 180.0 - d)
            if d > cfg["lr_dup_ang"]:
                continue
            dist = abs(float((S["point"] - T["point"]) @ _perp(T["dir"])))
            if dist <= cfg["lr_dup_dist"]:
                dup = True
                break
        if not dup:
            keep.append(S)
    return keep


def _families(strokes, cfg):
    """Group strokes by direction (mod 180)."""
    fams = []
    for S in sorted(strokes, key=lambda x: -x["count"]):
        placed = False
        for f in fams:
            d = abs(S["ang"] - f["ang"])
            d = min(d, 180.0 - d)
            if d <= cfg["lr_fam_tol"]:
                f["lines"].append(S)
                placed = True
                break
        if not placed:
            fams.append({"ang": S["ang"], "lines": [S]})
    for f in fams:
        f["lines"] = f["lines"][:cfg["lr_max_lines"]]
    return [f for f in fams if len(f["lines"]) >= 2]


def _perim_support_batch(mask, corners, nsamp):
    """Red fraction of the perimeter of every candidate rectangle.

    The perimeter is sampled `nsamp` times per side and the four side
    supports are combined weighted by the side length.  The mask should be
    slightly dilated so that the sampling density does not bias long sides.
    """
    H, W = mask.shape
    n = len(corners)
    nsamp = int(nsamp)
    ts = (np.arange(nsamp) + 0.5) / nsamp
    P = np.empty((n, 4, nsamp, 2), np.float64)
    L = np.empty((n, 4), np.float64)
    for i in range(4):
        a = corners[:, i][:, None, :]
        b = corners[:, (i + 1) % 4][:, None, :]
        P[:, i] = a + ts[None, :, None] * (b - a)
        L[:, i] = np.linalg.norm(b - a, axis=2)[:, 0]
    Q = P.reshape(-1, 2)
    xs = np.clip(np.round(Q[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(Q[:, 1]).astype(int), 0, H - 1)
    red = (mask[ys, xs] > 0).reshape(n, 4, nsamp)
    side = red.mean(axis=2)
    w = L / np.maximum(L.sum(axis=1, keepdims=True), 1e-6)
    return (side * w).sum(axis=1)


def _interior_red_batch(mask, corners):
    """Red fraction inside every candidate rectangle."""
    H, W = mask.shape
    out = np.ones(len(corners), np.float64)
    for i, q in enumerate(corners):
        x0 = int(max(q[:, 0].min(), 0)); x1 = int(min(q[:, 0].max(), W - 1))
        y0 = int(max(q[:, 1].min(), 0)); y1 = int(min(q[:, 1].max(), H - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = mask[y0:y1 + 1, x0:x1 + 1]
        if sub.size < 25:
            continue
        cnt = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.int32)
        cv2.fillPoly(cnt, [q.astype(np.int32) - np.array([x0, y0])], 1)
        inside = cnt > 0
        if not inside.any():
            continue
        out[i] = float((sub[inside] > 0).mean())
    return out


def _isect_matrix(PA, DA, PB, DB):
    """Intersections of every line of A with every line of B."""
    nA, nB = len(PA), len(PB)
    p1 = np.repeat(PA, nB, axis=0)
    d1 = np.repeat(DA, nB, axis=0)
    p2 = np.tile(PB, (nA, 1))
    d2 = np.tile(DB, (nA, 1))
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    ok = np.abs(den) > 1e-6
    t = np.where(ok, ((p2[:, 0] - p1[:, 0]) * d2[:, 1]
                      - (p2[:, 1] - p1[:, 1]) * d2[:, 0]) / np.where(ok, den, 1.0), 0.0)
    X = p1 + t[:, None] * d1
    return X.reshape(nA, nB, 2), ok.reshape(nA, nB)


def detect_linrect(img, p, cfg, pipe=None):
    """Rectangles built from the red strokes of the mask."""
    mask = red_mask(img, p["sat_min"], p["val_min"], p["hue_max"])
    if p["denoise_min_area"] > 0:
        mask = denoise(mask, p["denoise_min_area"])
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return []
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    # dilated mask for the perimeter support: the sampling density must not
    # bias long sides (a 7px stroke sampled every 5px would read as ~70%)
    mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) > 0

    strokes = []
    for ang in _angles(pts, cfg):
        strokes += _strokes_at(pts, ang, cfg)
    strokes = _dedup(strokes, cfg)
    fams = _families(strokes, cfg)
    if len(fams) < 2:
        return []

    out = []
    for fi, fa in enumerate(fams):
        for fb in fams[fi + 1:]:
            diff = abs(fa["ang"] - fb["ang"]) % 180.0
            diff = min(diff, 180.0 - diff)
            if abs(diff - 90.0) > 8.0:
                continue
            PA = np.array([L["point"] for L in fa["lines"]])
            DA = np.array([L["dir"] for L in fa["lines"]])
            PB = np.array([L["point"] for L in fb["lines"]])
            DB = np.array([L["dir"] for L in fb["lines"]])
            nA = _perp(DA[0])
            nB = _perp(DB[0])
            oA = PA @ nA
            oB = PB @ nB
            ia, ja = np.triu_indices(len(PA), 1)
            ib, jb = np.triu_indices(len(PB), 1)
            sA = np.abs(oA[ja] - oA[ia])
            sB = np.abs(oB[jb] - oB[ib])
            okA = (sA >= cfg["lr_min_side"]) & (sA <= cfg["lr_max_side"])
            okB = (sB >= cfg["lr_min_side"]) & (sB <= cfg["lr_max_side"])
            if not okA.any() or not okB.any():
                continue
            A_idx = np.nonzero(okA)[0]
            B_idx = np.nonzero(okB)[0]
            AA, BB = np.meshgrid(A_idx, B_idx, indexing="ij")
            AA = AA.ravel(); BB = BB.ravel()
            sa = sA[AA]; sb = sB[BB]
            aspect = np.maximum(sa, sb) / np.maximum(np.minimum(sa, sb), 1e-6)
            keep = aspect <= cfg["lr_max_aspect"]
            if not keep.any():
                continue
            AA, BB = AA[keep], BB[keep]
            ia2 = ia[AA]; ja2 = ja[AA]
            ib2 = ib[BB]; jb2 = jb[BB]
            X, okm = _isect_matrix(PA, DA, PB, DB)
            corners = np.empty((len(AA), 4, 2), np.float64)
            corners[:, 0] = X[ia2, ib2]
            corners[:, 1] = X[ia2, jb2]
            corners[:, 2] = X[ja2, jb2]
            corners[:, 3] = X[ja2, ib2]
            good = okm[ia2, ib2] & okm[ia2, jb2] & okm[ja2, jb2] & okm[ja2, ib2]
            if not good.any():
                continue
            corners = corners[good]
            inb = ((corners[:, :, 0].min(axis=1) > -5)
                   & (corners[:, :, 0].max(axis=1) < img.shape[1] + 5)
                   & (corners[:, :, 1].min(axis=1) > -5)
                   & (corners[:, :, 1].max(axis=1) < img.shape[0] + 5))
            if not inb.any():
                continue
            corners = corners[inb]
            sup = _perim_support_batch(mask_d, corners, cfg["lr_samples"])
            sel = sup >= cfg["lr_perim"]
            if not sel.any():
                continue
            corners = corners[sel]
            inter = _interior_red_batch(mask, corners)
            sel = inter <= cfg["lr_interior"]
            out.extend(list(corners[sel]))
    return out


if __name__ == "__main__":
    from common import IMG
    from canny_quad import get_ids
    t0 = time.time()
    ids = get_ids()[:20]
    cfg = dict(LR_DEFAULT)
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        r = detect_linrect(img, {"sat_min": 100, "val_min": 140, "hue_max": 4,
                                  "denoise_min_area": 40}, cfg)
        print(iid, len(r), f"{time.time()-t0:.1f}s")