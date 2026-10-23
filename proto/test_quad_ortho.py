"""Unit tests for proto/quad_ortho.py (LOOP-2 orthogonality rule).

Run: python proto/test_quad_ortho.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from quad_ortho import interior_angles, is_orthogonal_quad

W, H = 800, 600
FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")
    if not ok:
        FAILS.append(name)


def rot_rect(cx, cy, w, h, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (pts @ R.T) + np.array([cx, cy])


def main():
    print("Unit tests: quad_ortho")
    print("A) interior_angles sanity")
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    angs = interior_angles(sq)
    check("square all 90", all(abs(a - 90) < 1e-6 for a in angs), True)

    print("\nB) exactly-four-edges / rotated rects")
    check("axis-aligned rect", is_orthogonal_quad(rot_rect(400, 300, 100, 60, 0), W, H), True)
    check("rotated 45 rect", is_orthogonal_quad(rot_rect(400, 300, 120, 80, 45), W, H), True)
    check("rotated 17 rect", is_orthogonal_quad(rot_rect(400, 300, 90, 40, 17), W, H), True)
    triangle = np.array([[100, 100], [200, 100], [150, 200]], float)
    check("triangle (3 edges)", is_orthogonal_quad(triangle, W, H), False)
    pentagon = np.array([[0, 0], [10, 0], [15, 10], [5, 18], [-5, 10]], float)
    check("pentagon (5 edges)", is_orthogonal_quad(pentagon, W, H), False)

    print("\nC) non-rectangular quads must be rejected (deviation > 15 deg)")
    para = np.array([[100, 100], [200, 100], [260, 160], [160, 160]], float)  # 45/135
    check("parallelogram (45deg) rejected", is_orthogonal_quad(para, W, H), False)
    trap = np.array([[100, 100], [220, 100], [180, 180], [140, 180]], float)  # ~63/117
    ok, d = is_orthogonal_quad(trap, W, H, return_details=True)
    check("strong trapezoid rejected", ok, False)
    rhom = np.array([[100, 100], [180, 100], [220, 170], [140, 170]], float)  # ~60/120
    check("rhombus (60deg) rejected", is_orthogonal_quad(rhom, W, H), False)
    moved = np.array([[0, 0], [100, 40], [100, 60], [0, 60]], float)
    check("rect with one corner moved 40px rejected",
          is_orthogonal_quad(moved, W, H), False)

    print("\nC2) mild non-rectangularity is tolerated (pixel-noise band)")
    mild = np.array([[100, 100], [200, 100], [180, 180], [120, 180]], float)  # ~76/104
    ok15, d15 = is_orthogonal_quad(mild, W, H, tol=15, return_details=True)
    ok5, d5 = is_orthogonal_quad(mild, W, H, tol=5, return_details=True)
    print(f"    mild quad max_dev={d15['max_dev']:.1f} deg -> tol15={ok15} tol5={ok5}")
    check("mild quad accepted at tol=15 (by design)", ok15, True)
    check("mild quad rejected at tol=5", ok5, False)

    print("\nD) a vertex on the image edge invalidates the rectangle")
    # rect whose left corners sit exactly on x=0
    border = np.array([[0, 100], [120, 100], [120, 200], [0, 200]], float)
    check("rect with 2 corners on left edge -> invalid",
          is_orthogonal_quad(border, W, H), False)
    # clipped box: 2 corners at the image corner
    clip3 = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)
    check("rect with 2 corners at image corner -> invalid",
          is_orthogonal_quad(clip3, W, H), False)
    # a single vertex on the edge is enough
    one = np.array([[0, 100], [120, 100], [120, 200], [1, 200]], float)
    check("rect with 1 corner on the edge -> invalid",
          is_orthogonal_quad(one, W, H), False)
    # 1 px inside the edge is still valid
    inside = np.array([[1, 100], [120, 100], [120, 200], [1, 200]], float)
    check("rect 1px inside the edge -> valid",
          is_orthogonal_quad(inside, W, H), True)
    # the outermost pixel ring on the right / bottom counts as well
    right = np.array([[W - 1, 100], [W - 1, 200], [W - 1, 200], [600, 100]], float)
    check("rect on the right edge -> invalid",
          is_orthogonal_quad(right, W, H), False)
    # non-orthogonal interior corner + boundary corners -> reject
    bad_clip = np.array([[0, 0], [100, 0], [140, 80], [0, 80]], float)  # corner(2) not 90
    check("clipped but interior corner not 90 -> reject",
          is_orthogonal_quad(bad_clip, W, H), False)
    # all four corners on boundary -> invalid
    allb = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    check("all corners on boundary -> invalid", is_orthogonal_quad(allb, W, H), False)

    print("\nE) degenerate input")
    check("duplicate corner rejected",
          is_orthogonal_quad(np.array([[10, 10], [10, 10], [50, 10], [50, 50]], float), W, H),
          False)
    check("NaN-free tiny quad", is_orthogonal_quad(np.array([[10, 10], [11, 10], [11, 11], [10, 11]], float), W, H), True)

    print("\nF) tolerance behaviour")
    near = rot_rect(400, 300, 100, 60, 0)
    near[1] += np.array([0, 6])  # ~80 deg-ish at two corners
    ok10, d10 = is_orthogonal_quad(near, W, H, tol=5, return_details=True)
    ok20, d20 = is_orthogonal_quad(near, W, H, tol=20, return_details=True)
    print(f"    max_dev={d10['max_dev']:.2f} deg -> tol5={ok10} tol20={ok20}")

    print(f"\n{'='*50}")
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()