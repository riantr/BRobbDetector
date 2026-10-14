import os, sys, json
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import load_expected, IMG, LAB

def analyze_svd_spectrum(n_images=15):
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:n_images]

    all_singular_values_a = []
    all_singular_values_b = []
    box_info = []

    for k_id in ids:
        path = os.path.join(IMG, k_id + ".jpg")
        img = cv2.imread(path)
        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        m1 = cv2.inRange(hsv, (0, 80, 110), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 80, 110), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)

        ys, xs = np.where(mask > 0)
        if len(xs) < 100:
            continue

        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        if w < 20 or h < 20:
            continue

        a_roi = lab[y0:y1+1, x0:x1+1, 1].astype(np.float64)
        b_roi = lab[y0:y1+1, x0:x1+1, 2].astype(np.float64)

        try:
            u_a, s_a, vt_a = np.linalg.svd(a_roi, full_matrices=False)
            u_b, s_b, vt_b = np.linalg.svd(b_roi, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        all_singular_values_a.append(s_a[:50])
        all_singular_values_b.append(s_b[:50])
        box_info.append({"id": k_id, "w": w, "h": h, "n_red": len(xs)})

    if not all_singular_values_a:
        print("No valid images found")
        return

    print(f"Analyzed {len(all_singular_values_a)} images\n")

    print("=== Bounding box sizes ===")
    for info in box_info[:10]:
        print(f"  {info['id'][:25]} {info['w']}x{info['h']} red_pixels={info['n_red']}")

    print("\n=== Singular value energy distribution (a* channel) ===")
    max_k = min(30, min(len(s) for s in all_singular_values_a))
    for k in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]:
        if k > max_k:
            break
        energies = []
        for s in all_singular_values_a:
            total = np.sum(s[:max_k]**2)
            top_k = np.sum(s[:k]**2)
            energies.append(top_k / total * 100 if total > 0 else 0)
        print(f"  rank {k:2d}: {np.mean(energies):.1f}% energy (min={np.min(energies):.1f} max={np.max(energies):.1f})")

    print("\n=== Singular value energy distribution (b* channel) ===")
    for k in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]:
        if k > max_k:
            break
        energies = []
        for s in all_singular_values_b:
            total = np.sum(s[:max_k]**2)
            top_k = np.sum(s[:k]**2)
            energies.append(top_k / total * 100 if total > 0 else 0)
        print(f"  rank {k:2d}: {np.mean(energies):.1f}% energy (min={np.min(energies):.1f} max={np.max(energies):.1f})")

    print("\n=== Singular value decay (a* channel, first image) ===")
    s = all_singular_values_a[0]
    for i in range(min(20, len(s))):
        print(f"  σ[{i:2d}] = {s[i]:.1f}  (σ[{i}]/σ[0] = {s[i]/s[0]:.4f})")

    print("\n=== Reconstruction error vs rank (a* channel) ===")
    for k in [1, 2, 3, 5, 8, 10, 15, 20]:
        if k > max_k:
            break
        errors = []
        for s in all_singular_values_a:
            tail = np.sum(s[k:max_k]**2)
            errors.append(np.sqrt(tail))
        print(f"  rank {k:2d}: Frobenius error = {np.mean(errors):.1f} (min={np.min(errors):.1f} max={np.max(errors):.1f})")

if __name__ == "__main__":
    analyze_svd_spectrum()