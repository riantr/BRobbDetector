import os, sys, glob, json
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import load_expected, IMG, LAB

def analyze_color_scattering(n_images=20):
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:n_images]

    hsv_pixels = []
    lab_pixels = []
    rgb_pixels = []
    ycc_pixels = []

    for k in ids:
        path = os.path.join(IMG, k + ".jpg")
        img = cv2.imread(path)
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        m1 = cv2.inRange(hsv, (0, 80, 110), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 80, 110), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)

        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            continue

        hsv_pixels.append(hsv[ys, xs])
        lab_pixels.append(lab[ys, xs])
        rgb_pixels.append(img[ys, xs][:, ::-1])  # BGR -> RGB
        ycc_pixels.append(ycc[ys, xs])

    hsv_arr = np.vstack(hsv_pixels)
    lab_arr = np.vstack(lab_pixels)
    rgb_arr = np.vstack(rgb_pixels)
    ycc_arr = np.vstack(ycc_pixels)

    print(f"Analyzed {len(hsv_arr)} red pixels from {len(hsv_pixels)} images\n")

    def report(name, arr, labels):
        print(f"=== {name} ===")
        for i, label in enumerate(labels):
            ch = arr[:, i].astype(np.float64)
            mean = ch.mean()
            std = ch.std()
            cv = std / mean * 100 if mean != 0 else 0
            p5 = np.percentile(ch, 5)
            p95 = np.percentile(ch, 95)
            rng = p95 - p5
            print(f"  {label}: mean={mean:.1f} std={std:.1f} CV={cv:.1f}% P5={p5:.1f} P95={p95:.1f} range90={rng:.1f}")
        print()

    report("HSV", hsv_arr, ["H", "S", "V"])
    report("Lab", lab_arr, ["L*", "a*", "b*"])
    report("RGB", rgb_arr, ["R", "G", "B"])
    report("YCbCr", ycc_arr, ["Y", "Cr", "Cb"])

    # Lab Delta E analysis
    lab_mean = lab_arr.mean(axis=0)
    dL = lab_arr[:, 0].astype(np.float64) - lab_mean[0]
    da = lab_arr[:, 1].astype(np.float64) - lab_mean[1]
    db = lab_arr[:, 2].astype(np.float64) - lab_mean[2]
    deltaE = np.sqrt(dL**2 + da**2 + db**2)
    print(f"=== Lab Delta E (from centroid) ===")
    print(f"  Centroid: L*={lab_mean[0]:.1f} a*={lab_mean[1]:.1f} b*={lab_mean[2]:.1f}")
    print(f"  Delta E: mean={deltaE.mean():.1f} std={deltaE.std():.1f} P50={np.percentile(deltaE, 50):.1f} P90={np.percentile(deltaE, 90):.1f} P95={np.percentile(deltaE, 95):.1f} max={deltaE.max():.1f}")
    print()

    # HSV hue analysis (circular)
    hues = hsv_arr[:, 0].astype(np.float64)
    hues_rad = hues * np.pi / 180.0
    mean_sin = np.sin(hues_rad).mean()
    mean_cos = np.cos(hues_rad).mean()
    mean_hue = np.arctan2(mean_sin, mean_cos) * 180.0 / np.pi
    if mean_hue < 0:
        mean_hue += 180
    circular_std = np.sqrt(-2 * np.log(np.sqrt(mean_sin**2 + mean_cos**2))) * 180.0 / np.pi
    print(f"=== HSV Hue (circular) ===")
    print(f"  Mean hue: {mean_hue:.1f}°")
    print(f"  Circular std: {circular_std:.1f}°")
    print(f"  Hue values at 0° and 180° indicate red wrap-around problem")
    print()

    # Per-channel scattering comparison (normalized std)
    print("=== Scattering Comparison (std / range) ===")
    spaces = [
        ("HSV-H", hsv_arr[:, 0], 180),
        ("HSV-S", hsv_arr[:, 1], 256),
        ("HSV-V", hsv_arr[:, 2], 256),
        ("Lab-L", lab_arr[:, 0], 256),
        ("Lab-a", lab_arr[:, 1], 256),
        ("Lab-b", lab_arr[:, 2], 256),
        ("RGB-R", rgb_arr[:, 0], 256),
        ("RGB-G", rgb_arr[:, 1], 256),
        ("RGB-B", rgb_arr[:, 2], 256),
    ]
    for name, ch, rng in spaces:
        std = ch.astype(np.float64).std()
        print(f"  {name}: std={std:.1f} normalized={std/rng*100:.1f}%")
    print()

    # Lab a*-b* 2D scatter (this is the key for red detection)
    print("=== Lab a*-b* 2D Analysis ===")
    a = lab_arr[:, 1].astype(np.float64)
    b = lab_arr[:, 2].astype(np.float64)
    print(f"  a*: mean={a.mean():.1f} std={a.std():.1f}")
    print(f"  b*: mean={b.mean():.1f} std={b.std():.1f}")
    print(f"  Chroma C* = sqrt(a²+b²): mean={np.sqrt(a**2+b**2).mean():.1f} std={np.sqrt(a**2+b**2).std():.1f}")
    print(f"  Hue angle h = atan2(b,a): mean={np.degrees(np.arctan2(b, a)).mean():.1f}° std={np.degrees(np.arctan2(b, a)).std():.1f}°")

if __name__ == "__main__":
    analyze_color_scattering()