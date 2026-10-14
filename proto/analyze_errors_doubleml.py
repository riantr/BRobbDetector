"""
Copy detection error data to Image@hard_20261010 / Label@hard_20261010 / Det@hard_20261010,
build a per-image panel with rich features, and run DoubleML causal analysis.
"""
import os, sys, json, shutil, csv, re, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from common import IMG, LAB, DATA, load_expected, iou_obb, red_mask

BENCH = r"D:\src\CodeArts\BRobbDetector\benchmark"
BENCH_JSON = os.path.join(BENCH, "m8_growbox_v3_final.json")
FN_DETAIL = os.path.join(BENCH, "fn_detail.json")
OUT_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"

HARD_DATE = "20261010"
HARD_IMG = os.path.join(DATA, f"Image@hard_{HARD_DATE}")
HARD_LAB = os.path.join(DATA, f"Label@hard_{HARD_DATE}")
HARD_DET = os.path.join(DATA, f"Det@hard_{HARD_DATE}")
PANEL_CSV = os.path.join(BENCH, f"panel_hard_{HARD_DATE}.csv")
REPORT_MD = os.path.join(BENCH, f"doubleml_report_{HARD_DATE}.md")


def parse_points(s):
    pts = []
    for p in s.split(';'):
        x, y = p.split(',')
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)


def load_gt_boxes(image_id):
    path = os.path.join(LAB, image_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]


def load_det_boxes(image_id):
    path = os.path.join(OUT_DIR, image_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_points(info["points"]) for info in data.get("infos", [])]


def box_geometry(pts):
    """Return (cx, cy, w, h, area, aspect, min_side, max_side) for 4-corner OBB."""
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    w, h = abs(w), abs(h)
    min_s, max_s = min(w, h), max(w, h)
    area = w * h
    aspect = max_s / min_s if min_s > 0 else 0
    return cx, cy, w, h, area, aspect, min_s, max_s


def extract_image_features(image_id):
    """Extract image-level features: dimensions, red density, component count, etc."""
    img_path = os.path.join(IMG, image_id + ".jpg")
    img = cv2.imread(img_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    img_area = H * W

    mask = red_mask(img)
    red_count = np.count_nonzero(mask)
    red_density = red_count / img_area

    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_comp = max(n_comp - 1, 0)

    comp_areas = []
    if n_comp > 0:
        for i in range(1, n_comp + 1):
            comp_areas.append(stats[i, cv2.CC_STAT_AREA])
    comp_areas = np.array(comp_areas) if comp_areas else np.array([0])

    largest_comp_area = comp_areas.max() if len(comp_areas) > 0 else 0
    largest_comp_frac = largest_comp_area / img_area

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    return {
        "img_width": W,
        "img_height": H,
        "img_area": img_area,
        "red_pixel_count": red_count,
        "red_density": red_density,
        "n_red_components": n_comp,
        "largest_comp_frac": largest_comp_frac,
        "brightness": brightness,
        "contrast": contrast,
    }


def extract_box_features(boxes):
    """Extract aggregate statistics for a list of OBB boxes."""
    if not boxes:
        return {
            "n_boxes": 0, "avg_area": 0, "avg_aspect": 0, "avg_min_side": 0,
            "avg_max_side": 0, "std_aspect": 0, "min_aspect": 0, "max_aspect": 0,
            "avg_cx": 0, "avg_cy": 0, "spread_x": 0, "spread_y": 0,
            "box_density": 0, "has_strip": 0, "n_strip": 0,
        }
    geos = [box_geometry(b) for b in boxes]
    aspects = np.array([g[5] for g in geos])
    areas = np.array([g[4] for g in geos])
    min_sides = np.array([g[6] for g in geos])
    max_sides = np.array([g[7] for g in geos])
    cxs = np.array([g[0] for g in geos])
    cys = np.array([g[1] for g in geos])

    n = len(boxes)
    strip_flags = [1 if (g[6] / g[7] < 0.5) else 0 for g in geos]

    return {
        "n_boxes": n,
        "avg_area": float(areas.mean()),
        "avg_aspect": float(aspects.mean()),
        "avg_min_side": float(min_sides.mean()),
        "avg_max_side": float(max_sides.mean()),
        "std_aspect": float(aspects.std()),
        "min_aspect": float(aspects.min()),
        "max_aspect": float(aspects.max()),
        "avg_cx": float(cxs.mean()),
        "avg_cy": float(cys.mean()),
        "spread_x": float(cxs.std()),
        "spread_y": float(cys.std()),
        "box_density": n / (800.0 * 600.0),
        "has_strip": 1 if any(strip_flags) else 0,
        "n_strip": sum(strip_flags),
    }


def compute_iou_stats(det_boxes, gt_boxes):
    """Compute IoU statistics between detected and GT boxes."""
    if not det_boxes or not gt_boxes:
        return {"best_iou": 0.0, "avg_iou_matched": 0.0, "n_matched": 0}

    ious = []
    for db in det_boxes:
        best = 0.0
        for gb in gt_boxes:
            iou = iou_obb(db, gb)
            if iou > best:
                best = iou
        ious.append(best)

    matched = [i for i in ious if i >= 0.3]
    return {
        "best_iou": float(max(ious)),
        "avg_iou_matched": float(np.mean(matched)) if matched else 0.0,
        "n_matched": len(matched),
    }


def step1_copy_error_data():
    """Copy error images, GT labels, and detection output to hard directories."""
    print("=" * 60)
    print("STEP 1: Copy error data to hard directories")
    print("=" * 60)

    with open(BENCH_JSON, encoding="utf-8") as f:
        bench = json.load(f)

    error_images = [p for p in bench["per_image"] if p["fn"] > 0 or p["fp"] > 0]
    print(f"Error images: {len(error_images)} (FN={sum(p['fn'] for p in error_images)}, FP={sum(p['fp'] for p in error_images)})")

    for d in [HARD_IMG, HARD_LAB, HARD_DET]:
        os.makedirs(d, exist_ok=True)

    copied_img = copied_lab = copied_det = 0
    missing_img = missing_lab = missing_det = 0

    for p in error_images:
        iid = p["id"]

        src_img = os.path.join(IMG, iid + ".jpg")
        dst_img = os.path.join(HARD_IMG, iid + ".jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            copied_img += 1
        else:
            missing_img += 1

        src_lab = os.path.join(LAB, iid + ".json")
        dst_lab = os.path.join(HARD_LAB, iid + ".json")
        if os.path.exists(src_lab):
            shutil.copy2(src_lab, dst_lab)
            copied_lab += 1
        else:
            missing_lab += 1

        src_det = os.path.join(OUT_DIR, iid + ".json")
        dst_det = os.path.join(HARD_DET, iid + ".json")
        if os.path.exists(src_det):
            shutil.copy2(src_det, dst_det)
            copied_det += 1
        else:
            missing_det += 1

    print(f"  Images copied: {copied_img} (missing: {missing_img})")
    print(f"  Labels copied: {copied_lab} (missing: {missing_lab})")
    print(f"  Detections copied: {copied_det} (missing: {missing_det})")
    print(f"  -> {HARD_IMG}")
    print(f"  -> {HARD_LAB}")
    print(f"  -> {HARD_DET}")
    return error_images


def step2_build_panel(error_images):
    """Build per-image panel with rich features for DoubleML analysis."""
    print("\n" + "=" * 60)
    print("STEP 2: Build per-image panel with features")
    print("=" * 60)

    with open(BENCH_JSON, encoding="utf-8") as f:
        bench = json.load(f)

    fn_lookup = {}
    if os.path.exists(FN_DETAIL):
        with open(FN_DETAIL, encoding="utf-8") as f:
            fn_detail = json.load(f)
        fn_lookup = {item["image"]: item for item in fn_detail}

    exp = load_expected()

    all_images = bench["per_image"]
    print(f"Processing {len(all_images)} images (all, not just errors)...")

    rows = []
    for idx, p in enumerate(all_images):
        iid = p["id"]
        gt_boxes = load_gt_boxes(iid)
        det_boxes = load_det_boxes(iid)
        csv_count = exp.get(iid, 0)

        img_feat = extract_image_features(iid)
        if img_feat is None:
            continue

        gt_feat = extract_box_features(gt_boxes)
        det_feat = extract_box_features(det_boxes)
        iou_stats = compute_iou_stats(det_boxes, gt_boxes)

        csv_gt_gap = csv_count - len(gt_boxes)
        gt_incomplete = 1 if csv_count > len(gt_boxes) else 0
        has_fn = 1 if p["fn"] > 0 else 0
        has_fp = 1 if p["fp"] > 0 else 0
        has_error = 1 if (p["fn"] > 0 or p["fp"] > 0) else 0
        error_type = 0
        if p["fn"] > 0 and p["fp"] > 0:
            error_type = 3
        elif p["fn"] > 0:
            error_type = 1
        elif p["fp"] > 0:
            error_type = 2

        fn_iou = 0.0
        fn_det_gt_ratio = 0.0
        fn_red_density = 0.0
        if iid in fn_lookup:
            fd = fn_lookup[iid]
            fn_iou = fd.get("best_iou", 0.0)
            fn_det_gt_ratio = fd.get("det_gt_ratio", 0.0)
            fn_red_density = fd.get("red_density_around_gt", 0.0)

        row = {
            "image_id": iid,
            "tp": p["tp"],
            "fp": p["fp"],
            "fn": p["fn"],
            "gt_count": p["gt"],
            "det_count": p["det"],
            "csv_count": csv_count,
            "csv_gt_gap": csv_gt_gap,
            "gt_incomplete": gt_incomplete,
            "has_fn": has_fn,
            "has_fp": has_fp,
            "has_error": has_error,
            "error_type": error_type,
            "fn_iou": fn_iou,
            "fn_det_gt_ratio": fn_det_gt_ratio,
            "fn_red_density": fn_red_density,
        }
        for k, v in img_feat.items():
            row[k] = v
        for k, v in gt_feat.items():
            row[f"gt_{k}"] = v
        for k, v in det_feat.items():
            row[f"det_{k}"] = v
        for k, v in iou_stats.items():
            row[k] = v

        rows.append(row)

    columns = list(rows[0].keys())
    with open(PANEL_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    print(f"Panel saved: {PANEL_CSV} ({len(rows)} rows, {len(columns)} columns)")
    print(f"Columns: {', '.join(columns)}")
    return rows


def step3_run_doubleml(rows):
    """Run DoubleML causal analysis on the panel."""
    print("\n" + "=" * 60)
    print("STEP 3: DoubleML causal analysis")
    print("=" * 60)

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    import doubleml as dml

    df = pd.DataFrame(rows)

    covariate_cols = [
        "img_width", "img_height", "img_area",
        "red_pixel_count", "red_density", "n_red_components",
        "largest_comp_frac", "brightness", "contrast",
        "gt_n_boxes", "gt_avg_area", "gt_avg_aspect", "gt_avg_min_side",
        "gt_avg_max_side", "gt_std_aspect", "gt_has_strip", "gt_n_strip",
        "gt_box_density", "gt_spread_x", "gt_spread_y",
        "csv_gt_gap",
    ]
    covariate_cols = [c for c in covariate_cols if c in df.columns]

    models = [
        {
            "name": "M1_gt_incomplete_on_fp",
            "desc": "GT incompleteness -> FP count",
            "Y": "fp",
            "D": "gt_incomplete",
            "X": covariate_cols,
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M2_gt_incomplete_on_has_fp",
            "desc": "GT incompleteness -> has_any_FP",
            "Y": "has_fp",
            "D": "gt_incomplete",
            "X": covariate_cols,
            "ml_l": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "type": "IRM",
        },
        {
            "name": "M3_gt_avg_aspect_on_fn",
            "desc": "GT avg aspect ratio -> FN count",
            "Y": "fn",
            "D": "gt_avg_aspect",
            "X": [c for c in covariate_cols if c != "gt_avg_aspect"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M4_gt_has_strip_on_fn",
            "desc": "GT has strip-like box -> FN count",
            "Y": "fn",
            "D": "gt_has_strip",
            "X": [c for c in covariate_cols if c != "gt_has_strip" and c != "gt_n_strip"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M5_red_density_on_fn",
            "desc": "Red pixel density -> FN count",
            "Y": "fn",
            "D": "red_density",
            "X": [c for c in covariate_cols if c != "red_density" and c != "red_pixel_count"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M6_red_density_on_fp",
            "desc": "Red pixel density -> FP count",
            "Y": "fp",
            "D": "red_density",
            "X": [c for c in covariate_cols if c != "red_density" and c != "red_pixel_count"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M7_n_red_components_on_fp",
            "desc": "N red components -> FP count",
            "Y": "fp",
            "D": "n_red_components",
            "X": [c for c in covariate_cols if c != "n_red_components"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M8_gt_box_density_on_fn",
            "desc": "GT box density -> FN count",
            "Y": "fn",
            "D": "gt_box_density",
            "X": [c for c in covariate_cols if c != "gt_box_density" and c != "gt_n_boxes"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
        {
            "name": "M9_gt_incomplete_on_has_error",
            "desc": "GT incompleteness -> has_any_error",
            "Y": "has_error",
            "D": "gt_incomplete",
            "X": covariate_cols,
            "ml_l": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "type": "IRM",
        },
        {
            "name": "M10_gt_avg_min_side_on_fn",
            "desc": "GT avg min side -> FN count",
            "Y": "fn",
            "D": "gt_avg_min_side",
            "X": [c for c in covariate_cols if c != "gt_avg_min_side" and c != "gt_avg_max_side" and c != "gt_avg_area"],
            "ml_l": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "ml_m": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
            "type": "PLR",
        },
    ]

    results = []
    for m in models:
        print(f"\n  Running {m['name']}: {m['desc']}")
        try:
            Y = df[m["Y"]].values
            D = df[m["D"]].values
            X = df[m["X"]].values

            if m["type"] == "PLR":
                obj = dml.DoubleMLData.from_arrays(X, Y, D)
                dml_obj = dml.DoubleMLPLR(obj, m["ml_l"], m["ml_m"])
            elif m["type"] == "IRM":
                obj = dml.DoubleMLData.from_arrays(X, Y, D)
                dml_obj = dml.DoubleMLIRM(obj, m["ml_l"], m["ml_m"])

            dml_obj.fit()

            coef = float(dml_obj.coef[0])
            se = float(dml_obj.se[0])
            t_stat = float(dml_obj.t_stat[0])
            p_val = float(dml_obj.pval[0])
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

            print(f"    theta={coef:.4f} (se={se:.4f}, t={t_stat:.2f}, p={p_val:.4f})")
            print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

            results.append({
                "name": m["name"],
                "desc": m["desc"],
                "type": m["type"],
                "Y": m["Y"],
                "D": m["D"],
                "theta": coef,
                "se": se,
                "t_stat": t_stat,
                "p_val": p_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "name": m["name"],
                "desc": m["desc"],
                "type": m["type"],
                "Y": m["Y"],
                "D": m["D"],
                "theta": None,
                "se": None,
                "t_stat": None,
                "p_val": None,
                "ci_lower": None,
                "ci_upper": None,
                "error": str(e),
            })

    return results


def step4_feature_importance(rows):
    """RandomForest feature importance for FP and FN prediction."""
    print("\n" + "=" * 60)
    print("STEP 4: RandomForest feature importance")
    print("=" * 60)

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    df = pd.DataFrame(rows)

    feature_cols = [
        "img_width", "img_height", "img_area",
        "red_pixel_count", "red_density", "n_red_components",
        "largest_comp_frac", "brightness", "contrast",
        "gt_n_boxes", "gt_avg_area", "gt_avg_aspect", "gt_avg_min_side",
        "gt_avg_max_side", "gt_std_aspect", "gt_has_strip", "gt_n_strip",
        "gt_box_density", "gt_spread_x", "gt_spread_y",
        "csv_gt_gap", "gt_incomplete",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    results = {}
    for target in ["fp", "fn", "has_error"]:
        X = df[feature_cols].values
        y = df[target].values
        rf = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42)
        rf.fit(X, y)
        importances = dict(zip(feature_cols, rf.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
        results[target] = importances
        print(f"\n  Feature importance for {target}:")
        for feat, imp in list(importances.items())[:10]:
            print(f"    {feat:30s} {imp:.4f}")

    return results


def step5_write_report(error_images, dml_results, rf_results):
    """Write markdown report."""
    print("\n" + "=" * 60)
    print("STEP 5: Write report")
    print("=" * 60)

    lines = []
    lines.append(f"# DoubleML Error Cause Analysis ({HARD_DATE})")
    lines.append("")
    lines.append("## Error Summary")
    lines.append("")
    n_fn = sum(p["fn"] for p in error_images)
    n_fp = sum(p["fp"] for p in error_images)
    fn_only = sum(1 for p in error_images if p["fn"] > 0 and p["fp"] == 0)
    fp_only = sum(1 for p in error_images if p["fp"] > 0 and p["fn"] == 0)
    both = sum(1 for p in error_images if p["fn"] > 0 and p["fp"] > 0)
    lines.append(f"- Error images: {len(error_images)} / 500")
    lines.append(f"- FN-only images: {fn_only}, FP-only: {fp_only}, Both: {both}")
    lines.append(f"- Total FN boxes: {n_fn}, Total FP boxes: {n_fp}")
    lines.append(f"- Data copied to: `{HARD_IMG}`, `{HARD_LAB}`, `{HARD_DET}`")
    lines.append("")

    lines.append("## DoubleML Causal Models")
    lines.append("")
    lines.append("| Model | Y | D | Type | theta | SE | t-stat | p-value | 95% CI | Significant? |")
    lines.append("|-------|---|---|------|-------|-----|--------|---------|--------|-------------|")
    for r in dml_results:
        if r["theta"] is not None:
            sig = "**YES**" if r["p_val"] < 0.05 else "no"
            ci = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
            lines.append(
                f"| {r['name']} | {r['Y']} | {r['D']} | {r['type']} | "
                f"{r['theta']:.4f} | {r['se']:.4f} | {r['t_stat']:.2f} | "
                f"{r['p_val']:.4f} | {ci} | {sig} |"
            )
        else:
            lines.append(f"| {r['name']} | {r['Y']} | {r['D']} | {r['type']} | ERROR | - | - | - | - | - |")
    lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    for r in dml_results:
        if r["theta"] is not None:
            sig = "statistically significant" if r["p_val"] < 0.05 else "not significant"
            direction = "positive" if r["theta"] > 0 else "negative"
            lines.append(
                f"- **{r['name']}** ({r['desc']}): theta={r['theta']:.4f}, "
                f"p={r['p_val']:.4f} ({sig}). "
                f"Effect is {direction}: "
            )
            if r["theta"] > 0 and r["p_val"] < 0.05:
                lines.append(f"  increasing {r['D']} increases {r['Y']}.")
            elif r["theta"] < 0 and r["p_val"] < 0.05:
                lines.append(f"  increasing {r['D']} decreases {r['Y']}.")
            else:
                lines.append(f"  no clear causal effect.")
    lines.append("")

    lines.append("## RandomForest Feature Importance")
    lines.append("")
    for target, importances in rf_results.items():
        lines.append(f"### Predicting {target}")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("|---------|-----------|")
        for feat, imp in list(importances.items())[:15]:
            lines.append(f"| {feat} | {imp:.4f} |")
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    sig_models = [r for r in dml_results if r["theta"] is not None and r["p_val"] < 0.05]
    if sig_models:
        lines.append(f"Significant causal factors ({len(sig_models)}):")
        for r in sig_models:
            lines.append(f"- **{r['D']}** -> {r['Y']}: theta={r['theta']:.4f} (p={r['p_val']:.4f})")
    else:
        lines.append("No statistically significant causal factors found at p<0.05.")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {REPORT_MD}")


def main():
    error_images = step1_copy_error_data()
    rows = step2_build_panel(error_images)
    dml_results = step3_run_doubleml(rows)
    rf_results = step4_feature_importance(rows)
    step5_write_report(error_images, dml_results, rf_results)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()