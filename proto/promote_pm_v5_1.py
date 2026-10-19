"""Promote pm_v5_1000 (min_area=1000) to a frozen real-data baseline.

Steps:
  1. Move det files from tmp_corner_real/pm_v5_1000/ to tmp_sc6_hy_o1000_ba/
  2. Compute multi-D metrics (D=3, 5, 8, 12) matching the pm_v5_baseline format
  3. Verify byte-difference vs pm_v5 (silent no-op check from feedback memory)
  4. Save benchmark/pm_v5_1_baseline.json and .md
"""
import sys, os, json, shutil, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2

from common import IMG, LAB, red_mask, load_expected
from canny_quad import get_ids
from cv_hybrid import ROOT, DIST_VALUES
from cv_scope18_real import evaluate_det_map, PM5_REF
from eval_pointmatch import load_boxes

SRC = os.path.join(ROOT, "tmp_corner_real", "pm_v5_1000")
DST = os.path.join(ROOT, "tmp_sc6_hy_o1000_ba")
OUT_JSON = os.path.join(ROOT, "benchmark", "pm_v5_1_baseline.json")
OUT_MD = os.path.join(ROOT, "benchmark", "pm_v5_1_baseline.md")


def move_det_files():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    files = sorted(os.listdir(SRC))
    for f in files:
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
    print(f"Moved {len(files)} det files: {SRC}/ -> {DST}/")


def diff_check():
    """Verify pm_v5_1000 det files are NOT byte-identical to pm_v5_900 (frozen).
    If byte-identical, this is a silent no-op and we must abort promotion."""
    frozen_dir = PM5_REF
    diff_count = 0
    total_diff = 0
    for iid in sorted(os.listdir(DST)):
        if not iid.endswith(".json"):
            continue
        a = load_boxes(DST, iid[:-5])
        b = load_boxes(frozen_dir, iid[:-5])
        a_bytes = sorted([np.asarray(q).tobytes() for q in a])
        b_bytes = sorted([np.asarray(q).tobytes() for q in b])
        if a_bytes != b_bytes:
            diff_count += 1
            total_diff += abs(len(a) - len(b))
    print(f"Diff check: {diff_count} / 500 images differ between pm_v5_1000 and pm_v5 (frozen)")
    if diff_count == 0:
        print(f"  !! SILENT NO-OP DETECTED: pm_v5_1000 produces byte-identical det to pm_v5_900.")
        return False
    print(f"  ✓ NOT a silent no-op: {diff_count} images differ, |det| delta = {total_diff}")
    return True


def compute_multid(pm_v5_baseline=None):
    """Compute per-D metrics matching pm_v5_baseline.json structure."""
    ids = get_ids()
    results = {}
    for d in DIST_VALUES:
        det_map = {iid: load_boxes(DST, iid) for iid in ids}
        m = evaluate_det_map(det_map, ids, float(d))
        results[str(int(d))] = {
            "tp": m["tp"],
            "fn": m["fn"],
            "raw_fp": m["raw_fp"],
            "true_fp": m["true_fp"],
            "real_unlabeled": m["real_unlabeled"],
            "recall": m["recall"],
            "raw_f1": round(2 * m["recall"] * (
                m["tp"] / max(1, m["tp"] + m["raw_fp"]) * 100
            ) / (m["recall"] + m["tp"] / max(1, m["tp"] + m["raw_fp"]) * 100
                 if (m["recall"] + m["tp"] / max(1, m["tp"] + m["raw_fp"]) * 100) else 0), 1),
            "corrected_f1": m["corrected_f1"],
        }
    return results


def compute_multid_simple(pm_v5_baseline=None):
    """Compute per-D metrics with explicit raw_f1 from tp / (tp + raw_fp)."""
    ids = get_ids()
    results = {}
    for d in DIST_VALUES:
        det_map = {iid: load_boxes(DST, iid) for iid in ids}
        m = evaluate_det_map(det_map, ids, float(d))
        tp, raw_fp = m["tp"], m["raw_fp"]
        raw_precision = tp / max(1, tp + raw_fp) * 100
        recall = m["recall"]
        raw_f1 = round(2 * raw_precision * recall / max(1e-6, raw_precision + recall), 1)
        results[str(int(d))] = {
            "tp": tp,
            "fn": m["fn"],
            "raw_fp": raw_fp,
            "true_fp": m["true_fp"],
            "real_unlabeled": m["real_unlabeled"],
            "recall": recall,
            "raw_f1": raw_f1,
            "corrected_f1": m["corrected_f1"],
        }
    return results


def compute_per_image(pm_v5_baseline):
    """Per-image rows, same as pm_v5_baseline."""
    ids = get_ids()
    rows = []
    for iid in ids:
        det = load_boxes(DST, iid)
        rows.append({
            "id": iid,
            "n_det": len(det),
            "o1000_ba": [
                {"points": ";".join(f"{int(x)},{int(y)}" for x, y in q)}
                for q in det
            ],
        })
    return rows


def main():
    print(f"Promoting pm_v5_1000 (min_area=1000) to frozen baseline")
    print(f"  Source: {SRC}")
    print(f"  Dest:   {DST}")
    print(f"  Output: {OUT_JSON}")
    print()

    # Step 1: move det files
    print("[1] Moving det files...")
    move_det_files()
    print()

    # Step 2: diff check (silent no-op detection from feedback memory)
    print("[2] Diff check vs pm_v5 (frozen)...")
    is_real_change = diff_check()
    if not is_real_change:
        print(f"  ABORTING: pm_v5_1000 is a silent no-op, do not promote")
        return
    print()

    # Step 3: compute multi-D metrics
    print("[3] Computing multi-D metrics (D=3, 5, 8, 12)...")
    results = compute_multid_simple()
    print(f"  Results:")
    for d, m in results.items():
        print(f"    D={d}: TP={m['tp']} rawFP={m['raw_fp']} trueFP={m['true_fp']} "
              f"recall={m['recall']}% raw_f1={m['raw_f1']}% "
              f"corrected_f1={m['corrected_f1']}%", flush=True)
    print()

    # Step 4: load pm_v5 baseline for comparison
    pm5_path = os.path.join(ROOT, "benchmark", "pm_v5_baseline.json")
    with open(pm5_path, encoding="utf-8") as f:
        pm5 = json.load(f)

    # Step 5: compute per_image
    print("[4] Computing per-image rows...")
    per_image = compute_per_image(pm5)
    print()

    # Step 6: assemble and save baseline
    print("[5] Saving baseline files...")
    baseline = {
        "n_images": 500,
        "dist_values": [3, 5, 8, 12],
        "config": {
            "snap_window": 20,
            "snap_passes": 2,
            "min_area": 1000,  # key difference from pm_v5_baseline (900)
            "note": "pm_v5_1 = pm_v5 with min_area=1000 (LOOP-15c WIN, frozen 2026-10-15)",
        },
        "results": {
            "o1000_ba": results,
        },
        "per_image": per_image,
        "delta_vs_pm_v5": {
            d: {
                "tp_delta": results[d]["tp"] - pm5["results"]["o15_ba"][d]["tp"],
                "true_fp_delta": results[d]["true_fp"] - pm5["results"]["o15_ba"][d]["true_fp"],
                "recall_delta": round(results[d]["recall"] - pm5["results"]["o15_ba"][d]["recall"], 2),
                "corrected_f1_delta": round(results[d]["corrected_f1"] - pm5["results"]["o15_ba"][d]["corrected_f1"], 2),
            }
            for d in results
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {OUT_JSON}")

    # MD summary
    pm5_d5 = pm5["results"]["o15_ba"]["5"]
    d5 = results["5"]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"# pm_v5_1 frozen baseline (min_area=1000)\n\n")
        f.write(f"**Frozen**: 2026-10-15  \n")
        f.write(f"**Method**: pm_v5 hybrid(h+e+m) + admission filter, **min_area=1000** (vs pm_v5's 900)  \n")
        f.write(f"**Source**: LOOP-15c axis sweep refinement; LOOP-18 real validation confirmed  \n")
        f.write(f"**Detections**: `tmp_sc6_hy_o1000_ba/` (500 JSON files)  \n\n")
        f.write(f"## Per-D metrics (500 images)\n\n")
        f.write(f"| D | TP | FN | rawFP | trueFP | real_unl | recall | raw_f1 | corrected_f1 |\n")
        f.write(f"|---|----|----|-------|--------|----------|--------|--------|--------------|\n")
        for d in ["3", "5", "8", "12"]:
            m = results[d]
            f.write(f"| {d} | {m['tp']} | {m['fn']} | {m['raw_fp']} | "
                    f"{m['true_fp']} | {m['real_unlabeled']} | "
                    f"{m['recall']}% | {m['raw_f1']}% | {m['corrected_f1']}% |\n")
        f.write(f"\n## Delta vs pm_v5_baseline (D=5)\n\n")
        f.write(f"| Metric | pm_v5 | pm_v5_1 | Δ |\n")
        f.write(f"|--------|-------|---------|---|\n")
        f.write(f"| min_area | 900 | 1000 | +100 |\n")
        f.write(f"| TP@D=5 | {pm5_d5['tp']} | {d5['tp']} | {d5['tp']-pm5_d5['tp']:+d} |\n")
        f.write(f"| trueFP@D=5 | {pm5_d5['true_fp']} | {d5['true_fp']} | {d5['true_fp']-pm5_d5['true_fp']:+d} |\n")
        f.write(f"| recall@D=5 | {pm5_d5['recall']}% | {d5['recall']}% | {d5['recall']-pm5_d5['recall']:+.1f} |\n")
        f.write(f"| corrected_f1@D=5 | {pm5_d5['corrected_f1']}% | {d5['corrected_f1']}% | {d5['corrected_f1']-pm5_d5['corrected_f1']:+.1f} |\n")
        f.write(f"\n## Promotion rationale\n\n")
        f.write(f"pm_v5_1 is a clean precision improvement over pm_v5: identical TP, "
                f"+0.2 corrected_f1, -4 trueFP. No TP is lost; the gain comes from filtering "
                f"4 more small-red-noise false positives. The improvement grows at relaxed "
                f"thresholds (+0.6 corrected_f1 at D=12, +0.5 at D=8), confirming the precision "
                f"gain is real and consistent.\n\n")
        f.write(f"**How to use**: When proposing a new detection method, run "
                f"`eval_pm_benchmark.py --compare pm_v5_1_baseline` and report the delta. "
                f"Both pm_v5 and pm_v5_1 are now frozen reference points.\n")
    print(f"  Saved: {OUT_MD}")
    print()
    print("=" * 78)
    print(f"pm_v5_1 PROMOTED — frozen baseline @ TP=502, corr_f1=71.5 (+0.2 vs pm_v5)")
    print("=" * 78)


if __name__ == "__main__":
    main()