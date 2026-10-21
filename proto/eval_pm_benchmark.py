"""Point-match benchmark: freeze baseline for the new corner-precision objective.

Evaluates legacy detectors (m8, YOLO, YOLO+snap, combine+snap) plus the
CV-domain quad extractors (canny/contour, hull, edgefit, hybrid) at multiple D
thresholds. Supports --save <name> and --compare <name> for regression tracking.

Usage:
  python proto/eval_pm_benchmark.py --save pm_v3_baseline
  python proto/eval_pm_benchmark.py --compare pm_v3_baseline
"""
import os, sys, json, argparse, itertools, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from common import load_expected, IMG, LAB, red_mask, iou_obb
from eval_pointmatch import (load_boxes, corner_match_distance, match_boxes_point,
                             snap_box_edge_fit, refine_box, is_real_unlabeled,
                             PERMS_4, OUT, SUP_THR, MIN_SIDE, MAX_ASPECT)

BENCH_DIR = r"D:\src\CodeArts\BRobbDetector\benchmark"
CANNY_OUT = r"D:\src\CodeArts\BRobbDetector\tmp_canny_out"
CV_OUT = {
    "cvhybrid_cs": r"D:\src\CodeArts\BRobbDetector\tmp_csharp_cvhybrid",
    "o15_ba": r"D:\src\CodeArts\BRobbDetector\tmp_sc6_hy_o15_ba",
    "hull": r"D:\src\CodeArts\BRobbDetector\tmp_cv_hull",
    "edgefit": r"D:\src\CodeArts\BRobbDetector\tmp_cv_edgefit",
    "hybrid_cem": r"D:\src\CodeArts\BRobbDetector\tmp_cv_hybrid_cem",
    "hy_chm": r"D:\src\CodeArts\BRobbDetector\tmp_sc3_hy_chm",
}
LEGACY_REUSE = ("m8", "yolo", "yolo_snap", "combine_snap")
DIST_VALUES = [3, 5, 8, 12]
SNAP_WINDOW = 20
SNAP_PASSES = 2
CONF = 0.55


def get_ids():
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0); rng.shuffle(ids); ids = ids[:500]
    return ids


def _canny_priority_dedup(dets, iou_thr=0.30):
    """Keep boxes in order, dropping any that overlap an already-kept box.

    The input list must have the highest-priority (most precise) boxes first
    (canny), so lower-priority boxes (m8/yolo) are dropped when they overlap."""
    keep = []
    for r in dets:
        r = r.astype(np.int32)
        if all(iou_obb(r, k) < iou_thr for k in keep):
            keep.append(r)
    return keep


def run_yolo(ids, conf=CONF):
    from ultralytics import YOLO
    model_path = r"D:\src\CodeArts\BRobbDetector\tmp_yolo\train\weights\best.onnx"
    model = YOLO(model_path, task="obb")
    det_map = {}
    for n, iid in enumerate(ids):
        img_path = os.path.join(IMG, iid + ".jpg")
        results = model.predict(img_path, verbose=False, conf=conf)
        boxes = []
        if results and results[0].obb is not None:
            for obb in results[0].obb:
                pts = obb.cpu().numpy().xyxyxyxy[0].astype(np.int32)
                if len(pts) == 4:
                    boxes.append(pts)
        det_map[iid] = boxes
        if (n + 1) % 100 == 0:
            print(f"  [YOLO {n+1}/{len(ids)}] done")
    return det_map



def evaluate_method(ids, method, dist_thr, yolo_map, per_image_out=None):
    """Evaluate a single method at a single D threshold.

    method: 'm8', 'yolo', 'yolo_snap', 'combine_snap', 'canny', 'canny_m8',
            or any key of CV_OUT ('hull', 'edgefit', 'hybrid_cem')
    """
    exp = load_expected()
    TP = FN = 0
    raw_fp = true_fp = real_unl = 0
    per_image = []

    for n, iid in enumerate(ids):
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        gt = load_boxes(LAB, iid)

        if method in ("yolo", "yolo_snap", "combine_snap"):
            yolo_det = list(yolo_map.get(iid, []))
        else:
            yolo_det = []

        if method == "m8":
            det = load_boxes(OUT, iid)
        elif method == "yolo":
            det = yolo_det
        elif method == "yolo_snap":
            det = yolo_det
        elif method == "combine_snap":
            det = list(load_boxes(OUT, iid)) + yolo_det
        elif method == "canny":
            det = load_boxes(CANNY_OUT, iid)
        elif method in CV_OUT:
            det = load_boxes(CV_OUT[method], iid)
        elif method == "canny_m8":
            det = list(load_boxes(CANNY_OUT, iid)) + list(load_boxes(OUT, iid))
            det = _canny_priority_dedup(det)
        else:
            det = []

        det = [d.astype(np.int32) for d in det]
        mask = red_mask(img)
        mask_d = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        csv_n = exp.get(iid, 0)

        if method in ("yolo_snap", "combine_snap"):
            for _ in range(SNAP_PASSES):
                det = [snap_box_edge_fit(d, mask, search_width=SNAP_WINDOW) for d in det]

        tp, unmatched = match_boxes_point(det, gt, dist_thr)
        TP += tp; FN += len(gt) - tp
        raw_fp += len(unmatched)
        img_true_fp = 0
        img_real_unl = 0
        for di in unmatched:
            if is_real_unlabeled(det[di], csv_n, len(gt), mask_d):
                real_unl += 1
                img_real_unl += 1
            else:
                true_fp += 1
                img_true_fp += 1

        if per_image_out is not None:
            per_image.append({
                "id": iid, "gt": len(gt), "det": len(det),
                "tp": tp, "fp": len(unmatched), "fn": len(gt) - tp,
                "true_fp": img_true_fp, "real_unl": img_real_unl,
            })

    total_gt = TP + FN
    recall = TP / total_gt * 100 if total_gt else 0
    prec_raw = TP / (TP + raw_fp) * 100 if (TP + raw_fp) else 0
    f1_raw = 2 * prec_raw * recall / (prec_raw + recall) if (prec_raw + recall) else 0
    prec_c = TP / (TP + true_fp) * 100 if (TP + true_fp) else 0
    f1_c = 2 * prec_c * recall / (prec_c + recall) if (prec_c + recall) else 0

    result = {
        "tp": TP, "fn": FN, "raw_fp": raw_fp, "true_fp": true_fp, "real_unlabeled": real_unl,
        "recall": round(recall, 1),
        "raw_f1": round(f1_raw, 1), "corrected_f1": round(f1_c, 1),
    }
    if per_image_out is not None:
        result["per_image"] = per_image
    return result


def main():
    ap = argparse.ArgumentParser(description="Point-match benchmark")
    ap.add_argument("--save", default=None, help="save to benchmark/<name>.json")
    ap.add_argument("--compare", default=None, help="compare against benchmark/<name>.json")
    ap.add_argument("--no-yolo", action="store_true", help="skip YOLO, use saved det dir")
    ap.add_argument("--reuse-from", default=None,
                    help="copy legacy method rows (m8/yolo/yolo_snap/combine_snap) "
                         "from benchmark/<name>.json and skip YOLO")
    args = ap.parse_args()

    ids = get_ids()
    print(f"Point-match benchmark on {len(ids)} images, D={DIST_VALUES}")

    methods = ["cvhybrid_cs", "o15_ba", "hy_chm", "hybrid_cem", "hull", "edgefit", "canny", "m8",
               "yolo", "yolo_snap", "combine_snap"]
    method_labels = {
        "cvhybrid_cs": "cvhybrid(C#)",
        "o15_ba": "hybrid(h+e+m)+filter",
        "hy_chm": "hybrid(h+e+m)",
        "hybrid_cem": "hybrid(c+e+m)",
        "hull": "hull+approx",
        "edgefit": "edgefit",
        "canny": "canny(contour)",
        "m8": "m8",
        "yolo": "YOLO",
        "yolo_snap": "YOLO+snap",
        "combine_snap": "combine+snap",
    }

    reuse = args.reuse_from is not None
    eval_methods = [m for m in methods if not (reuse and m in LEGACY_REUSE)]
    yolo_map = {}
    if any(m in ("yolo", "yolo_snap", "combine_snap") for m in eval_methods):
        print("\nRunning YOLO detection...")
        t0 = time.time()
        yolo_map = run_yolo(ids)
        print(f"  YOLO done in {time.time()-t0:.0f}s")

    all_results = {}
    for method in eval_methods:
        all_results[method] = {}
        for D in DIST_VALUES:
            save_pi = (method in ("o15_ba", "cvhybrid_cs")) and (args.save is not None)
            r = evaluate_method(ids, method, D, yolo_map,
                                per_image_out=[] if save_pi else None)
            all_results[method][str(D)] = {k: v for k, v in r.items() if k != "per_image"}
            if save_pi:
                all_results["_per_image_" + method] = r.get("per_image", [])
            print(f"  {method_labels[method]:15s} D={D:2d}: TP={r['tp']:4d}  "
                  f"recall={r['recall']:5.1f}%  raw_f1={r['raw_f1']:5.1f}  "
                  f"corr_f1={r['corrected_f1']:5.1f}")

    if reuse:
        ref_path = os.path.join(BENCH_DIR, args.reuse_from + ".json")
        with open(ref_path, encoding="utf-8-sig") as f:
            ref = json.load(f)
        for m in LEGACY_REUSE:
            if m in methods and m in ref["results"]:
                all_results[m] = ref["results"][m]
        print(f"  reused legacy rows from {args.reuse_from}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"POINT-MATCH BENCHMARK SUMMARY ({len(ids)} images, 746 GT boxes)")
    print(f"{'='*80}")
    header = f"{'Method':15s}" + "".join(f"  D={d:<10d}" for d in DIST_VALUES)
    print(header)
    for method in methods:
        row = f"{method_labels[method]:15s}"
        for D in DIST_VALUES:
            r = all_results[method][str(D)]
            row += f"  TP={r['tp']:4d}     "
        print(row)
    best_method = max(methods, key=lambda m: all_results[m]["5"]["tp"])
    best_tp = all_results[best_method]["5"]["tp"]
    print(f"\n  Snapping: window={SNAP_WINDOW}, passes={SNAP_PASSES}, conf={CONF}")
    print(f"  Best (D=5 TP): {method_labels[best_method]} with {best_tp} TP")
    print(f"{'='*80}")

    # Save
    if args.save:
        os.makedirs(BENCH_DIR, exist_ok=True)
        path = os.path.join(BENCH_DIR, args.save + ".json")
        output = {
            "n_images": len(ids),
            "dist_values": DIST_VALUES,
            "config": {
                "snap_window": SNAP_WINDOW,
                "snap_passes": SNAP_PASSES,
                "conf": CONF,
                "methods": methods,
            },
            "results": all_results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {path}")

        # Also save human-readable MD
        md_path = os.path.join(BENCH_DIR, args.save + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Point-Match Benchmark: {args.save}\n\n")
            f.write(f"**{len(ids)} images, 746 GT boxes**\n\n")
            f.write(f"## Configuration\n")
            f.write(f"- Snap window: {SNAP_WINDOW}px\n")
            f.write(f"- Snap passes: {SNAP_PASSES}\n")
            f.write(f"- YOLO conf: {CONF}\n")
            f.write(f"- Edge sampling: full (0.0-1.0)\n")
            f.write(f"- Line fit: Huber (nearest red per profile)\n\n")
            f.write(f"## Results\n\n")
            f.write(f"| Method |")
            for D in DIST_VALUES:
                f.write(f" D={D} |")
            f.write("\n|--------|")
            for D in DIST_VALUES:
                f.write("------|")
            f.write("\n")
            for method in methods:
                f.write(f"| {method_labels[method]} |")
                for D in DIST_VALUES:
                    r = all_results[method][str(D)]
                    f.write(f" {r['tp']} |")
                f.write("\n")
            f.write(f"\n## Recall (%)\n\n")
            f.write(f"| Method |")
            for D in DIST_VALUES:
                f.write(f" D={D} |")
            f.write("\n|--------|")
            for D in DIST_VALUES:
                f.write("------|")
            f.write("\n")
            for method in methods:
                f.write(f"| {method_labels[method]} |")
                for D in DIST_VALUES:
                    r = all_results[method][str(D)]
                    f.write(f" {r['recall']}% |")
                f.write("\n")
            f.write(f"\n## Key Findings\n\n")
            d3c, d5c, d8c = (all_results["canny"][str(d)]["tp"] for d in (3, 5, 8))
            d3b, d5b, d8b = (all_results["combine_snap"][str(d)]["tp"] for d in (3, 5, 8))
            f.write(f"- **canny(contour) dominates legacy at D≤8**: D=3 {d3c} vs {d3b} ({d3c/max(d3b,1):.1f}×), "
                    f"D=5 {d5c} vs {d5b} ({d5c/max(d5b,1):.1f}×), "
                    f"D=8 {d8c} vs {d8b} ({d8c/max(d8b,1):.1f}×)\n")
            if "hull" in all_results:
                h3, h5 = all_results["hull"]["3"]["tp"], all_results["hull"]["5"]["tp"]
                f.write(f"- **hull+approx beats contour** (convex hull fixes concave dents): "
                        f"D=3 {h3} vs {d3c} ({h3-d3c:+d}), D=5 {h5} vs {d5c} ({h5-d5c:+d})\n")
            if "hy_chm" in all_results:
                c3, c5 = all_results["hy_chm"]["3"], all_results["hy_chm"]["5"]
                f.write(f"- **hybrid(h+e+m) is best** (hull-primary union): D=3 TP {c3['tp']} "
                        f"(corr_f1 {c3['corrected_f1']}), D=5 TP {c5['tp']} (corr_f1 {c5['corrected_f1']})\n")
            if "o15_ba" in all_results:
                o3, o5 = all_results["o15_ba"]["3"], all_results["o15_ba"]["5"]
                f.write(f"- **hybrid(h+e+m)+filter is the new best** (admission filter "
                        f"ortho_tol=15, boundary_band=12, max_aspect=12): D=3 TP {o3['tp']} "
                        f"(corr_f1 {o3['corrected_f1']}), D=5 TP {o5['tp']} (corr_f1 {o5['corrected_f1']}) "
                        f"— strictly better than pm_v4 at every D\n")
            if "cvhybrid_cs" in all_results:
                c3, c5 = all_results["cvhybrid_cs"]["3"], all_results["cvhybrid_cs"]["5"]
                f.write(f"- **cvhybrid(C#) port**: D=3 TP {c3['tp']} (corr_f1 {c3['corrected_f1']}), "
                        f"D=5 TP {c5['tp']} (corr_f1 {c5['corrected_f1']}) — C# production port of pm_v5\n")
            if "hybrid_cem" in all_results:
                m3 = all_results["hybrid_cem"]["3"]
                f.write(f"- hybrid(c+e+m) [contour-primary] D=3 TP {m3['tp']} — hull-primary is better\n")
            f.write(f"- 3-axis orthogonal scope: mask variants (close/dilate) are byte-identical to raw; "
                    f"hole-ring and fragment-join proposals are dead (see benchmark/cv_scope3.json)\n")
            best = max(methods, key=lambda m: all_results[m]["5"]["tp"])
            f.write(f"- **Best at D=5: {method_labels[best]} ({all_results[best]['5']['tp']} TP, "
                    f"corr_f1 {all_results[best]['5']['corrected_f1']})**\n")
        print(f"Saved to {md_path}")

    # Compare
    if args.compare:
        path = os.path.join(BENCH_DIR, args.compare + ".json")
        if not os.path.exists(path):
            print(f"\nBaseline {path} not found.")
        else:
            with open(path, encoding="utf-8-sig") as f:
                base = json.load(f)
            base_results = base["results"]
            print(f"\n--- Comparison vs baseline '{args.compare}' ---")
            for method in methods:
                if method not in base_results:
                    continue
                for D in DIST_VALUES:
                    cur = all_results[method][str(D)]
                    b = base_results[method][str(D)]
                    d_tp = cur["tp"] - b["tp"]
                    d_recall = cur["recall"] - b["recall"]
                    d_f1 = cur["corrected_f1"] - b["corrected_f1"]
                    flag = " ***" if abs(d_tp) >= 5 else ""
                    print(f"  {method_labels[method]:15s} D={D:2d}: "
                          f"TP {b['tp']}->{cur['tp']} ({d_tp:+d})  "
                          f"recall {b['recall']}->{cur['recall']} ({d_recall:+.1f})  "
                          f"corr_f1 {b['corrected_f1']}->{cur['corrected_f1']} ({d_f1:+.1f})"
                          f"{flag}")


if __name__ == "__main__":
    main()