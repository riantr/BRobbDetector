"""LOOP-78: C# / Python parity + benchmark for the new edge rule.

Runs the Python production pipeline (proto/scope45_holes.py) over the 500
benchmark images with the 2026-10-21 edge rule, writes the detections in the
CLI JSON format and compares them box by box with the C# output produced by

    dotnet run -- --cli tmp_csharp_in tmp_csharp_cvhybrid --method cvhybrid

Use --run-csharp to produce that output automatically (the C# CLI takes a
DIRECTORY, and the annotated image directory holds 56832 images, so only the
500 benchmark images are staged into a temporary directory first).

Then scores both against
  * the raw GT (1121 boxes) and
  * the GT filtered with the same rule (the rectangles that touch the image
    edge are not valid targets either -> 1076 boxes).
"""
import os, sys, json, argparse, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

from common import IMG, LAB, red_mask
from eval_pointmatch import load_boxes, match_boxes_point, is_real_unlabeled
from scope45_holes import P, HOLE_DEFAULT, PIPE_DEFAULT, detect, load_all
from scope45_holes import mask_cache
from quad_ortho import quad_on_boundary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_OUT = "tmp_py_edge"
CS_OUT = "tmp_csharp_cvhybrid"
STAGE_DIR = "tmp_bench_imgs"


def stage_images(ids):
    """Copy the benchmark images into a small directory for the C# CLI."""
    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    os.makedirs(STAGE_DIR)
    for iid in ids:
        shutil.copy2(os.path.join(IMG, iid + ".jpg"),
                     os.path.join(STAGE_DIR, iid + ".jpg"))


def run_csharp(ids):
    stage_images(ids)
    if os.path.isdir(CS_OUT):
        shutil.rmtree(CS_OUT)
    dll = os.path.join(ROOT, "BRobbDetector", "bin", "Debug", "net10.0-windows",
                       "BRobbDetector.dll")
    if os.path.exists(dll):
        cmd = ["dotnet", dll, "--cli", STAGE_DIR, CS_OUT, "--method", "cvhybrid"]
    else:
        cmd = ["dotnet", "run", "--project",
               os.path.join(ROOT, "BRobbDetector", "BRobbDetector.csproj"),
               "-c", "Debug", "--", "--cli", STAGE_DIR, CS_OUT, "--method", "cvhybrid"]
    print("running C# detector ...")
    subprocess.run(cmd, check=True, cwd=ROOT)
    shutil.rmtree(STAGE_DIR, ignore_errors=True)


def write_json(path, iid, quads):
    infos = [{"points": ";".join(f"{int(p[0])},{int(p[1])}" for p in q), "type": ""}
             for q in quads]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"img_name": iid, "infos": infos}, f, ensure_ascii=False)


def run_python(ids, outdir):
    os.makedirs(outdir, exist_ok=True)
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        quads = []
        if img is not None:
            quads = [np.asarray(q, np.int32) for q in
                     detect(img, P, HOLE_DEFAULT, PIPE_DEFAULT)]
        write_json(os.path.join(outdir, iid + ".json"), iid, quads)


def compare(ids):
    mismatch = 0
    npy = ncs = 0
    for iid in ids:
        a = load_boxes(PY_OUT, iid)
        b = load_boxes(CS_OUT, iid)
        npy += len(a)
        ncs += len(b)
        if len(a) != len(b) or any((x != y).any() for x, y in zip(a, b)):
            mismatch += 1
            if mismatch <= 5:
                print(f"  MISMATCH {iid}: py={len(a)} cs={len(b)}")
    return mismatch, npy, ncs


def score(ids, detdir, gt_map, exp):
    tot = {D: {"tp": 0, "raw_fp": 0, "true_fp": 0, "real_unl": 0} for D in (3, 5, 8, 12)}
    for iid in ids:
        det = [np.asarray(d, np.int32) for d in load_boxes(detdir, iid)]
        gt = gt_map[iid]
        for D in tot:
            tp, unmatched = match_boxes_point(det, gt, float(D))
            tot[D]["tp"] += tp
            tot[D]["raw_fp"] += len(unmatched)
            for di in unmatched:
                if is_real_unlabeled(det[di], exp.get(iid, 0), len(gt),
                                     mask_cache.get(iid)):
                    tot[D]["real_unl"] += 1
                else:
                    tot[D]["true_fp"] += 1
    return tot


def summarize(tag, ids, gt_map, detdir, exp):
    tot = score(ids, detdir, gt_map, exp)
    total_gt = sum(len(v) for v in gt_map.values())
    res = {}
    for D, t in tot.items():
        rec = t["tp"] / max(total_gt, 1) * 100
        prec = t["tp"] / max(t["tp"] + t["true_fp"], 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        res[str(D)] = {**t, "recall": round(rec, 1), "precision": round(prec, 1),
                       "corrected_f1": round(f1, 1)}
    r5 = res["5"]
    print(f"{tag:34s} ngt={total_gt:4d} D=5 TP={r5['tp']:4d} rec={r5['recall']:5.1f} "
          f"prec={r5['precision']:5.1f} trueFP={r5['true_fp']:4d} unl={r5['real_unl']:3d} "
          f"corr_f1={r5['corrected_f1']:5.1f}")
    return {"ngt": total_gt, "res": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/pm_v9_edge_rule.json")
    ap.add_argument("--skip-py", action="store_true")
    ap.add_argument("--run-csharp", action="store_true",
                    help="run the C# detector on the 500 benchmark images first")
    args = ap.parse_args()

    ids, gt_map, exp = load_all()
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            continue
        mask_cache[iid] = cv2.dilate(red_mask(img, 80, 110, 10),
                                      np.ones((3, 3), np.uint8), iterations=3) > 0

    if args.run_csharp:
        run_csharp(ids)

    if not args.skip_py:
        run_python(ids, PY_OUT)
        print("python detection done")

    mism, npy, ncs = compare(ids)
    print(f"parity: py={npy} cs={ncs} mismatched images={mism}")

    # GT filtered with the same rule
    gt_filtered = {}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG, iid + ".jpg"))
        if img is None:
            gt_filtered[iid] = gt_map[iid]
            continue
        H, W = img.shape[:2]
        gt_filtered[iid] = [g for g in gt_map[iid]
                            if not quad_on_boundary(g, W, H, 0.0)]

    out = {"parity": {"py": npy, "cs": ncs, "mismatched_images": mism}}
    out["cs_rawGT"] = summarize("C#  (raw GT)", ids, gt_map, CS_OUT, exp)
    out["cs_consistentGT"] = summarize("C#  (consistent GT)", ids, gt_filtered, CS_OUT, exp)
    out["py_rawGT"] = summarize("py  (raw GT)", ids, gt_map, PY_OUT, exp)
    out["py_consistentGT"] = summarize("py  (consistent GT)", ids, gt_filtered, PY_OUT, exp)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()