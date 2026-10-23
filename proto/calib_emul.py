"""Calibrate the emulation stroke model against the real annotation.

The emulation is only a useful proxy for the real CARS data if the *same*
things break the detector.  This script generates a small set with a given
stroke model, measures it and compares it with the real reference:

  metric                        real          emulation
  recall D=3 / D=5              78.5 / 83.6   ...
  miss by min side coverage     83/61/36/9/2/0
  red band width p50            4.0 px        ...
  TP corner distance p50        1.41 px       ...

Usage:
  python proto/calib_emul.py --n 300 --set stroke_min=2 --set stroke_max=4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = r"D:\src\CodeArts\BRobbDetector"
PY = sys.executable

# The real reference (benchmark/pm_v9_edge_rule.json, benchmark/real_degrade_full.json,
# proto/measure_real_line.py).
REAL = {
    "rec3": 78.5, "rec5": 83.6, "rec8": 85.8, "rec12": 87.0,
    "miss": [83.3, 60.7, 35.5, 9.2, 1.5, 0.3],
    "band_p50": 4.0, "band_p90": 7.5,
    "corner_p50": 1.41, "corner_p90": 2.24,
    "cov_mean": 0.866,
}


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def gen(args, sets, tag):
    img = os.path.join(ROOT, "data", "Image@calib_%s" % tag)
    lab = os.path.join(ROOT, "data", "Label@calib_%s" % tag)
    shutil.rmtree(img, ignore_errors=True)
    shutil.rmtree(lab, ignore_errors=True)
    cmd = [PY, "proto/gen_emulation.py", "--n", str(args.n), "--seed", str(args.seed),
           "--size-pool", "benchmark/real_gt_size_pool.json",
           "--recipe-pool", "benchmark/real_gt_recipe.json",
           "--stroke-model", "wobbly", "--overshoot", "3", "--min-gap", "8",
           "--out-img", img, "--out-lab", lab,
           "--manifest", "benchmark/calib_%s_manifest.json" % tag]
    for kv in sets:
        k, v = kv.split("=", 1)
        cmd += ["--%s" % k.replace("_", "-"), v]
    r = run(cmd)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("generation failed")
    return img, lab


def measure(img, lab, workers, out_json):
    """Coverage + band width of the generated set."""
    r = run([PY, "proto/qc_wobbly.py", "--img-dir", img, "--lab-dir", lab,
             "--workers", str(workers), "--out", out_json])
    return {"qc": r.stdout}


def broken_side_table(qc_json, gt_json):
    """Recall per number of broken sides (the real reference: a box with one
    broken side is found in 43% of the cases, with two in 22%)."""
    import collections
    qc = json.load(open(qc_json, encoding="utf-8"))
    gt = json.load(open(gt_json, encoding="utf-8"))["gt"]
    key = {(r["stem"], r["idx"]): r for r in qc}
    agg = collections.defaultdict(lambda: [0, 0])
    for r in gt:
        q = key.get((r["stem"], r["idx"]))
        if q is None:
            continue
        nmiss = sum(1 for c in q["cover"] if c < 0.999)
        a = agg[nmiss]
        a[0] += 1
        a[1] += 1 if r["kind"] == "tp" else 0
    return {k: (v[0], v[1], 100.0 * v[1] / max(v[0], 1)) for k, v in sorted(agg.items())}


def benchmark(img, lab, workers, tag):
    """The point-match benchmark of the red view."""
    r = run([PY, "proto/eval_emulation.py", "--img-dir", img, "--lab-dir", lab,
             "--colors", "red", "--no-allmask", "--workers", str(workers),
             "--analyse", "red",
             "--save-gt-records", "calib_%s_gt" % tag])
    out = {"log": r.stdout}
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-2000:])
        raise SystemExit("benchmark failed")
    return out


def parse(log):
    """Pull the numbers out of the benchmark log."""
    import re
    res = {}
    # the D=3/5/8/12 table line of the red view:
    #   red  206/314/38.4/39.6/39.0  453/67/84.4/87.1/85.7 ...
    for line in log.splitlines():
        s = line.strip()
        if not s.startswith("red ") or "/" not in s:
            continue
        nums = re.findall(r"(\d+)\s*/\s*(\d+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", s)
        if len(nums) >= 4:
            for D, g in zip((3, 5, 8, 12), nums[:4]):
                res["tp%d" % D] = int(g[0])
                res["fp%d" % D] = int(g[1])
                res["rec%d" % D] = float(g[2])
                res["prec%d" % D] = float(g[3])
                res["f1%d" % D] = float(g[4])
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("[") and "boxes" in s and "miss" in s:
            lo = float(s[1:s.index(",")])
            res.setdefault("miss", {})[lo] = float(s.split()[-1].rstrip("%"))
    for line in log.splitlines():
        if line.strip().startswith("red ") and "ndet" in line:
            p = line.replace("=", " ").split()
            res["ngt"] = int(p[p.index("TP") + 2])
            res["ndet"] = int(p[p.index("ndet") + 1])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--set", action="append", default=[],
                    help="generator option, e.g. stroke_min=2")
    args = ap.parse_args()

    t0 = time.time()
    img, lab = gen(args, args.set, args.tag)
    qc_json = os.path.join(ROOT, "benchmark", "calib_%s_qc.json" % args.tag)
    gt_json = os.path.join(ROOT, "benchmark", "calib_%s_gt.json" % args.tag)
    qc = measure(img, lab, args.workers, qc_json)
    bm = benchmark(img, lab, args.workers, args.tag)
    res = parse(bm["log"])

    print("\n=== %s   %s" % (args.tag, " ".join(args.set)))
    for line in qc["qc"].splitlines():
        if ("band width" in line or "real annotation" in line
                or "mean side coverage" in line or "broken sides" in line):
            print("  " + line.strip())
    for D in (3, 5, 8, 12):
        if "rec%d" % D in res:
            print("  D=%-2d  recall %5.1f (real %5.1f)   precision %5.1f"
                  % (D, res["rec%d" % D], REAL["rec%d" % D], res.get("prec%d" % D, 0)))
    print("  min side coverage -> miss")
    for lo in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
        e = res.get("miss", {}).get(lo)
        r = REAL["miss"][[0.0, 0.1, 0.3, 0.5, 0.7, 0.9].index(lo)]
        if e is not None:
            print("    [%.1f,..)  emul %5.1f%%   real %5.1f%%" % (lo, e, r))
    try:
        tab = broken_side_table(qc_json, gt_json)
        real = {0: 96.9, 1: 42.9, 2: 22.0, 3: 7.1, 4: 0.0}
        print("  broken sides -> recall")
        for k, (n, tp, pct) in tab.items():
            print("    %d broken: %5d boxes  recall %5.1f%%   (real %5.1f%%)"
                  % (k, n, pct, real.get(k, float("nan"))))
    except Exception as e:                       # pragma: no cover
        print("  (broken side table failed: %s)" % e)
    print("  (%.1fs)" % (time.time() - t0))
    with open(os.path.join(ROOT, "benchmark", "calib_%s.json" % args.tag), "w",
              encoding="utf-8") as f:
        json.dump({"sets": args.set, "res": res}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())