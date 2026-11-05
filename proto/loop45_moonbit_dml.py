"""LOOP-45: independent re-estimation of the repo's causal claims with the
pure-MoonBit DoubleML port (`riantr/moonbit_doubleML`).

The driver reads the *existing* frozen panels (no row dropping, no re-binning),
generates a tiny MoonBit executable that embeds the panel as arrays, runs the
port's DoubleMLPLR, and compares sign + p<0.05 status against the Python
DoubleML results recorded in `benchmark/or2_route_verdicts.md` and
`benchmark/cv_scope24.json`.

Acceptance (pre-registered in the LOOP-45 goal): a claim is REPLICATED only
when the sign matches AND the p<0.05 status matches. Theta/CI deltas are
reported but do not decide, because the MoonBit port's default nuisance is a
linear learner while the Python panel used RandomForest nuisances.

Usage: python proto/loop45_moonbit_dml.py
"""
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(ROOT, "benchmark", "or2_panels")
STAGE_PANEL = os.path.join(ROOT, "benchmark", "cv_scope24_panel.json")
OUT = os.path.join(ROOT, "benchmark", "loop45_moonbit_dml.json")
VERDICT = os.path.join(ROOT, "benchmark", "loop45_moonbit_dml_verdict.md")
MBT = os.path.join(ROOT, "tmp_loop45_mbt")
MOON = "moon"
FOLD_SEED = 42
N_FOLDS = 3

X6 = ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio", "x_n_holes", "x_gt_min_side_mean"]


def _read_csv(name):
    import pandas as pd
    return pd.read_csv(os.path.join(PANEL_DIR, name + ".csv"))


def build_cases():
    """Each case: (claim_id, panel, treatment, outcome, x_cols, orig_theta, orig_p)."""
    cases = []
    # P1-a: each enforcement mode vs none, conditioning on the other mode.
    cases += [
        ("P1-a:edge", "p1_a", "t_edge", "y_tp5", X6 + ["t_blend"], +0.0020, 0.718),
        ("P1-a:blend", "p1_a", "t_blend", "y_tp5", X6 + ["t_edge"], -0.0005, 0.901),
        ("P1-b:tol", "p1_b", "t_tol", "y_tp5", X6, +0.0013, 0.051),
        ("P1-c:band", "p1_c", "t_band", "y_tp5", X6, -0.0080, 5.41e-04),
        ("P1-d:joint", "p1_d", "t_joint", "y_tp5", X6, -0.0055, 0.125),
        # P2-a: box level, overlap degree vs detection@D5.
        ("P2-a:any", "p2a_overlap", "t_overlap_any", "y_det5",
         ["x_min_side", "x_area", "x_aspect", "x_edge_dist", "x_n_gt_image"], -0.1818, 3.23e-09),
        ("P2-a:bin", "p2a_overlap", "t_overlap_bin", "y_det5",
         ["x_min_side", "x_area", "x_aspect", "x_edge_dist", "x_n_gt_image"], -0.0807, 3.11e-08),
        # P2-b: 3-arm mask panel; each arm is conditioned on the other.
        ("P2-b:oracle", "p2b_mask", "t_mask_oracle", "y_tp5",
         ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio", "t_mask_wide"], +0.2244, 4.55e-09),
        ("P2-b:wide", "p2b_mask", "t_mask_wide", "y_tp5",
         ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio", "t_mask_oracle"], +0.0077, 0.429),
        # P2-c: corpus scale (300 vs 2000), two outcomes.
        ("P2-c:y5", "p2c_scale", "t_scale", "y_tp5",
         ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio"], +0.5348, 3.16e-53),
        ("P2-c:y12", "p2c_scale", "t_scale", "y_tp12",
         ["x_n_gt", "x_img_w", "x_img_h", "x_red_ratio"], +0.3778, 9.76e-31),
        # P2-d: hard cardinality cap; ex_soft is a non-treatment covariate.
        ("P2-d:cap", "p2d_cardinality", "t_cap", "y_tp5",
         ["x_n_gt", "x_csv_n", "x_red_ratio", "ex_soft"], -0.0535, 3.75e-04),
        # P2-e: box-level agreement.
        ("P2-e:agree", "p2e_agreement", "t_agree", "y_is_tp5",
         ["x_sup", "x_n_det_image", "x_det_min_side", "x_det_area"], +0.0517, 5.48e-03),
    ]
    # Stage panel (500 images, 6 stages, outcome = image TP).
    stage_claims = [
        ("stage1:red_mask", "s1_red_ratio", -1.571387, 0.384835),
        ("stage2:denoise", "s2_kept_area_ratio", -1.792326, 0.004583),
        ("stage3:outer_contours", "s3_n_contours", +0.494134, 0.090562),
        ("stage4:base_quads", "s4_n_total_candidates", +0.282581, 0.17454),
        ("stage5:admission", "s5_pass_rate", +1.598792, 0.09235),
        ("stage6:dedup", "s6_dedup_rate", +1.094931, 0.370967),
    ]
    for cid, treat, theta, p in stage_claims:
        cases.append((cid, "stage", treat, "tp",
                      [c for c in ("n_gt", "s1_red_ratio", "s2_kept_area_ratio",
                                   "s3_n_contours", "s4_n_total_candidates",
                                   "s5_pass_rate", "s6_dedup_rate") if c != treat],
                      theta, p))
    return cases


def load_case(panel, treatment, outcome, x_cols):
    import pandas as pd
    import numpy as np
    if panel == "stage":
        rows = json.load(open(STAGE_PANEL, encoding="utf-8"))
        df = pd.DataFrame(rows)
        df["block"] = df["iid"]
    else:
        df = _read_csv(panel)
    cols = [treatment, outcome] + list(x_cols) + ["block"]
    for c in cols:
        if c not in df.columns:
            raise KeyError("panel %s missing column %s" % (panel, c))
    sub = df[cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    # deterministic cluster ids in sorted block order
    blocks = sorted(sub["block"].astype(str).unique())
    bmap = {b: i for i, b in enumerate(blocks)}
    sub["_cl"] = sub["block"].astype(str).map(bmap)
    return sub, treatment, outcome, list(x_cols)


def _fmt(v):
    if math.isnan(v) or math.isinf(v):
        raise ValueError("non-finite value in panel")
    return repr(float(v))


def write_mbt(sub, treatment, outcome, x_cols):
    os.makedirs(os.path.join(MBT, "cmd", "panel"), exist_ok=True)
    with open(os.path.join(MBT, "cmd", "panel", "moon.pkg"), "w", encoding="utf-8") as f:
        f.write('import {\n  "riantr/moonbit_doubleML",\n}\n\npkgtype(kind: "executable")\n')
    n = len(sub)
    x_rows = ",\n    ".join(
        "[" + ", ".join(_fmt(r[c]) for c in x_cols) + "]" for _, r in sub.iterrows()
    )
    y = ", ".join(_fmt(v) for v in sub[outcome].tolist())
    d = ", ".join(_fmt(v) for v in sub[treatment].tolist())
    cl = ", ".join(str(int(v)) for v in sub["_cl"].tolist())
    src = f"""fn main {{
  let x = @moonbit_doubleML.Matrix::from_rows([
    {x_rows}
  ])
  let y : Array[Double] = [{y}]
  let d : Array[Double] = [{d}]
  let cl : Array[Int32] = [{cl}]
  let data = @moonbit_doubleML.DoubleMLData::new(x, y, d, cluster_vars=cl)
  let m = @moonbit_doubleML.DoubleMLPLR::new(data, n_folds={N_FOLDS}, n_rep=1, seed={FOLD_SEED}).fit()
  let c = m.coef()
  let s = m.se()
  let (lo, hi) = m.confint()
  let z = if s > 0.0 {{ c / s }} else {{ 0.0 }}
  let p = 2.0 * (1.0 - @moonbit_doubleML.norm_cdf(z.abs()))
  println("RESULT c=" + c.to_string() + " se=" + s.to_string() + " lo=" + lo.to_string() + " hi=" + hi.to_string() + " p=" + p.to_string())
}}
"""
    with open(os.path.join(MBT, "cmd", "panel", "main.mbt"), "w", encoding="utf-8") as f:
        f.write(src)
    return n, len(x_cols)


def run_mbt():
    r = subprocess.run([MOON, "run", "--target", "native", "cmd/panel"],
                       cwd=MBT, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("moon run failed:\nSTDOUT:\n%s\nSTDERR:\n%s"
                           % (r.stdout[-4000:], r.stderr[-4000:]))
    for line in reversed(r.stdout.splitlines()):
        if line.startswith("RESULT "):
            kv = dict(p.split("=", 1) for p in line[len("RESULT "):].split())
            return {k: float(v) for k, v in kv.items()}, r.stdout
    raise RuntimeError("no RESULT line:\n" + r.stdout[-4000:])


def main():
    import pandas as pd
    cases = build_cases()
    print("LOOP-45: %d claims, MoonBit DoubleML PLR, n_folds=%d seed=%d"
          % (len(cases), N_FOLDS, FOLD_SEED), flush=True)
    results = []
    t0 = time.time()
    for cid, panel, treat, outcome, x_cols, otheta, op in cases:
        sub, _, _, _ = load_case(panel, treat, outcome, x_cols)
        n, pcols = write_mbt(sub, treat, outcome, x_cols)
        res, raw = run_mbt()
        sign_ok = (res["c"] > 0) == (otheta > 0)
        sig_ok = (res["p"] < 0.05) == (op < 0.05)
        rep = bool(sign_ok and sig_ok)
        results.append({
            "claim": cid, "panel": panel, "treatment": treat, "outcome": outcome,
            "n_units": int(n), "n_blocks": int(sub["_cl"].nunique()),
            "n_covariates": int(pcols), "learner": "linear_regression",
            "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
            "theta": round(res["c"], 6), "se": round(res["se"], 6),
            "ci95": [round(res["lo"], 6), round(res["hi"], 6)],
            "p": res["p"],
            "original_theta": otheta, "original_p": op,
            "sign_match": bool(sign_ok), "significance_match": bool(sig_ok),
            "replicated": rep,
        })
        print("  %-22s n=%4d b=%3d  theta %+0.4f (orig %+0.4f)  p=%.3g (orig %.3g)  %s"
              % (cid, n, sub["_cl"].nunique(), res["c"], otheta, res["p"], op,
                 "REPLICATED" if rep else "NOT-REPLICATED"), flush=True)
    summary = {
        "n_claims": len(results),
        "replicated": sum(1 for r in results if r["replicated"]),
        "sign_match": sum(1 for r in results if r["sign_match"]),
        "significance_match": sum(1 for r in results if r["significance_match"]),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out = {
        "loop": "LOOP-45",
        "tool": "riantr/moonbit_doubleML@0.164.62",
        "moon": "0.1.20261029",
        "anchor": {"ngt": 1076, "tp": {"3": 857, "5": 901, "8": 926, "12": 938},
                   "detections": 1070,
                   "source": "benchmark/pm_v10_refine.json"},
        "acceptance": "sign match AND p<0.05 status match",
        "summary": summary,
        "claims": results,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # verdict markdown
    lines = [
        "# LOOP-45 — MoonBit DoubleML replication of the causal claims",
        "",
        "Tool: `riantr/moonbit_doubleML@0.164.62` (moon 0.1.20261029). "
        "Panels reused unchanged; PLR with linear learners, n_folds=3, seed=42, "
        "cluster-robust on the image block. Acceptance: **sign + p<0.05 status**.",
        "",
        "Frozen anchor: 1076 rule-consistent GT, 1070 detections, D=3/5/8/12 = 857/901/926/938.",
        "",
        "| claim | n | blocks | theta (MoonBit) | theta (Python) | p (MoonBit) | p (Python) | sign | sig<.05 | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append("| %s | %d | %d | %+0.4f | %+0.4f | %.3g | %.3g | %s | %s | **%s** |"
                     % (r["claim"], r["n_units"], r["n_blocks"], r["theta"],
                        r["original_theta"], r["p"], r["original_p"],
                        "OK" if r["sign_match"] else "MISMATCH",
                        "OK" if r["significance_match"] else "MISMATCH",
                        "REPLICATED" if r["replicated"] else "NOT-REPLICATED"))
    s = summary
    lines += [
        "",
        "## Summary",
        "",
        "- claims: **%d**, replicated: **%d**, sign matches: **%d**, significance matches: **%d**."
        % (s["n_claims"], s["replicated"], s["sign_match"], s["significance_match"]),
        "- wall clock: %.1f s." % s["wall_clock_s"],
        "",
        "A NOT-REPLICATED claim is not automatically refuted: the MoonBit port uses "
        "linear nuisance learners while the Python panels used RandomForest PLR, so "
        "the point estimates and p-values can differ. The sign/significance test is "
        "the pre-registered robustness check.",
        "",
    ]
    open(VERDICT, "w", encoding="utf-8").write("\n".join(lines))
    print("\nSUMMARY: %d/%d replicated; sign %d/%d; significance %d/%d; %.1f s"
          % (s["replicated"], s["n_claims"], s["sign_match"], s["n_claims"],
             s["significance_match"], s["n_claims"], s["wall_clock_s"]), flush=True)
    print("saved %s and %s" % (OUT, VERDICT))


if __name__ == "__main__":
    main()