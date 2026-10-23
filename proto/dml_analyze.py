"""LOOP-81: DoubleML analysis of the factorial panel (proto/dml_panel.py).

The panel is a complete block design over 4 detector knobs (block = image):
every image is run through all 16 cells of a 2^4 factorial.  This script

  1. estimates the CAUSAL effect of every knob on the per-image outcome
     (TP / true-FP count at D=3/5) with DoubleML PLR (partially linear
     regression, random forest nuisance) and cluster robust standard errors
     (the cluster is the image: the 16 rows of one image are not independent),
  2. estimates the INTERACTIONS, because the D=5 TP gain of pm_v10 is not a
     main effect at all: it only appears when hole_fit_pass AND hole_refine
     are both on.  A product of two binary treatments cannot be a DoubleML
     treatment (the m-learner prediction is then not binary), so the
     interactions are estimated by STRATIFIED DML runs (the effect of a
     inside the stratum b=0 and inside b=1) and cross-checked against the
     exact design based interaction contrast,
  3. cross-checks every DoubleML estimate against the exact design based
     estimate of the same factorial (the paired within-image contrast),
  4. checks the additivity: the sum of the four main effects must reproduce
     the global pm_v9 -> pm_v10 config change, which is what a config knob
     actually buys in production.

A single treatment run additionally produces a CATE (conditional average
treatment effect) over the image covariates - i.e. WHICH images benefit from
a knob - via DoubleML's best linear predictor.

Usage:
    python proto/dml_analyze.py --panel benchmark/dml_panel_real.json \
        --out benchmark/dml_real.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dml_panel import FACTORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTCOMES = ["tp5", "fp5", "tp3", "fp3"]
TCOLS = [f[0] for f in FACTORS]


# ----------------------------------------------------------------- panel io

def load_panel(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["rows"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    covs = [c for c in df.columns
            if c not in TCOLS and not c.startswith(("tp", "fp"))
            and c not in ("iid", "cell", "ndet", "unl5")]
    covs = [c for c in covs if df[c].std() > 0]
    return data, df, covs


# ----------------------------------------------------- design based effects

def paired_effect(df, y, tcol):
    """Exact design based estimate: the paired within-image contrast."""
    d = []
    for _, g in df.groupby("iid"):
        a = g.loc[g[tcol] == 1, y]
        b = g.loc[g[tcol] == 0, y]
        if len(a) and len(b):
            d.append(float(a.mean() - b.mean()))
    d = np.asarray(d, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(d.mean()), se, n


def paired_interaction(df, y, a, b):
    """Exact design based interaction contrast (2x2 within-image)."""
    d = []
    for _, g in df.groupby("iid"):
        m = {}
        for va in (0, 1):
            for vb in (0, 1):
                s = g.loc[(g[a] == va) & (g[b] == vb), y]
                m[(va, vb)] = float(s.mean()) if len(s) else np.nan
        if any(np.isnan(v) for v in m.values()):
            continue
        d.append((m[(1, 1)] - m[(1, 0)]) - (m[(0, 1)] - m[(0, 0)]))
    d = np.asarray(d, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(d.mean()), se, n


def paired_conditional(df, y, a, b, bval):
    """Effect of `a` inside the stratum b == bval (exact, within image)."""
    sub = df[df[b] == bval]
    d = []
    for _, g in sub.groupby("iid"):
        x = g.loc[g[a] == 1, y]
        z = g.loc[g[a] == 0, y]
        if len(x) and len(z):
            d.append(float(x.mean() - z.mean()))
    d = np.asarray(d, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(d.mean()), se, n


def paired_joint_conditional(df, y, eff, f1, f2):
    """Effect of `eff` inside each joint stratum of (f1, f2).

    The marginal conditional effects hide the interaction: the interesting
    question is what happens when TWO other knobs are on at the same time
    (e.g. the cost of a stricter red threshold once the geometry is fixed).
    """
    out = {}
    for v1 in (0, 1):
        for v2 in (0, 1):
            sub = df[(df[f1] == v1) & (df[f2] == v2)]
            d = []
            for _, g in sub.groupby("iid"):
                x = g.loc[g[eff] == 1, y]
                z = g.loc[g[eff] == 0, y]
                if len(x) and len(z):
                    d.append(float(x.mean() - z.mean()))
            d = np.asarray(d, float)
            n = len(d)
            se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            out["%s|%s=%d,%s=%d" % (eff, f1, v1, f2, v2)] = {
                "theta": float(d.mean()) if n else 0.0, "se": se, "n": n}
    return out


def blocked_ols(df, y, tcols):
    """OLS of y on the treatment dummies with IMAGE fixed effects.

    This is the textbook estimate of the factorial main effects of the
    complete block design; it is the reference the DoubleML estimate has to
    agree with.
    """
    import statsmodels.api as sm
    sub = df[["iid", y] + tcols].copy()
    dummies = pd.get_dummies(sub["iid"], drop_first=True).astype(float)
    X = pd.concat([sub[tcols].astype(float).reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    yv = sub[y].astype(float).reset_index(drop=True)
    res = sm.OLS(yv, X).fit(cov_type="cluster",
                             cov_kwds={"groups": sub["iid"].reset_index(drop=True)})
    out = {}
    for c in tcols:
        out[c] = (float(res.params[c]), float(res.bse[c]), float(res.pvalues[c]))
    return out


# ----------------------------------------------------------------- doubleml

def _ml(kind="reg"):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, max_depth=8,
                                  min_samples_leaf=5, random_state=42,
                                  n_jobs=-1)


def dml_plr(df, y, dcols, xcols, n_rep=2, n_folds=5):
    """DoubleML PLR with cluster robust (image) standard errors."""
    from doubleml import DoubleMLData, DoubleMLPLR
    cols = ["iid", y] + list(dcols) + list(xcols)
    sub = df[cols].copy()
    dml_data = DoubleMLData(sub, y_col=y, d_cols=list(dcols),
                             x_cols=list(xcols), cluster_cols=["iid"])
    model = DoubleMLPLR(dml_data, _ml(), _ml(), n_folds=n_folds, n_rep=n_rep)
    model.fit()
    coef, se, p, ci = model.coef, model.se, model.pval, model.confint()
    out = {}
    for i, c in enumerate(dcols):
        out[c] = {
            "theta": float(np.ravel(coef)[i]),
            "se": float(np.ravel(se)[i]),
            "ci_low": float(np.asarray(ci)[i][0]),
            "ci_high": float(np.asarray(ci)[i][1]),
            "p": float(np.ravel(p)[i]),
        }
    return out, model, sub


def dml_cate(model, sub, covs):
    """CATE of a single treatment run over the image covariates."""
    basis = pd.DataFrame(index=sub.index)
    for c in covs:
        s = sub[c].astype(float)
        if s.std() > 0:
            basis[c] = (s - s.mean()) / s.std()
    try:
        m = model.cate(basis, cov_type="cluster",
                       cov_kwds={"groups": sub["iid"].to_numpy()})
        coef = np.ravel(m.coef)
        se = np.ravel(m.se)
        return {c: {"theta": float(coef[i]), "se": float(se[i])}
                for i, c in enumerate(basis.columns)}
    except Exception as exc:                       # pragma: no cover
        return {"_error": str(exc)[:200]}


# ----------------------------------------------------------------- report

def fmt(res, n_img):
    return "  |  ".join(
        "%s %+0.4f [%+0.4f,%+0.4f] x%d=%+0.1f p=%.3f" %
        (c, r["theta"], r["ci_low"], r["ci_high"], n_img, r["theta"] * n_img,
         r["p"]) for c, r in res.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-rep", type=int, default=2)
    ap.add_argument("--cate-outcomes", default="tp5,fp5")
    ap.add_argument("--outcomes", default=",".join(OUTCOMES))
    args = ap.parse_args()
    outcomes = [o for o in args.outcomes.split(",") if o]

    data, df, covs = load_panel(args.panel)
    n_img = data["n_images"]
    print("panel: %s  %d images, %d rows, %d covariates" %
          (args.panel, n_img, len(df), len(covs)), flush=True)
    print("covariates: %s" % ", ".join(covs), flush=True)

    results = {"panel": args.panel, "n_images": n_img, "n_rows": len(df),
               "covariates": covs, "outcomes": {}}

    # ---------------------------------------------------------- design table
    print("\n" + "=" * 104)
    print("DESIGN: the 16 cells (aggregate over the %d images)" % n_img)
    print("=" * 104)
    summ = data["cell_summary"]
    base = summ["ms8-fp0-rf0-sup50"]          # pm_v9
    win = summ["ms14-fp2-rf1-sup50"]           # pm_v10
    print("%-24s %6s %6s %6s %6s %6s %6s   %s" %
          ("cell", "ndet", "tp3", "fp3", "tp5", "fp5", "fp8", "delta vs pm_v9"))
    for name, a in summ.items():
        mark = "  <== pm_v9" if name == "ms8-fp0-rf0-sup50" else (
            "  <== pm_v10" if name == "ms14-fp2-rf1-sup50" else "")
        print("%-24s %6d %6d %6d %6d %6d %6d   tp%+4d fp%+4d%s" %
              (name, a["ndet"], a["tp3"], a["fp3"], a["tp5"], a["fp5"], a["fp8"],
               a["tp5"] - base["tp5"], a["fp5"] - base["fp5"], mark))
    results["cell_summary"] = summ
    results["global_contrast_pm9_to_pm10"] = {
        k: win[k] - base[k] for k in
        ("ndet", "tp3", "fp3", "tp5", "fp5", "tp8", "fp8", "tp12", "fp12")}

    # ---------------------------------------------------------- doubleml
    for y in outcomes:
        t_start = time.time()
        print("\n" + "=" * 104)
        print("OUTCOME %s  (per-image count; x%d = the effect on the whole "
              "%d image benchmark)" % (y, n_img, n_img))
        print("=" * 104)
        res = {"single": {}, "multi": {}, "stratified": {}, "paired": {},
               "paired_interaction": {}, "paired_conditional": {},
               "paired_joint_conditional": {}, "blocked_ols": {}}

        # 1) design based main effects and interactions (exact, no model)
        for t in TCOLS:
            th, se, n = paired_effect(df, y, t)
            res["paired"][t] = {"theta": th, "se": se, "n": n}
        for i, a in enumerate(TCOLS):
            for b in TCOLS[i + 1:]:
                th, se, n = paired_interaction(df, y, a, b)
                res["paired_interaction"]["%s:%s" % (a, b)] = {
                    "theta": th, "se": se, "n": n}
        print("  design (paired within image) main effects:")
        for t in TCOLS:
            r = res["paired"][t]
            print("    %-12s %+0.4f  se=%0.4f  x%d = %+0.1f" %
                  (t, r["theta"], r["se"], n_img, r["theta"] * n_img), flush=True)
        print("  design interactions (x%d):" % n_img)
        for k, r in sorted(res["paired_interaction"].items(),
                           key=lambda kv: -abs(kv[1]["theta"])):
            print("    %-26s %+0.4f  se=%0.4f  x%d = %+0.1f" %
                  (k, r["theta"], r["se"], n_img, r["theta"] * n_img), flush=True)
        # conditional effects: the effect of a inside the stratum b=0 / b=1
        # (both directions: "a inside b" and "b inside a")
        print("  design conditional effects (effect of A inside B=0 / B=1):")
        for i, a in enumerate(TCOLS):
            for b in TCOLS[i + 1:]:
                for (eff, stratum) in ((a, b), (b, a)):
                    line = []
                    for bv in (0, 1):
                        th, se, n = paired_conditional(df, y, eff, stratum, bv)
                        res["paired_conditional"]["%s|%s=%d" % (eff, stratum, bv)] = {
                            "theta": th, "se": se, "n": n}
                        line.append("B=%d: %+0.4f(se %0.4f, x%d %+0.1f)" %
                                    (bv, th, se, n_img, th * n_img))
                    print("    %-12s inside %-12s  %s" % (eff, stratum, "   ".join(line)),
                          flush=True)

        # joint strata: the effect of one knob when TWO others are on at once
        print("  design joint-conditional effects (effect of A | B,C):")
        for eff in TCOLS:
            others = [c for c in TCOLS if c != eff]
            for i, f1 in enumerate(others):
                for f2 in others[i + 1:]:
                    jc = paired_joint_conditional(df, y, eff, f1, f2)
                    res["paired_joint_conditional"].update(jc)
                    parts = ["%s=%d,%s=%d: %+0.4f(x%d %+0.1f)" % (
                        f1, int(k.split(",")[0].split("=")[1]),
                        f2, int(k.split(",")[1].split("=")[1]),
                        v["theta"], n_img, v["theta"] * n_img)
                        for k, v in jc.items()]
                    print("    %-12s | %s" % (eff, "  ".join(parts)), flush=True)

        # 2) single treatment DML (the other knobs are controls)
        for t in TCOLS:
            x = list(covs) + [c for c in TCOLS if c != t]
            try:
                r, _, _ = dml_plr(df, y, [t], x, n_rep=args.n_rep,
                                  n_folds=args.n_folds)
            except Exception as exc:
                print("  single %-12s FAILED: %s" % (t, str(exc)[:160]), flush=True)
                continue
            res["single"][t] = r[t]
            print("  single  %-12s %s" % (t, fmt(r, n_img)), flush=True)

        # 3) multi treatment DML (all four knobs jointly)
        try:
            r, _, _ = dml_plr(df, y, TCOLS, list(covs), n_rep=args.n_rep,
                              n_folds=args.n_folds)
            res["multi"] = r
            print("  multi   %s" % fmt(r, n_img), flush=True)
        except Exception as exc:
            print("  multi   FAILED: %s" % str(exc)[:160], flush=True)

        # 4) stratified DML for the two strongest interactions
        top = sorted(res["paired_interaction"].items(),
                     key=lambda kv: -abs(kv[1]["theta"]))[:2]
        for key, _ in top:
            a, b = key.split(":")
            for strat in (0, 1):
                sub = df[df[b] == strat]
                x = list(covs) + [c for c in TCOLS if c not in (a, b)]
                try:
                    r, _, _ = dml_plr(sub, y, [a], x, n_rep=args.n_rep,
                                      n_folds=args.n_folds)
                except Exception as exc:
                    print("  strat %s|%s=%d FAILED: %s" %
                          (key, b, strat, str(exc)[:120]), flush=True)
                    continue
                res["stratified"]["%s|%s=%d" % (key, b, strat)] = r[a]
                print("  strat   effect of %s inside %s=%d : %s" %
                      (a, b, strat, fmt(r, n_img)), flush=True)

        # 5) blocked OLS reference (image fixed effects)
        try:
            bo = blocked_ols(df, y, TCOLS)
            res["blocked_ols"] = bo
            print("  blocked OLS (image FE): " +
                  "  ".join("%s %+0.4f(p=%.3f)" % (c, bo[c][0], bo[c][2])
                            for c in TCOLS), flush=True)
        except Exception as exc:
            print("  blocked OLS FAILED: %s" % str(exc)[:160], flush=True)

        # 6) additivity: sum of the main effects vs the global config change
        s = sum(res["single"][t]["theta"] for t in res["single"])
        g = results["global_contrast_pm9_to_pm10"].get(y, 0) / float(n_img)
        res["additivity_check"] = {"sum_main_effects": s,
                                   "global_contrast_per_image": g,
                                   "difference": s - g}
        print("  additivity: sum of the 4 DML main effects = %+0.5f/img "
              "(x%d = %+0.1f)  vs the global pm_v9->pm_v10 change %+0.5f/img "
              "(x%d = %+d)" % (s, n_img, s * n_img, g, n_img,
                              results["global_contrast_pm9_to_pm10"].get(y, 0)),
              flush=True)
        res["seconds"] = round(time.time() - t_start, 1)
        results["outcomes"][y] = res

    # ---------------------------------------------------------- CATE
    print("\n" + "=" * 104)
    print("CATE: which IMAGES benefit from a knob (single treatment run)")
    print("=" * 104)
    cate_out = {}
    for y in args.cate_outcomes.split(","):
        for t in TCOLS:
            x = list(covs) + [c for c in TCOLS if c != t]
            try:
                r, model, sub = dml_plr(df, y, [t], x, n_rep=args.n_rep,
                                        n_folds=args.n_folds)
            except Exception as exc:
                print("  %s/%s FAILED: %s" % (y, t, str(exc)[:120]), flush=True)
                continue
            c_ = dml_cate(model, sub, covs)
            if "_error" in c_:
                print("  %s/%s CATE failed: %s" % (y, t, c_["_error"]), flush=True)
                continue
            order = sorted(c_, key=lambda c: -abs(c_[c]["theta"]))
            cate_out["%s|%s" % (y, t)] = {"ate": r[t], "modifiers": c_}
            print("  %-5s %-12s ATE=%+0.4f | top effect modifiers: %s" %
                  (y, t, r[t]["theta"],
                   ", ".join("%s %+0.4f" % (c, c_[c]["theta"]) for c in order[:4])),
                  flush=True)
    results["cate"] = cate_out

    out = args.out or os.path.join(ROOT, "benchmark",
                                   "dml_%s.json" % data["dataset"])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()