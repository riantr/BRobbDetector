"""LOOP-49 held-out completion: run the chosen dense-cluster split lever on
seeds 1 and 2, update `benchmark/loop49_levers.json`, and regenerate the
verdict. Seed 0 is already DEAD (901 -> 901 TP, theta 0, p=0.724); the held-out
runs are recorded to satisfy Done-when #4, not to rescue the lever.

Usage: python proto/loop49_heldout.py
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop49_levers as L

ROOT = L.MB.ROOT
OUT_JSON = os.path.join(ROOT, "benchmark", "loop49_levers.json")
OUT_MD = os.path.join(ROOT, "benchmark", "loop49_levers_verdict.md")


def main():
    d = json.load(open(OUT_JSON, encoding="utf-8"))
    for seed in (1, 2):
        if d["seeds"].get(str(seed), {}).get("status") == "DONE":
            continue
        t0 = time.time()
        metric, rows, agg, ids = L.run_seed(seed)
        est = L.estimate(rows)
        pd.DataFrame(rows).to_csv(
            os.path.join(L.OUT_PANEL, "seed%d_split.csv" % seed), index=False)
        s, l = agg["shipped"], agg["split"]
        base = metric["shipped"][5]
        d_tp = l["5"]["tp"] - s["5"]["tp"]
        d_fp = l["5"]["true_fp"] - s["5"]["true_fp"]
        print("  seed %d base D5 %d TP / %d rawFP | split %d TP / %d rawFP "
              "| theta %+0.4f p=%.3g"
              % (seed, base["tp"], base["fp"], metric["split"][5]["tp"],
                 metric["split"][5]["fp"], est["theta"], est["p"]), flush=True)
        d["seeds"][str(seed)] = {
            "status": "DONE", "n_images": len(ids),
            "base_d5_tp": s["5"]["tp"], "base_d5_rawFP": base["fp"],
            "base_d5_dets": s["5"]["tp"] + base["fp"],
            "lever_d5_tp": l["5"]["tp"],
            "lever_d5_rawFP": metric["split"][5]["fp"],
            "delta_tp": d_tp, "delta_trueFP": d_fp,
            "trueFP_per_gain": (None if d_tp <= 0 else round(d_fp / d_tp, 3)),
            "estimate": est,
            "metrics": {a: metric[a] for a in ("shipped", "split")},
            "corrected": {a: {str(D): {"tp": agg[a][str(D)]["tp"],
                                       "true_fp": agg[a][str(D)]["true_fp"],
                                       "corr_f1": agg[a][str(D)]["corr_f1"]}
                              for D in (3, 5, 8, 12)} for a in agg},
            "wall_clock_s": round(time.time() - t0, 1),
        }
    s0 = d["seeds"]["0"]
    d["verdict"] = "DEAD" if not s0.get("seed0_pass") else (
        "CONFIRMED" if all(d["seeds"].get(x, {}).get("status") == "DONE"
                           and d["seeds"][x]["estimate"]["theta"] > 0
                           and d["seeds"][x]["estimate"]["p"] < 0.05
                           for x in ("1", "2")) else "NOT-REPLICATED")
    d["heldout_note"] = ("Seed 0 failed the pre-registered acceptance (0 TP gain, "
                         "theta 0, p=0.724). Seeds 1/2 were still measured to satisfy "
                         "Done-when #4; the overall verdict remains DEAD.")
    json.dump(d, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # regenerate the verdict markdown
    c = json.load(open(os.path.join(ROOT, "benchmark",
                                    "loop49_overlap_census.json"), encoding="utf-8"))
    Ln = ["# LOOP-49 — dense-cluster second-box proposal", "",
          "## Phase 0 — overlapping-FN census (%d rows)" % c["n_overlap_fn"], "",
          "| bucket | count |", "|---|---|"]
    for k, v in c["buckets"].items():
        Ln.append("| %s | %d |" % (k, v))
    Ln += ["", "Second box proposed: **D5 = %d, D12 = %d**; no-proposal fraction **%.1f%%** "
           "(stop line 80%%)." % (c["second_box"]["proposed_d5"],
                                 c["second_box"]["proposed_d12"],
                                 100 * c["second_box"]["no_proposal_fraction"]),
           "", "Stop object:", "```json", json.dumps(c["stop"], indent=1), "```", "",
           "## Phase 1 — the only eligible lever (concave-corner split)", "",
           "| seed | status | theta | p | D5 TP shipped -> lever | D5 rawFP | trueFP | trueFP/TP |",
           "|---|---|---|---|---|---|---|---|"]
    for seed in ("0", "1", "2"):
        r = d["seeds"].get(seed, {})
        if r.get("status") == "DONE":
            e = r["estimate"]; corr = r["corrected"]
            Ln.append("| %s | DONE | %+0.4f | %.3g | %d -> %d | %d -> %d | %d -> %d | %s |"
                      % (seed, e["theta"], e["p"], r["base_d5_tp"], r["lever_d5_tp"],
                         r["base_d5_rawFP"], r["lever_d5_rawFP"],
                         corr["shipped"]["5"]["true_fp"], corr["split"]["5"]["true_fp"],
                         r["trueFP_per_gain"]))
        else:
            Ln.append("| %s | %s | - | - | - | - | - | - |"
                      % (seed, r.get("status", "NOT-ATTEMPTED")))
    s1, s2 = d["seeds"].get("1", {}), d["seeds"].get("2", {})
    Ln += ["", "**Verdict: DEAD.** Seed 0 adds **0 TP** (901 -> 901); theta 0.0000, "
           "p=0.724. The held-out runs are recorded for completeness: "
           "seed 1 %d -> %d TP (theta %+0.4f, p=%.3g); seed 2 %d -> %d TP "
           "(theta %+0.4f, p=%.3g)."
           % (s1.get("base_d5_tp", 0), s1.get("lever_d5_tp", 0),
              s1.get("estimate", {}).get("theta", 0.0), s1.get("estimate", {}).get("p", 1.0),
              s2.get("base_d5_tp", 0), s2.get("lever_d5_tp", 0),
              s2.get("estimate", {}).get("theta", 0.0), s2.get("estimate", {}).get("p", 1.0)),
           "", "Base arm (seed 0): **901 TP / 169 rawFP / 1070 dets** — matches the frozen anchor.",
           "", "## All levers (routing rule: only the dominant bucket may run)", "",
           "| lever | name | target bucket | status | reason |",
           "|---|---|---|---|---|",
           "| dense-cluster split | concave-corner second-box proposal | no_proposal | **DEAD** | seed 0: 901 -> 901 TP, theta 0, p=0.724; held-out no rescue |",
           "| A | overlap-aware dedup | dedup | NOT-ATTEMPTED | census dedup = 0 |",
           "| B | cross-family agreement rescue | gate_side / gate_support | NOT-ATTEMPTED | census gate_side = 0, gate_support = 0 |",
           "| D | ridge-centre rescue of rejects | gate_support / gate_side | NOT-ATTEMPTED | census gate_support = 0, gate_side = 0 |",
           "| F | corner snap (5-8 px band) | corner_gt5 | NOT-ATTEMPTED | census corner_gt5 = 26; not the dominant bucket |",
           "", "## Reading", "",
           "- The overlap class is **effectively definitional**: 98/124 have no second-box "
           "candidate within D=12; the split recovered none on seed 0 and did not rescue "
           "the held-out seeds.",
           "- Do not re-propose fragment/hole merging, pairwise overlap merge, cut-side "
           "sliding, joint re-fitting, or per-image config selection; the overlap "
           "second-box proposal is now measured too.",
           "- The next actionable association after overlap is **candidate clutter** "
           "(`S4_quads` -10.2 pp/SD, `S5_edgefit_share` -9.0) via a hole-first / "
           "edgefit-fallback test — not another mask or orthogonality sweep.",
           "", "Artifacts: `benchmark/loop49_overlap_census.json`, "
           "`benchmark/loop49_levers.json`, `benchmark/loop49_panels/seed{0,1,2}_split.csv`, "
           "`proto/loop49_overlap_census.py`, `proto/loop49_levers.py`, "
           "`proto/loop49_heldout.py`. No detector/GT/data/C# changes.", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(Ln))
    print("verdict:", d["verdict"], flush=True)
    print("saved", OUT_JSON, OUT_MD, flush=True)


if __name__ == "__main__":
    main()