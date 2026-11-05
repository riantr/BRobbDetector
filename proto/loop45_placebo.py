"""LOOP-45 refutation: placebo (treatment permutation) on every REPLICATED
claim, estimated with the same MoonBit PLR.

For each replicated claim the treatment column is permuted (global shuffle,
seed fixed) 9 times; the placebo thetas form the null. The empirical p is
(1 + #{|theta_perm| >= |theta_obs|}) / (n_perm + 1), so the resolution is 0.1
at 9 permutations. Claims that did not replicate are recorded NOT-ATTEMPTED.

Usage: python proto/loop45_placebo.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop45_moonbit_dml as MB

OUT = os.path.join(MB.ROOT, "benchmark", "loop45_moonbit_dml.json")
VERDICT = os.path.join(MB.ROOT, "benchmark", "loop45_moonbit_dml_verdict.md")
N_PERM = 9


def main():
    d = json.load(open(OUT, encoding="utf-8"))
    results = []
    for k, c in enumerate(d["claims"]):
        if not c["replicated"]:
            results.append({"claim": c["claim"], "status": "NOT-ATTEMPTED",
                            "reason": "claim did not replicate; no refutation required",
                            "p_placebo": None, "n_perm": 0})
            continue
        case = [x for x in MB.build_cases() if x[0] == c["claim"]][0]
        sub, treat, outcome, x_cols = MB.load_case(c["panel"], c["treatment"],
                                                   c["outcome"], case[4])
        rng = np.random.RandomState(1234 + k)
        perms = []
        for j in range(N_PERM):
            s = sub.copy()
            s[treat] = rng.permutation(s[treat].values)
            MB.write_mbt(s, treat, outcome, x_cols)
            res, _ = MB.run_mbt()
            perms.append(float(res["c"]))
        obs = float(c["theta"])
        ge = sum(1 for v in perms if abs(v) >= abs(obs))
        p = (1.0 + ge) / (N_PERM + 1.0)
        results.append({"claim": c["claim"], "status": "DONE",
                        "method": "placebo treatment permutation",
                        "n_perm": N_PERM, "theta_observed": obs,
                        "placebo_thetas": [round(v, 6) for v in perms],
                        "p_placebo": round(p, 4),
                        "resolution": round(1.0 / (N_PERM + 1.0), 4),
                        "verdict": "PASS" if p <= 0.1 else "WEAK"})
        print("  %-22s obs %+0.4f  placebo [%+0.4f..%+0.4f]  p=%.2f  %s"
              % (c["claim"], obs, min(perms), max(perms), p, results[-1]["verdict"]),
              flush=True)
    d["refutations"] = results
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = ["", "## Refutation (placebo treatment permutation)", "",
             "Each replicated claim was re-estimated with the treatment permuted "
             "%d times (seed fixed). Empirical p = (1 + #{|theta_perm| >= |theta_obs|}) / (%d + 1); "
             "the resolution is %.2f." % (N_PERM, N_PERM, 1.0 / (N_PERM + 1.0)), "",
             "| claim | theta observed | placebo range | p_placebo | status |",
             "|---|---|---|---|---|"]
    for r in results:
        if r["status"] == "DONE":
            lines.append("| %s | %+0.4f | [%+0.4f, %+0.4f] | %.2f | %s |"
                         % (r["claim"], r["theta_observed"], min(r["placebo_thetas"]),
                            max(r["placebo_thetas"]), r["p_placebo"], r["verdict"]))
        else:
            lines.append("| %s | - | - | - | %s |" % (r["claim"], r["status"]))
    open(VERDICT, "a", encoding="utf-8").write("\n".join(lines) + "\n")
    print("done -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()