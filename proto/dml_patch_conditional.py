"""Merge the reverse-direction and joint-stratum conditional effects into an
already-computed panel analysis.

`proto/dml_analyze.py` now reports "effect of A inside stratum B" in BOTH
directions for every factor pair, plus the effect of A inside each JOINT
stratum of two other factors.  This script back-fills the missing entries into
an already-computed result file, so the artifacts stay in sync with the script
without re-running the (slow) DoubleML fits.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dml_analyze import (TCOLS, load_panel, paired_conditional,  # noqa: E402
                         paired_joint_conditional)


def patch(analysis_path, panel_path):
    with open(analysis_path, encoding="utf-8") as f:
        res = json.load(f)
    _, df, _ = load_panel(panel_path)
    n_img = int(res["n_images"])
    added = 0
    for oc, vals in res.get("outcomes", {}).items():
        pc = vals.setdefault("paired_conditional", {})
        for i, a in enumerate(TCOLS):
            for b in TCOLS[i + 1:]:
                for (eff, stratum) in ((a, b), (b, a)):
                    for bv in (0, 1):
                        key = "%s|%s=%d" % (eff, stratum, bv)
                        if key in pc:
                            continue
                        th, se, n = paired_conditional(df, oc, eff, stratum, bv)
                        pc[key] = {"theta": th, "se": se, "n": n}
                        added += 1
        jc = vals.setdefault("paired_joint_conditional", {})
        for eff in TCOLS:
            others = [c for c in TCOLS if c != eff]
            for i, f1 in enumerate(others):
                for f2 in others[i + 1:]:
                    for k, v in paired_joint_conditional(df, oc, eff, f1, f2).items():
                        if k not in jc:
                            jc[k] = v
                            added += 1
        print("  %s: %d conditional / %d joint entries" %
              (oc, len(pc), len(jc)))
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("patched %s (+%d entries, %d images)" % (analysis_path, added, n_img))


if __name__ == "__main__":
    patch(sys.argv[1], sys.argv[2])