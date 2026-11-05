"""LOOP-47: build and render the detector-stage -> four-corner-recall causal DAG
with the pure-MoonBit DoWhy port (`riantr/moonbit_dowhy@0.0.33`; the requested
`riantr/dowhy` is not a registry module).

The MoonBit program (tmp_dowhy_graph/cmd/graph) constructs the DAG, prints DOT,
and runs backdoor identification for every stage. This driver captures that
output and renders the same edge list as Mermaid (SVG/PNG) so the graph is
visible without Graphviz.

Usage: python proto/causal_graph.py
"""
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MBT = os.path.join(ROOT, "tmp_dowhy_graph")
OUT_DOT = os.path.join(ROOT, "benchmark", "causal_graph.dot")
OUT_MMD = os.path.join(ROOT, "benchmark", "causal_graph.mmd")
OUT_SVG = os.path.join(ROOT, "benchmark", "causal_graph.svg")
OUT_PNG = os.path.join(ROOT, "benchmark", "causal_graph.png")
OUT_JSON = os.path.join(ROOT, "benchmark", "causal_graph.json")
OUT_MD = os.path.join(ROOT, "benchmark", "causal_graph.md")

LABEL = {
    "W": "W: image geometry",
    "Ngt": "Ngt: GT box count",
    "Red": "Red: red annotation coverage",
    "Ovl": "Ovl: GT overlap density",
    "S1": "S1 red mask (hue/sat/val)",
    "S2": "S2 denoise",
    "S3": "S3 holes / outer contours",
    "S4": "S4 hole quad fit",
    "S5": "S5 edgefit",
    "S6": "S6 side refine",
    "S7": "S7 side-validity gate",
    "S8": "S8 orthogonality + edge rule",
    "S9": "S9 red-support gate",
    "S10": "S10 dedup",
    "Y": "Y: four-corner recall @ D=5",
    "U": "U: latent image difficulty",
}


def run_mbt():
    r = subprocess.run(["moon", "run", "--target", "native", "cmd/graph"],
                       cwd=MBT, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("moon run failed:\n%s\n%s" % (r.stdout[-3000:], r.stderr[-3000:]))
    return r.stdout


def parse(out):
    dot = re.search(r"DOT_BEGIN\s*(.*?)\s*DOT_END", out, re.S)
    obs = re.search(r"OBS_DOT_BEGIN\s*(.*?)\s*OBS_DOT_END", out, re.S)
    rows = []
    for line in out.splitlines():
        if line.startswith("TREAT "):
            p = dict(x.split("=", 1) for x in [])
            m = re.match(r"TREAT (\S+) KEY (\S+) ADJ (.*)$", line)
            rows.append({"treatment": m.group(1), "key": m.group(2),
                         "adjustment": [x for x in m.group(3).split(",") if x]})
    return dot.group(1), obs.group(1), rows


def mermaid(dot):
    nodes = re.findall(r'^\s*"([^"]+)";\s*$', dot, re.M)
    edges = re.findall(r'^\s*"([^"]+)" -> "([^"]+)";\s*$', dot, re.M)
    stages = [n for n in nodes if n.startswith("S") and n != "Y"]
    ctx = [n for n in nodes if n in ("W", "Ngt", "Red", "Ovl")]
    lat = [n for n in nodes if n == "U"]
    L = ["flowchart LR"]
    L.append("  classDef ctx fill:#eef2ff,stroke:#4455aa;")
    L.append("  classDef stage fill:#eef7ee,stroke:#337755;")
    L.append("  classDef out fill:#feecec,stroke:#aa3333;")
    L.append("  classDef latent fill:#f6f6f6,stroke:#888888,stroke-dasharray: 4 3;")
    L.append("  subgraph CTX[Image / GT context]")
    for n in ctx:
        L.append('    %s["%s"]:::ctx' % (n, LABEL[n]))
    for n in lat:
        L.append('    %s["%s"]:::latent' % (n, LABEL[n]))
    L.append("  end")
    L.append("  subgraph PIPE[Algorithm stages]")
    for n in stages:
        L.append('    %s["%s"]:::stage' % (n, LABEL[n]))
    L.append("  end")
    L.append('  Y["%s"]:::out' % LABEL["Y"])
    for a, b in edges:
        arrow = "-.->" if a == "U" else "-->"
        L.append("  %s %s %s" % (a, arrow, b))
    L.append("  Ovl -->|verified: -0.18 TP/img| Y")
    L.append("  S9 -->|admission is binding| S10")
    return "\n".join(L) + "\n"


def main():
    out = run_mbt()
    dot, obs, rows = parse(out)
    open(OUT_DOT, "w", encoding="utf-8").write(dot)
    open(OUT_MMD, "w", encoding="utf-8").write(mermaid(dot))
    # render (mmdc is installed; Graphviz dot is not)
    renders = {}
    mmdc = shutil.which("mmdc.cmd") or shutil.which("mmdc") or "mmdc"
    env = os.environ.copy()
    for chrome in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(chrome):
            env["PUPPETEER_EXECUTABLE_PATH"] = chrome
            break
    for ext, path in (("svg", OUT_SVG), ("png", OUT_PNG)):
        cmd = [os.environ.get("COMSPEC", "cmd"), "/c", mmdc,
               "-i", OUT_MMD, "-o", path, "-b", "white"] if mmdc.endswith(".cmd") else               [mmdc, "-i", OUT_MMD, "-o", path, "-b", "white"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        except Exception as exc:
            print("render %s failed: %s" % (ext, exc))
        renders[ext] = os.path.exists(path)
    data = {
        "graph": "detector stages -> four-corner recall @ D=5",
        "tool": "riantr/moonbit_dowhy@0.0.33 (MoonBit DoWhy 0.14 port)",
        "requested": "riantr/dowhy@latest (not a registry module)",
        "nodes": sorted(set(re.findall(r'^\s*"([^"]+)";\s*$', dot, re.M))),
        "edges": re.findall(r'^\s*"([^"]+)" -> "([^"]+)";\s*$', dot, re.M),
        "latent": "U (image difficulty) is in the DOT but removed for identification",
        "identification_observed_projection": rows,
        "renders": renders,
    }
    json.dump(data, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = [
        "# Causal graph — detector stages and four-corner recall",
        "",
        "Built with the pure-MoonBit DoWhy port **`riantr/moonbit_dowhy@0.0.33`**",
        "(the requested `riantr/dowhy@latest` does not exist in the registry).",
        "The graph is a **domain DAG**, not a learned graph; `U` is an unobserved",
        "image-difficulty node. Identification is run on the **observed projection**",
        "(U removed), so every adjustment set below assumes no hidden confounding.",
        "",
        "## Graph",
        "",
        "![causal graph](causal_graph.svg)",
        "",
        "Nodes: `W` image geometry, `Ngt` GT-box count, `Red` red annotation coverage,",
        "`Ovl` GT overlap density; `S1` red mask, `S2` denoise, `S3` holes/outer",
        "contours, `S4` hole quad fit, `S5` edgefit, `S6` side refine, `S7` side-validity",
        "gate, `S8` orthogonality + edge rule, `S9` red-support gate, `S10` dedup;",
        "`Y` four-corner recall @ D=5; `U` latent difficulty.",
        "",
        "## Backdoor adjustment sets (MoonBit DoWhy, observed projection)",
        "",
        "| treatment | backdoor key | adjustment set |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append("| %s | %s | %s |" % (r["treatment"], r["key"],
                                          ", ".join(r["adjustment"]) or "(none)"))
    lines += [
        "",
        "## What the graph encodes (evidence labels)",
        "",
        "- **Supported:** `Ovl -> Y` (OR2 P2-a: overlapping GT is detected at 66.0% vs 95.3%;"
        " theta −0.18 TP/img, p≈3e-9). `Red -> S1` (the hue 0–4 window is the mask fix;"
        " wide hue is harmful). `S3 -> S4/S5 -> S7 -> S8 -> S9 -> S10 -> Y` is the shipped"
        " pipeline order.",
        "- **Ceiling:** the oracle GT-stroke mask is **+109 TP** (1010 vs 901 at D=5), so the"
        " mask-to-candidate path is the largest causal headroom; it is not a deployable arm.",
        "- **Estimator-sensitive (do not over-read):** the six-stage panel's `S2` effect loses"
        " significance under the MoonBit linear PLR (LOOP-45: 2/6 replicate); `S3`/`S5`/`S6`"
        " flip sign or significance. The graph is the hypothesis; the DoubleML estimates are"
        " the evidence.",
        "- **Measured inert:** the orthogonality factor (`S8`) is causally inert (OR2: edge"
        " p=0.91, blend p=0.55, joint p=0.28, tol p=0.07); only the edge-band rule"
        " (`t_band`) moves, and that is a GT-rule/detection-rule mismatch.",
        "- **Binding gate:** admission (`S9`) is the binding constraint — in the S1–S6 loop"
        " only 311 of 936 YOLO proposals passed the gates; the proposal source was not the"
        " limit.",
        "- **Neutral:** dedup (`S10`) measured +0 TP / +0 trueFP in the S4 experiment.",
        "",
        "## Caveats",
        "",
        "- `U` is latent; the adjustment sets above are for the observed projection. Any"
        " hidden image-difficulty confounding can still bias the stage effects.",
        "- Stage features are post-treatment outputs; a stage's adjustment set is not a"
        " licence to read its coefficient as an intervention.",
        "- Do not re-run `card`/`side3`/`local`, fragment merging, cut-side sliding, joint"
        " re-fitting, global hue widening, or per-image config selection — all measured dead.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python proto/causal_graph.py",
        "```",
        "",
    ]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print("identification:", rows)
    print("renders:", renders)
    print("wrote", OUT_DOT, OUT_MMD, OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()