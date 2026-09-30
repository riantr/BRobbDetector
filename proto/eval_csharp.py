import os, sys, subprocess, json, shutil, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_expected, IMG

CSPROJ = r"D:\src\CodeArts\BRobbDetector\BRobbDetector\BRobbDetector.csproj"
TMP_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_cars_subset"
OUT_DIR = r"D:\src\CodeArts\BRobbDetector\tmp_cars_out"

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    method = sys.argv[2] if len(sys.argv) > 2 else "hole"
    flt = sys.argv[3] if len(sys.argv) > 3 else "none"
    cs = sys.argv[4] if len(sys.argv) > 4 else "hsvLabRgb"
    cm = sys.argv[5] if len(sys.argv) > 5 else "iter1"
    exp = load_expected()
    ids = [k for k, v in exp.items() if v >= 1 and os.path.exists(os.path.join(IMG, k + ".jpg"))]
    ids.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    ids = ids[:n]

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))

    for k in ids:
        src = os.path.join(IMG, k + ".jpg")
        dst = os.path.join(TMP_DIR, k + ".jpg")
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    print(f"Running C# detector on {n} images (method={method}, filter={flt}, cs={cs}, cm={cm})...")
    t0 = time.time()
    result = subprocess.run(
        ["dotnet", "run", "--project", CSPROJ, "--", "--cli", TMP_DIR, OUT_DIR, "--method", method, "--filter", flt, "--colorspace", cs, "--closemode", cm],
        capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace"
    )
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        return
    print(f"  Time: {time.time()-t0:.0f}s")

    exact = 0; over = 0; under = 0; total_exp = 0; total_det = 0
    for k in ids:
        jf = os.path.join(OUT_DIR, k + ".json")
        det = 0
        if os.path.exists(jf):
            with open(jf, encoding="utf-8") as fh:
                data = json.load(fh)
            det = len(data.get("infos", []))
        e = exp[k]
        d = det - e
        if d == 0: exact += 1
        elif d > 0: over += 1
        else: under += 1
        total_exp += e
        total_det += det

    print(f"\nimages={n}  exact={exact/n*100:.1f}%  over={over}  under={under}")
    print(f"total: expected={total_exp}  detected={total_det}  ratio={total_det/total_exp:.3f}")

    for k in ids:
        os.remove(os.path.join(TMP_DIR, k + ".jpg"))

if __name__ == "__main__":
    main()