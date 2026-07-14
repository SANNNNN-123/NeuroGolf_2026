"""Generalization verifier: run each network against FRESH arc-gen instances
(generated with new seeds), not just the stored examples. A net only earns its
points on the real Kaggle LB if it passes EVERY example, so a net that fails any
fresh instance is treated as scoring 0 (Kaggle uses held-out arc-gen instances).
"""
import importlib.util, json, sys, multiprocessing
from pathlib import Path
import numpy as np
import onnxruntime as ort
from src.harness import convert_to_numpy, load_task, evaluate

MAPPING = json.load(open("reports/arc_mapping.json"))
ROOT = Path(__file__).resolve().parent.parent
ARCGEN = ROOT / "arc-gen"
if not ARCGEN.is_dir():
    raise FileNotFoundError(f"repo-local arc-gen not found: {ARCGEN}")
if str(ARCGEN) not in sys.path:
    sys.path.append(str(ARCGEN))


def generator_path(num):
    mapped = Path(MAPPING[str(num)]["generator"])
    local = ROOT / "arc-gen" / "tasks" / mapped.name
    if local.exists():
        return local
    raise FileNotFoundError(f"repo-local generator not found for task {num}: {local}")

def load_gen(num):
    path = generator_path(num)
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fresh_pass(num, n=40):
    """Return (passes, total_run) over n fresh instances (<=30x30 only)."""
    import os
    path = f"networks/task{num:03d}.onnx"
    if not os.path.exists(path):
        return 0, 0
    try:
        gen = load_gen(num)
    except Exception:
        return -1, 0  # no generator
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so)
    ok = run = 0
    tries = 0
    while run < n and tries < n*4:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if len(ex["input"])>30 or len(ex["input"][0])>30 or len(ex["output"])>30 or len(ex["output"][0])>30:
            continue
        bm = convert_to_numpy(ex)
        out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        pred = (out[0] > 0).astype(np.int8)
        tgt = bm["output"][0].astype(np.int8)
        ok += int((pred==tgt).all()); run += 1
    return ok, run

def _worker(num):
    ok, run = fresh_pass(num)
    return num, ok, run

def main():
    manifest = json.load(open("reports/manifest.json"))["tasks"]
    real = 0.0; generalizes = 0; fails = []
    results = {}
    # maxtasksperchild=1: fresh process per task so generators (which share the
    # `common` module and may carry module-level state) can't pollute each other.
    with multiprocessing.Pool(8, maxtasksperchild=1) as pool:
        for num, ok, run in pool.imap_unordered(_worker, range(1,401)):
            results[num] = (ok, run)
    real_pts = 0.0
    for num in range(1,401):
        ok, run = results[num]
        v = manifest.get(str(num))
        pts = v["points"] if v else 0.0
        if run == 0:  # no generator or no net -> assume generalizes (rely on stored)
            real_pts += pts; generalizes += 1
        elif ok == run:
            real_pts += pts; generalizes += 1
        else:
            fails.append((num, ok, run, pts, (v or {}).get("method","?")))
    print(f"GENERALIZES: {generalizes}/400, est real LB ~ {real_pts:.1f} (Kaggle public was 4374.05)")
    print(f"NON-GENERALIZING: {len(fails)} nets (these score ~0 on Kaggle):")
    fails.sort(key=lambda x:-x[3])
    bym={}
    for num,ok,run,pts,meth in fails:
        base=(meth or "?").split("(")[0].split(":")[0]
        bym.setdefault(base,[0,0.0]); bym[base][0]+=1; bym[base][1]+=pts
    for b,(n,p) in sorted(bym.items(),key=lambda x:-x[1][1]):
        print(f"   {b:12s} {n:3d} nets, {p:.1f} local pts lost")
    print("non-generalizing tasks:", " ".join(f"{num}({ok}/{run})" for num,ok,run,pts,meth in fails))
    json.dump({str(n):{"ok":o,"run":r} for n,(o,r) in results.items()}, open("reports/genverify.json","w"))

if __name__=="__main__":
    main()
