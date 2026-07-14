"""Generalization-gated custom-net adoption (use this, NOT raw pipeline keep-best).

Adopts src/custom/taskN.py ONLY if it (a) passes the stored examples and (b)
generalizes to freshly generated arc-gen instances, and it beats the current
net on the REAL metric (the current net counts as 0 pts if it fails fresh).

Usage: python -m src.adopt N
"""
import importlib, json, sys, os
import numpy as np
import onnx
import onnxruntime as ort
from src.harness import load_task, evaluate, convert_to_numpy
from src.genverify import fresh_pass, load_gen

NF = 120
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh_ok_path(path, num, gen, n=NF):
    if gen is None:
        return True
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so)
    run = tries = 0
    while run < n and tries < n * 5:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if max(len(ex["input"]), len(ex["input"][0]),
               len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        bm = convert_to_numpy(ex)
        out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all():
            return False
        run += 1
    return run > 0


def main():
    num = int(sys.argv[1])
    task = load_task(num)
    path = f"networks/task{num:03d}.onnx"
    manifest = json.load(open("reports/manifest.json"))["tasks"]
    arcgen = os.path.join(ROOT, "arc-gen")
    if os.path.isdir(arcgen) and arcgen not in sys.path:
        sys.path.append(arcgen)
    try:
        gen = load_gen(num)
    except Exception:
        gen = None

    # current net's REAL value (0 if it fails fresh)
    cur_pts = 0.0
    cur_gen = False
    if os.path.exists(path):
        ev0 = evaluate(path, task)
        if ev0["ok"]:
            ok, run = fresh_pass(num, NF)
            cur_gen = (run == 0 or ok == run)
            cur_pts = ev0["points"] if cur_gen else 0.0
    print(f"current: generalizes={cur_gen}, real={cur_pts:.2f}")

    mod = importlib.import_module(f"src.custom.task{num:03d}")
    importlib.reload(mod)
    model = mod.build(task)
    ev = evaluate(model, task)
    if not ev["ok"]:
        print(f"REJECT: custom fails stored eval ({ev['fail']} fail, err={ev['error']})")
        return
    cand_path = f"reports/candidates/_adopt_task{num:03d}.onnx"
    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
    onnx.save(model, cand_path)
    cand_gen = fresh_ok_path(cand_path, num, gen)
    print(f"candidate: stored {ev['points']:.2f}, generalizes={cand_gen}")
    if not cand_gen:
        print("REJECT: custom does not generalize to fresh instances")
        return
    if ev["points"] <= cur_pts + 1e-9:
        print(f"REJECT: custom {ev['points']:.2f} <= current real {cur_pts:.2f}")
        return
    onnx.save(model, path)
    manifest[str(num)] = {"points": ev["points"], "memory": ev["memory"],
                          "params": ev["params"], "method": f"custom:task{num:03d}"}
    json.dump({"tasks": manifest}, open("reports/manifest.json", "w"), indent=1)
    print(f"ADOPTED: task{num:03d} real {cur_pts:.2f} -> {ev['points']:.2f} (generalizing)")


if __name__ == "__main__":
    main()
