"""Solve all tasks, keep the best network per task, emit manifest + scoreboard.

Keep-best contract: networks/taskXXX.onnx is only replaced when a candidate
scores strictly more points, so repeated runs (and new solver tiers) can only
improve the total. reports/manifest.json records method/points per task.
"""

import argparse
import datetime
import json
import multiprocessing
import zipfile
from .harness import ROOT, evaluate, load_task

NETWORKS = ROOT / "networks"
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "manifest.json"
N_TASKS = 400

def solve_custom(task, task_num=None):
    """Load src/custom/taskNNN.py if present and build its network."""
    import importlib
    try:
        mod = importlib.import_module(f"src.custom.task{task_num:03d}")
    except ModuleNotFoundError:
        return None
    model = mod.build(task)
    if model is None:
        return None
    return model, {"method": f"custom:task{task_num:03d}"}


SOLVER_CHAIN = {
    "custom": solve_custom,
}


def load_manifest():
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            return {int(k): v for k, v in json.load(f)["tasks"].items()}
    return {}


def solve_one(job):
    task_num, methods = job
    task = load_task(task_num)
    path = NETWORKS / f"task{task_num:03d}.onnx"
    best = None
    if path.exists():
        ev = evaluate(path, task)
        if ev["ok"]:
            best = {"points": ev["points"], "memory": ev["memory"],
                    "params": ev["params"], "method": None}  # method filled from manifest
    for name in methods:
        try:
            cur_pts = best["points"] if best else 0.0
            if name == "custom":
                res = SOLVER_CHAIN[name](task, task_num=task_num)
            else:
                raise ValueError(f"unsupported method after cleanup: {name}")
        except Exception as e:
            print(f"task{task_num:03d}: {name} crashed: {e}", flush=True)
            continue
        if res is None:
            continue
        model, meta = res
        ev = evaluate(model, task)
        if not ev["ok"]:
            print(f"task{task_num:03d}: {meta['method']} failed official eval: "
                  f"{ev['fail']} fail, err={ev['error']}", flush=True)
            continue
        if best is None or ev["points"] > best["points"] + 1e-9:
            import onnx
            onnx.save(model, path)
            best = {"points": ev["points"], "memory": ev["memory"],
                    "params": ev["params"], "method": meta["method"]}
    return task_num, best


def write_scoreboard(manifest):
    solved = {k: v for k, v in manifest.items() if v}
    total = sum(v["points"] for v in solved.values())
    by_method = {}
    for v in solved.values():
        m = (v.get("method") or "?").split("(")[0]
        by_method.setdefault(m, [0, 0.0])
        by_method[m][0] += 1
        by_method[m][1] += v["points"]
    lines = [
        "# NeuroGolf 2026 — Scoreboard",
        "",
        f"_Updated: {datetime.datetime.now():%Y-%m-%d %H:%M}_",
        "",
        f"**Total: {total:.2f} pts** — {len(solved)}/{N_TASKS} tasks solved "
        f"(max 10,000; unsolved tasks score 0)",
        "",
        "| method | tasks | points |",
        "|---|---:|---:|",
    ]
    for m, (n, pts) in sorted(by_method.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"| {m} | {n} | {pts:.2f} |")
    lines += ["", "## Per task", "", "| task | method | memory | params | points |", "|---|---|---:|---:|---:|"]
    for k in sorted(manifest):
        v = manifest[k]
        if v:
            lines.append(f"| {k:03d} | {v.get('method') or '?'} | {v['memory']} | {v['params']} | {v['points']:.2f} |")
        else:
            lines.append(f"| {k:03d} | — | | | 0.00 |")
    (REPORTS / "SCOREBOARD.md").write_text("\n".join(lines) + "\n")
    return total, len(solved)


def pack():
    out = ROOT / "submission" / "submission.zip"
    files = sorted(NETWORKS.glob("task*.onnx"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    print(f"packed {len(files)} networks -> {out} ({out.stat().st_size/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=f"1-{N_TASKS}")
    parser.add_argument("--methods", default="custom")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    NETWORKS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    manifest = load_manifest()

    if not args.report_only:
        lo, hi = (args.tasks.split("-") + [args.tasks])[:2]
        nums = range(int(lo), int(hi) + 1)
        methods = args.methods.split(",")
        jobs = [(n, methods) for n in nums]
        done = 0
        with multiprocessing.Pool(args.jobs) as pool:
            for task_num, best in pool.imap_unordered(solve_one, jobs):
                done += 1
                prev = manifest.get(task_num)
                if best:
                    if best["method"] is None and prev:
                        best["method"] = prev.get("method")
                    manifest[task_num] = best
                else:
                    manifest.setdefault(task_num, None)
                if done % 20 == 0 or best:
                    tag = f"{best['method']} {best['points']:.2f}" if best else "unsolved"
                    print(f"[{done}/{len(jobs)}] task{task_num:03d}: {tag}", flush=True)

    with open(MANIFEST, "w") as f:
        json.dump({"tasks": {str(k): v for k, v in sorted(manifest.items())}}, f, indent=1)
    total, solved = write_scoreboard(manifest)
    print(f"TOTAL: {total:.2f} pts, {solved}/{N_TASKS} solved")
    if args.pack:
        pack()


if __name__ == "__main__":
    main()
