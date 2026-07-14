"""Streamlit UI for inspecting NeuroGolf ONNX candidates.

Run:
    PYTHONPATH=. .venv/bin/streamlit run tools/onnx_viewer.py
"""

from __future__ import annotations

import html
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
from collections import Counter
from typing import Any

import numpy as np
import onnx
import onnxruntime
import streamlit as st

from src.harness import (
    ROOT,
    convert_to_numpy,
    evaluate,
    load_task,
    run_network,
    sanitize_model,
    score_network,
)
from tools.dashboard import render_dashboard


NETWORKS = ROOT / "networks"
CUSTOM = ROOT / "src" / "custom"
FRESH_CACHE = ROOT / "reports" / "fresh_cache"
REPORTS = ROOT / "reports"

ARC_COLORS = {
    -1: "#f3f4f6",
    0: "#111827",
    1: "#2563eb",
    2: "#dc2626",
    3: "#16a34a",
    4: "#facc15",
    5: "#6b7280",
    6: "#ec4899",
    7: "#f97316",
    8: "#06b6d4",
    9: "#7c3aed",
}


@st.cache_data
def load_manifest_tasks() -> dict[str, Any]:
    path = REPORTS / "manifest.json"
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    tasks = data.get("tasks", {})
    return tasks if isinstance(tasks, dict) else {}


def color_grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    h, w = grid.shape
    for r in range(min(h, 30)):
        for c in range(min(w, 30)):
            color = int(grid[r, c])
            if 0 <= color <= 9:
                arr[0, color, r, c] = 1.0
    return arr


def grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.shape == (1, 10, 30, 30):
        x = x[0]
    if x.shape != (10, 30, 30):
        raise ValueError(f"expected one-hot shape (1,10,30,30), got {arr.shape}")
    active = x > 0.5
    any_active = active.any(axis=0)
    grid = np.argmax(active, axis=0).astype(np.int16)
    grid[~any_active] = -1
    return grid


def normalize_arc_grid(grid: list[list[int]]) -> np.ndarray:
    out = np.full((30, 30), -1, dtype=np.int16)
    for r, row in enumerate(grid[:30]):
        for c, value in enumerate(row[:30]):
            out[r, c] = int(value)
    return out


def render_grid(grid: np.ndarray, key: str, mismatches: np.ndarray | None = None) -> None:
    rows = []
    for r in range(30):
        cells = []
        for c in range(30):
            color = ARC_COLORS.get(int(grid[r, c]), "#ffffff")
            border = "2px solid #ef4444" if mismatches is not None and mismatches[r, c] else "1px solid #d1d5db"
            title = html.escape(f"{key} ({r},{c}) = {int(grid[r, c])}")
            cells.append(
                f"<td title='{title}' style='width:13px;height:13px;"
                f"background:{color};border:{border};padding:0'></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        "<table style='border-collapse:collapse;table-layout:fixed'>"
        + "".join(rows)
        + "</table>",
        unsafe_allow_html=True,
    )


def load_python_builder(path: pathlib.Path, task: dict) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"candidate path does not exist: {path}"
    module_name = f"src.custom._viewer_candidate_{abs(hash(str(path)))}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None, f"cannot import candidate: {path}"
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "src.custom"
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        build = getattr(module, "build", None)
        if build is None:
            return None, f"{path} does not define build(task)"
        return build(task), None
    except Exception as exc:
        return None, f"build failed: {exc}"


def load_candidate_model(
    task_num: int,
    task: dict,
    mode: str,
    cand_path: str,
    code: str = "",
) -> tuple[Any | None, str | None]:
    if mode == "deployed":
        path = NETWORKS / f"task{task_num:03d}.onnx"
        if not path.exists():
            return None, f"missing deployed ONNX: {path}"
        return path, None
    if mode == "source":
        return load_python_builder(CUSTOM / f"task{task_num:03d}.py", task)
    if mode == "scratch path":
        if not cand_path.strip():
            return None, "enter a cand.py path, or choose deployed/source"
        return load_python_builder(pathlib.Path(cand_path).expanduser(), task)
    if mode == "code editor":
        if "def build(" not in code:
            return None, "code must define build(task)"
        tmp = tempfile.NamedTemporaryFile("w", suffix="_viewer_cand.py", delete=False)
        tmp.write(code)
        tmp.close()
        return load_python_builder(pathlib.Path(tmp.name), task)
    return None, f"unknown candidate mode: {mode}"


def make_session(model_or_path: Any) -> tuple[onnxruntime.InferenceSession | None, str | None]:
    try:
        if isinstance(model_or_path, (str, pathlib.Path)):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        sanitized = sanitize_model(onnx.ModelProto().FromString(model.SerializeToString()))
        if sanitized is None:
            return None, "sanitize failed"
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        return onnxruntime.InferenceSession(sanitized.SerializeToString(), options), None
    except Exception as exc:
        return None, f"ONNX Runtime load failed: {exc}"


def quick_score_model(model_or_path: Any, sample_input: np.ndarray) -> dict[str, Any]:
    """Profile one run for memory/params without verifying every stored example."""
    result = {"memory": None, "params": None, "points": 0.0, "error": None}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            if isinstance(model_or_path, (str, pathlib.Path)):
                model = onnx.load(model_or_path)
            else:
                model = model_or_path
            sanitized = sanitize_model(onnx.ModelProto().FromString(model.SerializeToString()))
            if sanitized is None:
                result["error"] = "sanitize failed"
                return result
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.profile_file_prefix = str(tmp / "viewer_prof")
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
            run_network(session, sample_input)
            trace_path = session.end_profiling()
            memory, params = score_network(sanitized, trace_path)
            if memory is None or params is None:
                result["error"] = "performance could not be measured"
                return result
            result["memory"] = memory
            result["params"] = params
            result["points"] = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
            return result
    except Exception as exc:
        result["error"] = f"quick score failed: {exc}"
        return result


def graph_summary(model_or_path: Any, limit: int = 120) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if isinstance(model_or_path, (str, pathlib.Path)):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        nodes = list(model.graph.node)
        op_counts = Counter(node.op_type for node in nodes)
        rows = []
        for idx, node in enumerate(nodes[:limit]):
            rows.append(
                {
                    "#": idx,
                    "op": node.op_type,
                    "name": node.name or "",
                    "inputs": ", ".join(node.input[:4]) + (" ..." if len(node.input) > 4 else ""),
                    "outputs": ", ".join(node.output),
                }
            )
        return {
            "nodes": len(nodes),
            "initializers": len(model.graph.initializer) + len(model.graph.sparse_initializer),
            "value_info": len(model.graph.value_info),
            "op_counts": dict(op_counts.most_common()),
            "rows": rows,
            "truncated": len(nodes) > limit,
        }, None
    except Exception as exc:
        return None, f"graph summary failed: {exc}"


def graph_dot(model_or_path: Any, limit: int = 80) -> tuple[str | None, str | None]:
    try:
        if isinstance(model_or_path, (str, pathlib.Path)):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        nodes = list(model.graph.node)
        visible = nodes[:limit]
        node_ids = {id(node): f"n{idx}" for idx, node in enumerate(visible)}
        output_owner: dict[str, str] = {}
        for node in visible:
            node_id = node_ids[id(node)]
            for out in node.output:
                if out:
                    output_owner[out] = node_id

        lines = [
            "digraph G {",
            "rankdir=LR;",
            "graph [bgcolor=\"transparent\", pad=\"0.15\", nodesep=\"0.35\", ranksep=\"0.45\"];",
            "node [shape=box, style=\"rounded,filled\", color=\"#334155\", fillcolor=\"#f8fafc\", fontname=\"Menlo\", fontsize=10];",
            "edge [color=\"#64748b\", arrowsize=0.7, fontname=\"Menlo\", fontsize=8];",
            "input [shape=oval, fillcolor=\"#dbeafe\", label=\"input\"];",
            "output [shape=oval, fillcolor=\"#dcfce7\", label=\"output\"];",
        ]
        for idx, node in enumerate(visible):
            node_id = node_ids[id(node)]
            raw_name = node.name or (node.output[0] if node.output else "")
            name = raw_name if len(raw_name) <= 28 else raw_name[:25] + "..."
            label = html.escape(f"{idx}: {node.op_type}\\n{name}")
            lines.append(f'{node_id} [label="{label}"];')

        for node in visible:
            dst = node_ids[id(node)]
            for inp in node.input:
                if not inp:
                    continue
                src = output_owner.get(inp)
                if src:
                    lines.append(f"{src} -> {dst};")
                elif inp == "input":
                    lines.append(f"input -> {dst};")
            for out in node.output:
                if out == "output":
                    lines.append(f"{dst} -> output;")

        if len(nodes) > limit:
            lines.append(f'trunc [shape=note, label="showing first {limit} of {len(nodes)} nodes"];')
        lines.append("}")
        return "\n".join(lines), None
    except Exception as exc:
        return None, f"graph render failed: {exc}"


def summarize_data_diff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed_counts = []
    changed_colors: Counter[int] = Counter()
    bboxes = []
    mostly_copy = 0
    examples = 0
    changed_examples = 0
    for row in rows:
        inp = np.asarray(row["input_grid"])
        out = np.asarray(row["expected_grid"])
        examples += 1
        changed = inp != out
        count = int(changed.sum())
        changed_counts.append(count)
        if count == 0:
            mostly_copy += 1
            continue
        changed_examples += 1
        if count <= 0.25 * 900:
            mostly_copy += 1
        vals, counts = np.unique(out[changed], return_counts=True)
        for value, n in zip(vals, counts):
            changed_colors[int(value)] += int(n)
        rr, cc = np.where(changed)
        bboxes.append((int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())))

    bbox_min = None
    bbox_max = None
    bbox_regular = False
    if bboxes:
        bbox_min = (min(b[0] for b in bboxes), min(b[1] for b in bboxes))
        bbox_max = (max(b[2] for b in bboxes), max(b[3] for b in bboxes))
        areas = [(b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in bboxes]
        bbox_regular = max(areas) <= 4 * max(1, min(areas))

    fixed_delta = len(changed_colors) <= 2 and changed_examples > 0
    return {
        "examples": examples,
        "changed_examples": changed_examples,
        "avg_changed_cells": float(np.mean(changed_counts)) if changed_counts else 0.0,
        "max_changed_cells": max(changed_counts) if changed_counts else 0,
        "changed_colors": dict(changed_colors),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_regular": bbox_regular,
        "mostly_copy": mostly_copy >= max(1, int(0.8 * examples)),
        "fixed_delta": fixed_delta,
    }


def model_cost_anatomy(model_or_path: Any, limit: int = 12) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if isinstance(model_or_path, (str, pathlib.Path)):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
        tensor_map = {t.name: t for t in list(graph.value_info) + list(graph.output)}
        node_by_output = {}
        for node in graph.node:
            for out in node.output:
                if out:
                    node_by_output[out] = node.op_type

        rows = []
        repeated = Counter()
        full_label_planes = []
        for name, value_info in tensor_map.items():
            if name in {"input", "output"} or not value_info.type.HasField("tensor_type"):
                continue
            tt = value_info.type.tensor_type
            if not tt.HasField("shape"):
                continue
            shape = []
            elems = 1
            dynamic = False
            for dim in tt.shape.dim:
                if not dim.HasField("dim_value"):
                    dynamic = True
                    break
                shape.append(int(dim.dim_value))
                elems *= int(dim.dim_value)
            if dynamic:
                continue
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
            bytes_ = elems * np.dtype(dtype).itemsize
            op = node_by_output.get(name, "?")
            repeated[(op, tuple(shape), str(np.dtype(dtype)))] += bytes_
            if len(shape) >= 2 and shape[-2:] == [30, 30] and bytes_ >= 900:
                full_label_planes.append(name)
            rows.append({"tensor": name, "op": op, "shape": shape, "dtype": str(np.dtype(dtype)), "bytes": int(bytes_)})

        rows.sort(key=lambda r: r["bytes"], reverse=True)
        expensive_ops = [
            {"op": op, "shape": list(shape), "dtype": dtype, "bytes": int(bytes_)}
            for (op, shape, dtype), bytes_ in repeated.most_common(8)
            if bytes_ >= 900
        ]
        return {
            "top_tensors": rows[:limit],
            "expensive_ops": expensive_ops,
            "has_full_label_plane": bool(full_label_planes),
            "full_label_planes": full_label_planes[:8],
            "has_repeated_planes": any(item["bytes"] >= 3600 for item in expensive_ops),
        }, None
    except Exception as exc:
        return None, f"cost anatomy failed: {exc}"


def review_questions(data_summary: dict[str, Any], cost_summary: dict[str, Any]) -> list[str]:
    questions = []
    if data_summary.get("mostly_copy"):
        questions.append("Output is mostly input copy: can this be a small overlay mask instead of rebuilding a full label/output plane?")
    if data_summary.get("fixed_delta"):
        questions.append("Changed cells use fixed colour(s): can colour routing be fixed colour rather than copied per-cell colour?")
    if data_summary.get("bbox_regular"):
        questions.append("Changed cells have regular bbox-like bounds: can a row/column profile or bbox factor replace a full 30x30 mask?")
    if cost_summary.get("has_full_label_plane"):
        questions.append("There is a full label plane: can the full label plane be folded into final Equal/Where/Einsum output?")
    if cost_summary.get("has_repeated_planes"):
        questions.append("Repeated full planes are expensive: can repeated scan/flood work collapse into walk-einsum or a stacked axis?")
    for item in cost_summary.get("expensive_ops", []):
        if item.get("op") in {"MaxPool", "Where", "QLinearConv", "Equal", "And", "Or"}:
            questions.append(f"Expensive {item['op']} block ({item['bytes']}B): is it solving a general case the generator never emits?")
            break
    if not questions:
        questions.append("No obvious visual-cost mismatch: inspect tasklog for known wall reasons before probing.")
    return questions


def task_context(task_num: int) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for path, key in [
        (REPORTS / "color_source_scan.json", "color"),
        (REPORTS / "walk_einsum_scan.json", "walk"),
        (REPORTS / "rect_sweep.json", "rect"),
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            if key == "rect" and isinstance(data, dict):
                row = next((r for r in data.get("rows", []) if int(r.get("task", -1)) == task_num), None)
            elif isinstance(data, list):
                row = next((r for r in data if int(r.get("task", -1)) == task_num), None)
            else:
                row = None
            if row:
                context[key] = row
        except Exception:
            continue
    return context


def stored_examples(task: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task.get(split, [])):
            rows.append({"source": split, "index": idx, "input": example["input"], "output": example["output"]})
    return rows


def fresh_examples(task_num: int) -> tuple[list[dict[str, Any]], str | None]:
    path = FRESH_CACHE / f"task{task_num:03d}.npz"
    if not path.exists():
        return [], f"fresh cache not found: {path}"
    try:
        data = np.load(path)
        inputs = data["inputs"]
        outputs = data["outputs"]
        rows = []
        for idx in range(len(inputs)):
            rows.append({"source": "fresh-cache", "index": idx, "input": inputs[idx], "output": outputs[idx]})
        return rows, None
    except Exception as exc:
        return [], f"fresh cache load failed: {exc}"


def example_to_arrays(example: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if example.get("source") == "fresh-cache":
        input_grid = np.asarray(example["input"], dtype=np.int16)
        expected_grid = np.asarray(example["output"], dtype=np.int16)
        return input_grid, expected_grid, color_grid_to_onehot(input_grid)
    benchmark = convert_to_numpy({"input": example["input"], "output": example["output"]})
    if benchmark is None:
        raise ValueError("example exceeds 30x30")
    return normalize_arc_grid(example["input"]), normalize_arc_grid(example["output"]), benchmark["input"]


def convertible_examples(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Drop examples the ONNX grid can't hold (>30x30) so the viewer never crashes.

    Official scoring also excludes these via convert_to_numpy returning None.
    """
    valid: list[dict[str, Any]] = []
    skipped = 0
    for example in examples:
        try:
            example_to_arrays(example)
        except Exception:
            skipped += 1
            continue
        valid.append(example)
    return valid, skipped


def run_examples(model_or_path: Any, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    session, err = make_session(model_or_path)
    if err:
        raise RuntimeError(err)
    assert session is not None
    rows = []
    for example in examples:
        input_grid, expected_grid, onehot_input = example_to_arrays(example)
        onehot_output = run_network(session, onehot_input)
        output_grid = grid_from_onehot(onehot_output)
        mismatches = output_grid != expected_grid
        rows.append(
            {
                "source": example.get("source", "stored"),
                "index": example.get("index", len(rows)),
                "input_grid": input_grid,
                "expected_grid": expected_grid,
                "output_grid": output_grid,
                "mismatches": mismatches,
                "ok": not bool(mismatches.any()),
            }
        )
    return rows


def preview_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        input_grid, expected_grid, _ = example_to_arrays(example)
        rows.append(
            {
                "source": example.get("source", "stored"),
                "index": example.get("index", len(rows)),
                "input_grid": input_grid,
                "expected_grid": expected_grid,
            }
        )
    return rows


def metric_text(result: dict[str, Any]) -> tuple[str, str, str]:
    memory = result.get("memory")
    params = result.get("params")
    points = result.get("points")
    return (
        f"{memory:,}" if isinstance(memory, int) else "-",
        f"{params:,}" if isinstance(params, int) else "-",
        f"{points:.6f}" if isinstance(points, float) else "-",
    )


def append_proposal(task_num: int, proposal: str, questions: list[str]) -> pathlib.Path:
    path = REPORTS / "tasklog" / f"task{task_num:03d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    checked = "\n".join(f"- {q}" for q in questions)
    text = (
        "\n\n## Human review proposal\n\n"
        f"{proposal.strip()}\n\n"
        "Prompt context:\n"
        f"{checked}\n"
    )
    with path.open("a") as f:
        f.write(text)
    return path


def _initial_task() -> int:
    """Initial task number for the viewer: ?task= query param, else drilled/default."""
    qp = st.query_params.get("task")
    if qp is not None and str(qp).isdigit():
        return max(1, min(400, int(qp)))
    return int(st.session_state.get("viewer_task", 187))


def render_viewer() -> None:
    st.title("🔬 NeuroGolf ONNX Viewer")

    with st.sidebar:
        st.header("Task")
        task_num = st.number_input(
            "Task", min_value=1, max_value=400, value=_initial_task(), step=1
        )
        manifest_tasks = load_manifest_tasks()
        manifest_row = manifest_tasks.get(f"{int(task_num):03d}") or manifest_tasks.get(str(int(task_num)))
        if manifest_row:
            st.caption("Current manifest")
            st.metric("points", f"{float(manifest_row.get('points', 0.0)):.6f}")
            c1, c2 = st.columns(2)
            c1.metric("mem", f"{int(manifest_row.get('memory', 0)):,}")
            c2.metric("params", f"{int(manifest_row.get('params', 0)):,}")
            st.caption(str(manifest_row.get("method", "-")))
        data_source = st.radio(
            "Examples",
            ["stored", "fresh cache"],
            horizontal=True,
            help="stored = bundled train/test/arc-gen examples. fresh cache = pre-generated random generator examples under reports/fresh_cache.",
        )
        max_examples = st.slider("Rows", min_value=1, max_value=200, value=30, step=1)
        st.divider()
        st.header("Candidate")
        mode = st.radio(
            "Model source",
            ["deployed", "source", "scratch path", "code editor"],
            help="deployed = networks/taskNNN.onnx. source = src/custom/taskNNN.py build(task). scratch path = any cand.py. code editor = edit a build(task) directly here.",
        )
        full_eval = st.checkbox(
            "Run full official eval",
            value=False,
            help="Slow: verifies all stored examples and profiles official cost. Keep off while browsing tasks.",
        )
        with st.expander("What do these mean?"):
            st.markdown(
                """
                - **stored**: task JSON에 들어있는 bundled `train`, `test`, `arc-gen` 예제입니다.
                - **fresh cache**: `reports/fresh_cache/taskNNN.npz`에 미리 생성해둔 랜덤 generator 예제입니다.
                - **deployed**: 현재 제출/패킹 대상인 `networks/taskNNN.onnx`입니다.
                - **source**: `src/custom/taskNNN.py`의 `build(task)`로 새로 만든 ONNX입니다.
                - **scratch path**: 임시 `cand.py` 경로를 넣어 `build(task)`를 실행합니다.
                - **code editor**: 이 UI에서 `build(task)` 코드를 직접 고쳐 실행합니다.
                """
            )
        cand_path = ""
        code = ""
        if mode == "scratch path":
            cand_path = st.text_input("cand.py path", "")
        elif mode == "code editor":
            default_path = CUSTOM / f"task{int(task_num):03d}.py"
            default_code = default_path.read_text() if default_path.exists() else "def build(task):\n    return None\n"
            code = st.text_area("Python candidate code", default_code, height=360)
        run_clicked = st.button("Build / generate outputs", type="primary")

    task = load_task(int(task_num))
    examples = stored_examples(task)
    if data_source == "fresh cache":
        examples, fresh_err = fresh_examples(int(task_num))
        if fresh_err:
            st.warning(fresh_err)
    examples, skipped = convertible_examples(examples)
    if skipped:
        st.warning(
            f"{skipped} example(s) exceed the 30×30 ONNX grid and were skipped "
            "(they are also excluded from official scoring)."
        )
    examples = examples[: int(max_examples)]

    if not examples:
        st.info("No examples available.")
        return

    if not run_clicked and mode in {"scratch path", "code editor"}:
        st.info("Edit the candidate, then press Run.")
        return

    model_or_path, model_err = load_candidate_model(int(task_num), task, mode, cand_path, code)
    if model_err:
        st.error(model_err)
        return

    assert model_or_path is not None
    first_input = example_to_arrays(examples[0])[2]
    eval_result = evaluate(model_or_path, task) if full_eval else quick_score_model(model_or_path, first_input)
    memory, params, points = metric_text(eval_result)
    total = "-"
    if eval_result.get("memory") is not None and eval_result.get("params") is not None:
        total = f"{eval_result['memory'] + eval_result['params']:,}"
    cols = st.columns(4 if not full_eval else 5)
    if full_eval:
        cols[0].metric("official pass / fail", f"{eval_result.get('pass', 0)} / {eval_result.get('fail', 0)}")
        metric_cols = cols[1:]
    else:
        metric_cols = cols
    metric_cols[0].metric("memory", memory)
    metric_cols[1].metric("params", params)
    metric_cols[2].metric("mem + params", total)
    metric_cols[3].metric("points", points)
    if not full_eval:
        st.caption("Fast mode: cost is profiled from one sample; pass/fail below is only for displayed rows. Enable full official eval for exact full stored verification.")
    if eval_result.get("error"):
        st.error(eval_result["error"])

    preview_rows = preview_examples(examples)
    data_summary = summarize_data_diff(preview_rows)
    cost_summary, cost_err = model_cost_anatomy(model_or_path)
    context = task_context(int(task_num))
    st.subheader("Human Review Board")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("changed examples", f"{data_summary['changed_examples']} / {data_summary['examples']}")
    r2.metric("avg changed cells", f"{data_summary['avg_changed_cells']:.1f}")
    r3.metric("output class", context.get("color", {}).get("class", "-"))
    r4.metric("walk repeat bytes", f"{context.get('walk', {}).get('repeat_bytes', 0):,}")

    with st.container(border=True):
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Data signals**")
            st.write(
                {
                    "mostly_copy": data_summary["mostly_copy"],
                    "fixed_delta": data_summary["fixed_delta"],
                    "bbox_regular": data_summary["bbox_regular"],
                    "changed_colors": data_summary["changed_colors"],
                    "bbox_min": data_summary["bbox_min"],
                    "bbox_max": data_summary["bbox_max"],
                }
            )
            if context:
                st.markdown("**Scanner context**")
                st.json(context, expanded=False)
        with right:
            st.markdown("**Cost anatomy**")
            if cost_err:
                st.warning(cost_err)
                cost_summary = {"expensive_ops": [], "has_full_label_plane": False, "has_repeated_planes": False, "top_tensors": []}
            assert cost_summary is not None
            st.write(
                {
                    "has_full_label_plane": cost_summary["has_full_label_plane"],
                    "has_repeated_planes": cost_summary["has_repeated_planes"],
                    "full_label_planes": cost_summary["full_label_planes"],
                }
            )
            st.dataframe(cost_summary["top_tensors"], use_container_width=True, hide_index=True)

    questions = review_questions(data_summary, cost_summary or {})
    with st.container(border=True):
        st.markdown("**Challenge prompts**")
        selected_questions = []
        for idx, question in enumerate(questions):
            if st.checkbox(question, value=False, key=f"q-{int(task_num)}-{idx}"):
                selected_questions.append(question)
        proposal = st.text_area(
            "Proposal",
            placeholder="예: output은 input copy + fixed green flood로 보임. full label plane 대신 walk-einsum 안에서 red/green mixer로 접을 수 있는지 probe.",
            height=100,
        )
        if st.button("Save proposal to tasklog"):
            if proposal.strip():
                path = append_proposal(int(task_num), proposal, selected_questions or questions)
                st.success(f"Saved to {path}")
            else:
                st.warning("Write a proposal first.")

    summary, summary_err = graph_summary(model_or_path)
    dot, dot_err = graph_dot(model_or_path)
    with st.expander("ONNX graph structure", expanded=True):
        if summary_err:
            st.warning(summary_err)
        elif summary is not None:
            s1, s2, s3 = st.columns(3)
            s1.metric("nodes", f"{summary['nodes']:,}")
            s2.metric("initializers", f"{summary['initializers']:,}")
            s3.metric("value_info", f"{summary['value_info']:,}")
            st.caption("Operator counts")
            st.dataframe(
                [{"op": op, "count": count} for op, count in summary["op_counts"].items()],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Node list")
            st.dataframe(summary["rows"], use_container_width=True, hide_index=True)
            if summary["truncated"]:
                st.caption("Node list truncated for UI speed.")
        if dot_err:
            st.warning(dot_err)
        elif dot:
            st.caption("Graph view")
            st.graphviz_chart(dot, use_container_width=True)

    if not run_clicked:
        st.info("Press Build / generate outputs to run the candidate. Data preview is shown below.")
        rows = preview_examples(examples)
        st.caption(f"Displayed examples: {len(rows)}")
        for row in rows:
            label = f"{row['source']} #{row['index']}"
            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("input")
                render_grid(row["input_grid"], f"{label}-input")
            with c2:
                st.caption("expected output")
                render_grid(row["expected_grid"], f"{label}-expected")
        return

    try:
        rows = run_examples(model_or_path, examples)
    except Exception as exc:
        st.error(str(exc))
        return

    ok_count = sum(1 for r in rows if r["ok"])
    st.caption(f"Displayed examples: {len(rows)}; displayed pass/fail: {ok_count}/{len(rows) - ok_count}")

    for row in rows:
        label = f"{row['source']} #{row['index']} - {'PASS' if row['ok'] else 'FAIL'}"
        st.markdown(f"**{label}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("input")
            render_grid(row["input_grid"], f"{label}-input")
        with c2:
            st.caption("expected output")
            render_grid(row["expected_grid"], f"{label}-expected")
        with c3:
            st.caption("candidate output")
            render_grid(row["output_grid"], f"{label}-output", row["mismatches"])
        with c4:
            st.caption("diff")
            diff_grid = np.where(row["mismatches"], 2, -1).astype(np.int16)
            render_grid(diff_grid, f"{label}-diff", row["mismatches"])


def _run_app() -> None:
    st.set_page_config(page_title="NeuroGolf", layout="wide")
    dashboard_page = st.Page(
        render_dashboard, title="Dashboard", icon="📊", default=True
    )
    viewer_page = st.Page(
        render_viewer, title="Task Viewer", icon="🔬", url_path="viewer"
    )
    # Expose the viewer page so the dashboard can drill down into it.
    st.session_state["_viewer_page"] = viewer_page
    st.navigation([dashboard_page, viewer_page]).run()


# Streamlit executes the target script as "__main__"; importing the module
# (e.g. in tests) must not launch the app.
if __name__ == "__main__":
    _run_app()
