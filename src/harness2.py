"""Relaxed local harness for NeuroGolf 2026.

Same bundled correctness gate as src.harness, but memory scoring uses
onnx.shape_inference(..., strict_mode=False) so Conv / ConvTranspose /
MaxPool graphs with negative pads can be scored locally.

ORT already runs these nets correctly (fail=0); the stock harness rejects
them only because strict shape inference errors on negative pads. Use this
module when evaluating submission folders or merge picks that include
LB-valid negative-pad graphs. For strict parity with data/neurogolf_utils.py,
keep using src.harness.
"""

from __future__ import annotations

import math
import pathlib
import tempfile

import numpy as np
import onnx

from src.harness import (
    DATA_DIR,
    EXCLUDED_OP_TYPES,
    FILESIZE_LIMIT_IN_BYTES,
    GRID_SHAPE,
    ROOT,
    calculate_params,
    convert_to_numpy,
    load_task,
    run_network,
    sanitize_model,
    verify_subset,
)
import onnxruntime

PROFILE_ITEMSIZE = {
    "float": 4,
    "float16": 2,
    "double": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "bool": 1,
}


def _trace_tensor_bytes(shape_dict: dict) -> int | None:
    for dtype_name, dims in shape_dict.items():
        itemsize = PROFILE_ITEMSIZE.get(dtype_name)
        if itemsize is None:
            continue
        return itemsize * math.prod(dims)
    return None


def calculate_memory(model, trace_path):
    """Like harness.calculate_memory, but tolerates negative-pad ops."""
    import json

    onnx.checker.check_model(model, full_check=False)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=False).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None
    node_outputs = {}
    tensor_names = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)
    tensor_memory = {}
    tensor_dtypes = {}
    tensor_map = {
        t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            continue
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            if tensor_name not in ("input", "output") and tensor_type.HasField("elem_type"):
                tensor_dtypes[tensor_name] = onnx.helper.tensor_dtype_to_np_dtype(
                    tensor_type.elem_type
                )
            continue
        num_elements = 1
        static_shape = True
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                static_shape = False
                break
            num_elements *= dim.dim_value
        if tensor_name in ("input", "output"):
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_dtypes[tensor_name] = np_dtype
        if static_shape:
            tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize

    seen = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)

    with open(trace_path, "r") as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name in ("input", "output"):
                continue
            if output_name in tensor_dtypes:
                itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
                mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            else:
                mem = _trace_tensor_bytes(shape_dict)
                if mem is None:
                    continue
            prev = tensor_memory.get(output_name, 0)
            tensor_memory[output_name] = max(prev, mem)
    return sum(tensor_memory.values())


def score_network(sanitized, trace_path):
    for node in sanitized.graph.node:
        if node.op_type.upper() in EXCLUDED_OP_TYPES:
            return None, None
        if "Sequence" in node.op_type:
            return None, None
    return calculate_memory(sanitized, trace_path), calculate_params(sanitized)


def evaluate(model_or_path, examples, keep_failures=False):
    """Full evaluation with relaxed memory scoring. Returns a result dict."""
    result = {
        "ok": False,
        "filesize": None,
        "memory": None,
        "params": None,
        "points": 0.0,
        "pass": 0,
        "fail": 0,
        "error": None,
        "failures": None,
        "relaxed_scoring": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        if isinstance(model_or_path, (str, pathlib.Path)):
            file_path = pathlib.Path(model_or_path)
        else:
            file_path = tmp / "model.onnx"
            onnx.save(model_or_path, file_path)
        filesize = file_path.stat().st_size
        result["filesize"] = filesize
        if filesize > FILESIZE_LIMIT_IN_BYTES:
            result["error"] = f"filesize {filesize} exceeds limit"
            return result
        try:
            sanitized = sanitize_model(onnx.load(file_path))
            if not sanitized:
                result["error"] = "sanitize failed"
                return result
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.profile_file_prefix = str(tmp / "prof")
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        except Exception as e:
            result["error"] = f"load: {e}"
            return result

        subsets = examples.get("train", []) + examples.get("test", [])
        r1, w1, f1 = verify_subset(session, subsets)
        r2, w2, f2 = verify_subset(session, examples.get("arc-gen", []))
        trace_path = session.end_profiling()
        result["pass"], result["fail"] = r1 + r2, w1 + w2
        if keep_failures:
            result["failures"] = {"arc-agi": f1, "arc-gen": f2}
        try:
            memory, params = score_network(sanitized, trace_path)
        except Exception as e:
            result["error"] = f"score: {e}"
            return result
        if memory is None or params is None or memory < 0 or params < 0:
            result["error"] = "performance could not be measured"
            return result
        result["memory"], result["params"] = memory, params
        if result["fail"] == 0:
            result["ok"] = True
            result["points"] = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("task_num", type=int)
    args = parser.parse_args()
    res = evaluate(args.model, load_task(args.task_num))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
