"""Local verification + scoring harness for NeuroGolf 2026.

Mirrors the official scoring in data/neurogolf_utils.py (2026-05-14 version)
exactly, minus the IPython/matplotlib display code, so results here should
match Kaggle's official scorer.
"""

import json
import math
import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID_SHAPE = [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
DATA_TYPE = onnx.TensorProto.FLOAT
IR_VERSION = 10
OPSET_IMPORTS = [onnx.helper.make_opsetid("", 10)]
FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
EXCLUDED_OP_TYPES = ["LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"]

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_task(task_num):
    with open(DATA_DIR / f"task{task_num:03d}.json") as f:
        return json.load(f)


def convert_to_numpy(example):
    benchmark = {}
    example_shape = (1, CHANNELS, HEIGHT, WIDTH)
    for mode in ["input", "output"]:
        benchmark[mode] = np.zeros(example_shape, dtype=np.float32)
        grid = example[mode]
        if max(len(grid), len(grid[0])) > 30:
            return None
        for r, _ in enumerate(grid):
            for c, color in enumerate(grid[r]):
                benchmark[mode][0][color][r][c] = 1.0
    return benchmark


def calculate_memory(model, trace_path):
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
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
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in ["input", "output"]:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize
        tensor_dtypes[tensor_name] = np_dtype

    seen = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

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
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], mem)
    return sum(tensor_memory.values())


def calculate_params(model):
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(d <= 0 for d in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d <= 0 for d in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                if any(d <= 0 for d in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def score_network(sanitized, trace_path):
    for node in sanitized.graph.node:
        if node.op_type.upper() in EXCLUDED_OP_TYPES:
            return None, None
        if "Sequence" in node.op_type:
            return None, None
    return calculate_memory(sanitized, trace_path), calculate_params(sanitized)


def sanitize_model(model):
    for node in model.graph.node:
        node.name = node.output[0]
        if "kernel_time" in node.output[0]:
            return None

    name_map, counter = {}, 0

    def get_safe_name(old_name):
        nonlocal counter
        if not old_name or old_name in ["input", "output"]:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for inp in model.graph.input:
        inp.name = get_safe_name(inp.name)
    for init in model.graph.initializer:
        init.name = get_safe_name(init.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = get_safe_name(node.input[i])
        for i in range(len(node.output)):
            node.output[i] = get_safe_name(node.output[i])
        if len(node.output) > 0 and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = get_safe_name(out.name)
    for vi in model.graph.value_info:
        vi.name = get_safe_name(vi.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def run_network(session, benchmark_input):
    result = session.run(["output"], {"input": benchmark_input})
    return (result[0] > 0.0).astype(float)


def verify_subset(session, example_subset):
    right, wrong = 0, 0
    failures = []
    for idx, example in enumerate(example_subset):
        benchmark = convert_to_numpy(example)
        if not benchmark:
            continue
        try:
            user_output = run_network(session, benchmark["input"])
            if np.array_equal(user_output, benchmark["output"]):
                right += 1
            else:
                wrong += 1
                failures.append(idx)
        except Exception as e:
            # ORT 1.2x raises onnxruntime.capi...Fail, not onnxruntime.ONNXRuntimeError
            # (that attribute no longer exists). Treat any session.run failure as a miss.
            if "onnxruntime" not in type(e).__module__:
                raise
            wrong += 1
            failures.append(idx)
    return right, wrong, failures


def evaluate(model_or_path, examples, keep_failures=False):
    """Full official-style evaluation. Returns a result dict."""
    result = {
        "ok": False, "filesize": None, "memory": None, "params": None,
        "points": 0.0, "pass": 0, "fail": 0, "error": None, "failures": None,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("task_num", type=int)
    args = parser.parse_args()
    res = evaluate(args.model, load_task(args.task_num))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
