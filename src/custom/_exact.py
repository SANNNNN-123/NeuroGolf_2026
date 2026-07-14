"""Small helpers for exact source builders.

These helpers are used for tasks whose live network is already a compact
one/few-op closed-form graph.  The task file still owns the graph structure and
initializers in Python source; it never loads `networks/taskNNN.onnx`.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def arr_b64(payload: str) -> np.ndarray:
    """Decode a base64-encoded `.npy` payload."""

    return np.load(io.BytesIO(base64.b64decode(payload)))


def tensor(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim > 0:
        arr = np.ascontiguousarray(arr)
    return numpy_helper.from_array(arr, name)


def model(name: str, nodes, inits, output_dtype=TensorProto.FLOAT, opset=11, value_infos=None):
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", output_dtype, [1, 10, 30, 30])],
        inits,
    )
    for vi in value_infos or []:
        graph.value_info.append(vi)
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
