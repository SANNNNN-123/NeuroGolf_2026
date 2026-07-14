"""Shared builder for the common.hpwl tasks (246: a2fd1cf0, 335: d4a91cb9).

Dot of color c0 at (r0,c0), dot of color c1 at (r1,c1); draw an L-path of
color cp between them. Same outer-product construction as task246 (see that
module's docstring), parameterized by the three colors.

Memory floor-break (fp16 MatMul chain):
  Cast the Concat outputs VrT [1,1,30,4] and VcT [1,1,4,30] to fp16 before the
  MatMul chain.  With a fp16 initializer My [1,10,4,4], the intermediate
  R10 = My @ VcT becomes [1,10,4,30] fp16 = 2400B (vs 4800B f32).  The second
  MatMul VrT16 @ R10 writes directly into the free fp16 output.

  Memory: 5520B vs 7440B.  Params unchanged at 520.
  Score: ~16.29 vs 16.02.

  All values in My are small integers (±1..4); fp16 represents them exactly.
  Values in VrT/VcT are 0/1 (occupancy) or small integers (cummax products) —
  exact in fp16.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build_hpwl(col0, col1, colp):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H16 = TensorProto.FLOAT16

    # row stream (f32): [1,1,30,1] vectors for the row dimension
    Wr = np.zeros((1, 10, 1, 18), np.float32)
    Wr[0, col0, 0, :] = 1.0
    Wr[0, col1, 0, :] = 2.0
    init("Wr", Wr)
    n("Conv", ["input", "Wr"], "vr", strides=[1, 18])
    n("Clip", ["vr"], "wr", min=0.0, max=1.0)
    n("Sub", ["vr", "wr"], "gr")
    n("MaxPool", ["wr"], "pr", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["wr"], "qr", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["pr", "qr"], "cr")
    n("ReduceMax", ["input"], "rowin", axes=[1, 3], keepdims=1)
    n("Concat", ["rowin", "vr", "gr", "cr"], "VrT", axis=3)   # [1,1,30,4] f32

    # col stream (f32): [1,1,1,30] vectors for the col dimension
    Wc = np.zeros((1, 10, 18, 1), np.float32)
    Wc[0, col0, :, 0] = 1.0
    Wc[0, col1, :, 0] = 2.0
    init("Wc", Wc)
    n("Conv", ["input", "Wc"], "vc", strides=[18, 1])
    n("Clip", ["vc"], "wc", min=0.0, max=1.0)
    n("Sub", ["vc", "wc"], "gc")
    n("MaxPool", ["wc"], "pc", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["wc"], "qc", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["pc", "qc"], "cc")
    n("ReduceMax", ["input"], "colin", axes=[1, 2], keepdims=1)
    n("Concat", ["colin", "vc", "gc", "cc"], "VcT", axis=2)   # [1,1,4,30] f32

    # per-channel outer-product coefficients (fp16 initializer to enable fp16 MatMul)
    My = np.zeros((1, 10, 4, 4), np.float16)
    My[0, 0, 0, 0] = 1
    My[0, 0, 1, 3] = -1
    My[0, 0, 2, 3] = 2
    My[0, 0, 3, 2] = -1
    My[0, col0, 1, 1] += 1
    My[0, col0, 1, 2] += -2
    My[0, col0, 2, 1] += -2
    My[0, col0, 2, 2] += 4
    My[0, col1, 2, 2] += 1
    My[0, colp, 1, 3] += 1
    My[0, colp, 1, 1] += -1
    My[0, colp, 1, 2] += 1
    My[0, colp, 2, 3] += -2
    My[0, colp, 2, 1] += 2
    My[0, colp, 2, 2] += -3
    My[0, colp, 3, 2] += 1
    init("My", My, dtype=np.float16)

    # Cast VrT and VcT to fp16 before MatMul to halve R10 memory:
    # VrT_f32 (480B) + VrT16 (240B) + VcT_f32 (480B) + VcT16 (240B)
    # = 1440B vs 960B f32-only — but R10 goes from 4800B to 2400B.
    # Net: 7440B → 5520B.
    n("Cast", ["VrT"], "VrT16", to=H16)           # [1,1,30,4] fp16 = 240B
    n("Cast", ["VcT"], "VcT16", to=H16)           # [1,1,4,30] fp16 = 240B
    n("MatMul", ["My", "VcT16"], "R10")            # [1,10,4,30] fp16 = 2400B
    n("MatMul", ["VrT16", "R10"], "output")        # [1,10,30,30] fp16 → free output

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", H16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 10)])
