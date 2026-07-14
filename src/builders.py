"""ONNX graph builders for NeuroGolf networks (opset 10, IR 10).

All builders produce single-input ('input') single-output ('output') graphs
with statically-defined shapes. Tensors named 'input'/'output' are free in
the memory score, so we route data directly into 'output' wherever possible.
"""

import numpy as np
import onnx

from .harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS


def _model(nodes, initializers, value_infos=()):
    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(
        list(nodes), "graph", [x], [y], list(initializers),
        value_info=list(value_infos))
    return onnx.helper.make_model(
        graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def identity_network():
    """0 params, 0 memory -> 25.0 points."""
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    return _model([node], [])


def memorizer_network(Z, R, O, k_proj, G=None, rmax=30, w6=5):
    """Exact-match lookup network.

    Pipeline (all integer-valued float math, exact in float32):
      input one-hot -> base-11 cell codes packed 4/cell (Conv stride 4)
      -> random +-1 projection to k dims -> mismatch-count vs stored Z [N,k]
      -> one-hot row selector -> MatMul with stored outputs O
      (base-11^6 packed) -> arithmetic unpack -> Equal-decode to one-hot.

    Z: float [N, k] projected stored inputs; R: float [240, k] +-1 projection;
    O: float [N, rmax*w6] packed stored outputs covering only the output
    bounding box (rmax rows x 6*w6 cols; the rest of the canvas is zero-padded
    in-graph). With G [N, U] (0/1 input-row -> output-group map), O is
    [U, rmax*w6] deduplicated outputs instead.
    """
    nodes, inits = [], []

    def init(name, arr, dtype=np.float32):
        t = onnx.numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name)
        inits.append(t)
        return name

    def scalar(name, value):
        return init(name, np.array(value, dtype=np.float32))

    def n(op, inputs, out, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [out], **attrs))
        return out

    # --- encode input to packed base-11 codes ---
    # pack4: kernel [1,10,1,4], stride 4, pad right 2 -> [1,1,30,8] (240 codes)
    w4 = np.zeros((1, 10, 1, 4), np.float32)
    for c in range(10):
        for j in range(4):
            w4[0, c, 0, j] = (c + 1) * 11 ** (3 - j)
    init("w4", w4)
    n("Conv", ["input", "w4"], "p4", kernel_shape=[1, 4], strides=[1, 4],
      pads=[0, 0, 0, 2])
    init("shape_flat4", np.array([1, 240], np.int64), np.int64)
    n("Reshape", ["p4", "shape_flat4"], "x4")  # [1,240]

    # --- match against stored inputs via exact projection ---
    init("R", R)
    n("MatMul", ["x4", "R"], "z")              # [1,k]
    init("Z", Z)
    n("Sub", ["Z", "z"], "d")                  # [N,k]
    n("Abs", ["d"], "da")
    n("Clip", ["da"], "dc", min=0.0, max=1.0)
    n("ReduceSum", ["dc"], "mc", axes=[1], keepdims=1)   # [N,1] mismatch count
    n("Clip", ["mc"], "mcc", min=0.0, max=1.0)
    scalar("one", 1.0)
    n("Sub", ["one", "mcc"], "sel")            # [N,1], 1 iff exact match
    n("Transpose", ["sel"], "selr", perm=[1, 0])  # [1,N]

    # --- select stored output and unpack base-11^5 digits ---
    sel_name = "selr"
    if G is not None:
        init("G", G)
        sel_name = n("MatMul", ["selr", "G"], "selg")  # [1,U]
    init("O", O)
    n("MatMul", [sel_name, "O"], "yp")         # [1, rmax*w6]
    init("shape_grid", np.array([1, 1, rmax, w6], np.int64), np.int64)
    n("Reshape", ["yp", "shape_grid"], "y")    # [1,1,rmax,w6]
    rem = "y"
    digits = []
    for j in range(5):
        p = 11 ** (5 - j)
        scalar(f"pw{j}", float(p))
        n("Div", [rem, f"pw{j}"], f"q{j}_raw")
        n("Floor", [f"q{j}_raw"], f"q{j}")
        digits.append(f"q{j}")
        n("Mul", [f"q{j}", f"pw{j}"], f"qm{j}")
        rem = n("Sub", [rem, f"qm{j}"], f"r{j}")
    digits.append(rem)                          # last digit, [1,1,rmax,w6]
    for d in digits:
        n("Unsqueeze", [d], f"{d}_u", axes=[4])
    nodes.append(onnx.helper.make_node(
        "Concat", [f"{d}_u" for d in digits], ["dig"], axis=4))  # [1,1,rmax,w6,6]
    init("shape_code", np.array([1, 1, rmax, 6 * w6], np.int64), np.int64)
    n("Reshape", ["dig", "shape_code"], "code")

    # --- decode codes (color c <-> code c+1, empty 0) to one-hot ---
    # Equal doesn't take floats until opset 11, so compare as int32.
    n("Cast", ["code"], "code_i", to=onnx.TensorProto.INT32)
    init("chvec", np.arange(1, 11, dtype=np.int32).reshape(1, 10, 1, 1), np.int32)
    n("Equal", ["code_i", "chvec"], "eq")      # bool [1,10,rmax,6*w6]
    if rmax == 30 and w6 == 5:
        n("Cast", ["eq"], "output", to=DATA_TYPE)
    else:
        n("Cast", ["eq"], "onehot_bbox", to=DATA_TYPE)
        n("Pad", ["onehot_bbox"], "output", mode="constant", value=0.0,
          pads=[0, 0, 0, 0, 0, 0, 30 - rmax, 30 - 6 * w6])
    return _model(nodes, inits)


# max cells per 2^24-bounded half-word, per code base (base = #colors used + 1)
CELLS_PER_HALF = {2: 24, 3: 15, 4: 12, 5: 10, 6: 9, 7: 8, 8: 8, 9: 7, 10: 7, 11: 6}


def memorizer_v4(Z, R, O64, k_proj, codes, G=None, rin=30, w4=8,
                 rmax=30, wB=5, base=11):
    """v4 lookup: fp64 double-packed storage + int32 match + bbox/base trims.

    Differences vs memorizer_network (v3):
    - O64 is float64 [M, H]: element i packs TWO base^m words (m =
      CELLS_PER_HALF[base] output cells each), P[i] + P[i+H] * 2^26, where
      P = packed vector (length rmax*wB, padded to 2H). float64 keeps all
      values exact below 2^52, the 2^26 split is exact, and each half is
      < 2^24 so the cast back to float32 is exact too.
    - base = (#distinct output colors)+1 instead of always 11: fewer colors
      -> more cells per word -> proportionally smaller O table. `codes` maps
      color->code (1..base-1); unused channels decode against -1 (never hit).
    - match path: Cast projected input codes to int32, Equal vs stored Z
      (int32), ReduceSum, Greater -> ~2x less match memory than Sub/Abs/Clip.
    - input codes restricted to the input bounding box (rin rows, w4 packed
      cols) -> R is [rin*w4, k] instead of [240, k].
    """
    nodes, inits = [], []
    D = rin * w4
    L = rmax * wB
    H = O64.shape[1]
    m = CELLS_PER_HALF[base]

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype else np.ascontiguousarray(arr)
        inits.append(onnx.numpy_helper.from_array(a, name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [out], **attrs))
        return out

    # --- encode input to packed base-11 codes (bbox-trimmed) ---
    w4k = np.zeros((1, 10, 1, 4), np.float32)
    for c in range(10):
        for j in range(4):
            w4k[0, c, 0, j] = (c + 1) * 11 ** (3 - j)
    init("w4", w4k)
    n("Conv", ["input", "w4"], "p4", kernel_shape=[1, 4], strides=[1, 4],
      pads=[0, 0, 0, 2])                         # [1,1,30,8]
    src = "p4"
    if (rin, w4) != (30, 8):
        init("sl_st", np.array([0, 0], np.int64))
        init("sl_en", np.array([rin, w4], np.int64))
        init("sl_ax", np.array([2, 3], np.int64))
        src = n("Slice", ["p4", "sl_st", "sl_en", "sl_ax"], "p4s")
    init("shape_flat", np.array([1, D], np.int64))
    n("Reshape", [src, "shape_flat"], "x")       # [1,D]

    # --- match against stored inputs (int32 Equal path) ---
    init("R", R, np.float32)
    n("MatMul", ["x", "R"], "z")                 # [1,k]
    n("Cast", ["z"], "zi", to=onnx.TensorProto.INT32)
    init("Z", Z, np.int32)
    n("Equal", ["Z", "zi"], "em")                # bool [N,k]
    n("Cast", ["em"], "emh", to=onnx.TensorProto.FLOAT16)
    n("ReduceProd", ["emh"], "sel", axes=[1], keepdims=1)  # [N,1], 1 iff all match
    sel_name = "sel"
    if G is not None:
        n("Cast", ["sel"], "self", to=DATA_TYPE)
        init("G", G.T, np.float32)               # [U,N]
        sel_name = n("MatMul", ["G", "self"], "selg")  # [U,1]
    n("Cast", [sel_name], "seld", to=onnx.TensorProto.DOUBLE)  # [M,1]

    # --- select packed row (column vector) and split the fp64 double-pack ---
    init("O", O64.T.copy(), np.float64)          # [H,M]
    n("MatMul", ["O", "seld"], "yp")             # [H,1] fp64
    init("c26", np.array(float(1 << 26), np.float64))
    n("Div", ["yp", "c26"], "qd")
    n("Floor", ["qd"], "bd")
    n("Mul", ["bd", "c26"], "bm")
    n("Sub", ["yp", "bm"], "ad")
    n("Cast", ["ad"], "a32", to=DATA_TYPE)       # [H,1] each < 2^24
    n("Cast", ["bd"], "b32", to=DATA_TYPE)
    nodes.append(onnx.helper.make_node("Concat", ["a32", "b32"], ["pcat"], axis=0))
    src = "pcat"
    if 2 * H != L:
        init("ps_st", np.array([0], np.int64))
        init("ps_en", np.array([L], np.int64))
        init("ps_ax", np.array([0], np.int64))
        src = n("Slice", ["pcat", "ps_st", "ps_en", "ps_ax"], "pflat")
    init("shape_grid", np.array([1, 1, rmax, wB], np.int64))
    n("Reshape", [src, "shape_grid"], "y")       # [1,1,rmax,wB]

    # --- unpack base^(m-1) digits (strided packing: digit j of word u is the
    # cell at column j*wB + u, so digits concat straight into the code canvas
    # along the width axis -- no Unsqueeze/5-D Concat/Reshape needed) ---
    rem = "y"
    digits = []
    for j in range(m - 1):
        p = base ** (m - 1 - j)
        init(f"pw{j}", np.array(float(p), np.float32))
        n("Div", [rem, f"pw{j}"], f"q{j}_raw")
        n("Floor", [f"q{j}_raw"], f"q{j}")
        digits.append(f"q{j}")
        n("Mul", [f"q{j}", f"pw{j}"], f"qm{j}")
        rem = n("Sub", [rem, f"qm{j}"], f"r{j}")
    digits.append(rem)
    nodes.append(onnx.helper.make_node(
        "Concat", digits, ["code"], axis=3))     # [1,1,rmax,wB*m]
    cols = wB * m
    src = "code"
    if cols > 30:
        init("cs_st", np.array([0], np.int64))
        init("cs_en", np.array([30], np.int64))
        init("cs_ax", np.array([3], np.int64))
        src = n("Slice", ["code", "cs_st", "cs_en", "cs_ax"], "code_c")
        cols = 30

    # --- decode to one-hot (unused channels compare against -1: never hit) ---
    n("Cast", [src], "code_i", to=onnx.TensorProto.INT32)
    chvec = np.full((1, 10, 1, 1), -1, np.int32)
    for color, code in codes.items():
        chvec[0, color, 0, 0] = code
    init("chvec", chvec)
    n("Equal", ["code_i", "chvec"], "eq")        # bool [1,10,rmax,cols]
    if rmax == 30 and cols == 30:
        n("Cast", ["eq"], "output", to=DATA_TYPE)
    else:
        n("Cast", ["eq"], "onehot_bbox", to=DATA_TYPE)
        n("Pad", ["onehot_bbox"], "output", mode="constant", value=0.0,
          pads=[0, 0, 0, 0, 0, 0, 30 - rmax, 30 - cols])
    return _model(nodes, inits)


def pack4_codes_bbox(grid_canvas, rin=30, w4=8):
    """numpy mirror of the v4 in-graph encoder (bbox-trimmed pack4)."""
    padded = np.zeros((30, 32), np.int64)
    padded[:, :30] = grid_canvas
    w = 11 ** np.arange(3, -1, -1)
    full = (padded.reshape(30, 8, 4) * w).sum(axis=2)  # [30,8]
    return full[:rin, :w4].reshape(-1)                 # [rin*w4]


def pack_base_codes(code_canvas, rmax, wB, base):
    """Pack the top-left rmax x wB*m region into base^m words [rmax*wB].

    Strided layout: digit j of word u is the cell at column j*wB + u, so the
    in-graph digit chain reassembles the canvas with a single width-Concat.
    """
    m = CELLS_PER_HALF[base]
    region = np.zeros((rmax, wB * m), np.int64)
    cols = min(30, wB * m)
    region[:, :cols] = code_canvas[:rmax, :cols]
    w = base ** np.arange(m - 1, -1, -1, dtype=np.int64)
    return (region.reshape(rmax, m, wB) * w[:, None]).sum(axis=1).reshape(-1)


def pack_double(p, H):
    """Pack a word vector [L] (each < 2^24) into fp64 pairs [H], H=ceil(L/2)."""
    q = np.zeros(2 * H, np.float64)
    q[:p.shape[0]] = p
    return q[:H] + q[H:] * float(1 << 26)


def pack4_codes(grid_canvas):
    """numpy mirror of the in-graph pack4 encoder. grid_canvas: int [30,30] codes 0..10."""
    padded = np.zeros((30, 32), np.int64)
    padded[:, :30] = grid_canvas
    w = 11 ** np.arange(3, -1, -1)
    return (padded.reshape(30, 8, 4) * w).sum(axis=2).reshape(-1)  # [240]


def pack6_codes(grid_canvas, rmax=30, w6=5):
    """11^6 - 1 = 1.77M < 2^24, so 6 cells/float still keeps float32 exact.

    Packs only the top-left rmax x 6*w6 region (the output bounding box).
    """
    region = np.zeros((rmax, w6 * 6), np.int64)
    cols = min(30, w6 * 6)
    region[:, :cols] = grid_canvas[:rmax, :cols]
    w = 11 ** np.arange(5, -1, -1)
    return (region.reshape(rmax, w6, 6) * w).sum(axis=2).reshape(-1)  # [rmax*w6]


def conv_network(weights, kh, kw, bias=None, groups=1):
    """Single Conv straight to 'output'. 0 memory.

    weights: float array of shape [10, 10//groups, kh, kw]; bias: optional [10].
    groups=10 (depthwise) cuts params 10x when each output channel only needs
    its own input channel.
    """
    weights = np.asarray(weights, dtype=np.float32)
    w = onnx.helper.make_tensor(
        "W", DATA_TYPE, list(weights.shape), weights.flatten())
    inits, inputs = [w], ["input", "W"]
    if bias is not None:
        inits.append(onnx.helper.make_tensor(
            "B", DATA_TYPE, [10], np.asarray(bias, dtype=np.float32).flatten()))
        inputs.append("B")
    pads = [kh // 2, kw // 2, kh // 2, kw // 2]
    kwargs = {"group": groups} if groups != 1 else {}
    node = onnx.helper.make_node(
        "Conv", inputs, ["output"], kernel_shape=[kh, kw], pads=pads, **kwargs)
    return _model([node], inits)
