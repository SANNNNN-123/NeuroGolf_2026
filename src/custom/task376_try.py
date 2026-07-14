"""Task 376 candidate — zigzag reflect-tile rows downward.

Rule (verified 39/39 in numpy): input is H x 17 (H in 3..6); output is
(4H-3) x 17 where output row r = input row tri(r), with
  p   = 2H-2
  tri = (H-1) - |(r mod p) - (H-1)|        (triangular reflection, period p)
rows r >= 4H-3 are outside the output grid (all zero).

Graph (opset 12). Build a (30,30) row-selection matrix S with
S[r,k]=1 iff k==tri(r) and r<4H-3, then output = MatMul(S, input) on the
height axis.
"""
import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    inits = [
        tensor("r", np.arange(30, dtype=np.float32).reshape(30, 1)),
        tensor("k", np.arange(30, dtype=np.float32).reshape(1, 30)),
        tensor("two", np.array(2.0, dtype=np.float32)),
        tensor("three", np.array(3.0, dtype=np.float32)),
        tensor("four", np.array(4.0, dtype=np.float32)),
        tensor("one", np.array(1.0, dtype=np.float32)),
        tensor("neg1", np.array(-1.0, dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("ReduceMax", ["input"], ["rowmax"], axes=[1, 3], keepdims=0),
        helper.make_node("ReduceSum", ["rowmax"], ["H"], keepdims=0),
        helper.make_node("Sub", ["H", "one"], ["m1"]),
        helper.make_node("Mul", ["two", "m1"], ["p"]),
        helper.make_node("Mul", ["four", "H"], ["fourH"]),
        helper.make_node("Sub", ["fourH", "three"], ["oh"]),
        helper.make_node("Div", ["r", "p"], ["rdivp"]),
        helper.make_node("Floor", ["rdivp"], ["rfl"]),
        helper.make_node("Mul", ["rfl", "p"], ["pq"]),
        helper.make_node("Sub", ["r", "pq"], ["rmod"]),
        helper.make_node("Sub", ["rmod", "m1"], ["d"]),
        helper.make_node("Abs", ["d"], ["ad"]),
        helper.make_node("Sub", ["m1", "ad"], ["tri"]),
        helper.make_node("Less", ["r", "oh"], ["ingrid"]),
        helper.make_node("Where", ["ingrid", "tri", "neg1"], ["trig"]),
        helper.make_node("Equal", ["trig", "k"], ["S"]),
        helper.make_node("Cast", ["S"], ["Sf"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["Sf", "input"], ["output"]),
    ]
    return model("task376_zigzag_reflect", nodes, inits, opset=12)
