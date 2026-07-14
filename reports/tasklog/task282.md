# task282 — b60334d2

**Rule:** Fixed 9x9 grid, background 0. Input places 3-4 gray(5) pixels at (r,c),
r,c in [1,7], pairwise non-overlapping (their 3x3 stamps never collide). For EACH
gray input pixel the OUTPUT paints blue(1) at the 4 orthogonal neighbours, gray(5) at
the 4 diagonal neighbours, and the centre becomes background (the input gray is NOT
copied). Pure local stamp; no detection, no argmax. Input colours {0,5}; output {0,1,5}.
**Current:** 18.198 pts, mem-0 single `Conv(input, W[10,10,3,3])` whose output IS the
graph output, mem 0, params 900. ⚠️ CORRECTION: a prior log claimed this public net has
"FRESH-RATE 0.00 (non-generalizing)". That was WRONG — re-verified the EXACT
networks/task282.onnx against the file-path-loaded generator in an ISOLATED process:
**200/200 fresh**. The base net generalizes perfectly. (The earlier 0.00 was almost
certainly a verify-harness contamination — the documented /tmp/arc-gen sys.path /
shared-generator-state pitfall.)
**Target tier:** detection/floor — genuine cross-channel 3x3 spatial neighbourhood op
that must also emit the subtractive bg channel-0 => the MEM-0 SINGLE-CONV HARD floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| prior | slice ch5 + banded 3x3 Conv + bands -> Equal | A | 1872 | 42 | 17.44 | 200/200 | exact but WORSE than mem-0 (1872 mem kills it) |
| 1 | dense Conv [10,10,3,3] (= public net) | floor | 0 | 900 | 18.198 | 200/200 | exact, at floor — this file |
| 2 | sparse_initializer -> Conv (26 nz) | — | — | — | — | — | BLOCKED: full_check shape-inf rejects sparse W ("unsupported type sparse_tensor(float)") -> "could not be measured" |
| 3 | Constant(sparse_value) -> Conv (opset11) | — | 3600 | 26 | 16.804 | 265/265 | WORSE: dense Constant output = 3600B intermediate |
| 4 | Slice input to ch{0,5} -> Conv [10,2,3,3] | — | 7200 | 180 | 16.09 | — | WORSE: [1,2,30,30] slice intermediate |

## Best achieved
18.198 @ mem 0 params 900 — matches the public net; this file reproduces it exactly,
fresh 200/200. Beats prior best stored? NO improvement available — INFEASIBLE to beat
by +0.3. (The prior session's 17.44 "candidate" is a REGRESSION vs the mem-0 18.20;
do NOT adopt it.)

## Irreducible-floor analysis
The Conv reads the [1,10,30,30] input DIRECTLY (free) and writes the [1,10,30,30]
output DIRECTLY (free), so mem=0 and the only cost is the weight initializer.
- O=10 forced: bg ch0 is a REQUIRED non-copy channel (out0 = in0 - 3x3 ring of in5,
  +1 at dot centre so the former gray cell -> bg).
- I=10 forced: the in5->{out0,out1,out5} cross-channel map breaks any group partition;
  reading fewer input channels needs an input Slice = a >=7200B 30x30 intermediate.
- k=3 forced by the stamp footprint (spans exactly +-1).
=> params = 10*10*3*3 = 900 irreducible as a dense mem-0 initializer. The 26-nonzero
weight cannot be golfed: every sparse route either is rejected by the scorer's strict
shape inference (sparse W -> Conv) or materialises a >=3600B dense intermediate. Any
decomposition (band-conv, slice) pays a >=1872B 30x30/9x9 plane that scores BELOW the
mem-0 18.20. The 9x9-active-canvas escape does NOT help: it only shrinks
intermediates, but the winning encoding has ZERO intermediates already.

## OPEN ANGLES (exhausted)
- sparse_initializer -> Conv: would give params 26 / mem 0 / pts 21.74 IF ORT's Conv
  accepted a sparse W AND onnx full_check did Conv shape-inference on sparse tensors.
  Both fail in the current toolchain. NOT achievable.

## INSIGHT (transferable)
⭐ MEM-0 SINGLE-CONV dot->box-stamp IS AT HARD FLOOR — a band-conv / slice decomposition
that LOOKS leaner (params 42 vs 900) is a REGRESSION because it materialises a
multi-hundred-byte 30x30/9x9 intermediate while the mem-0 conv has none (score is
25-ln(mem+params): 900 params at mem 0 = 18.20 beats 42 params at mem 1872 = 17.44).
When the public net is already mem-0 single-conv, the param count alone is the floor.
⭐ SPARSE-WEIGHT GOLF IS A DEAD END FOR Conv: `calculate_params` counts sparse tensors
by NONZERO count, but (1) sparse_initializer -> Conv is rejected by full_check strict
shape inference; (2) Constant(sparse_value) -> Conv passes but its DENSE output is a
counted intermediate (3600B here). Net loss for any small dense conv weight.
⭐ ALWAYS re-verify a "non-generalizing base net" claim with an ISOLATED file-path
generator load before trusting it — task282's prior "fresh 0.00" was a false alarm
from generator-state/sys.path contamination, and it nearly caused adoption of a
strictly worse 17.44 net over the correct 18.20.
