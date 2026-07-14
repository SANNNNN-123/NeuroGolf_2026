# task261 — a79310a0

**Rule:** A size<=7 BLACK square holds a set of CYAN (8) pixels. Output is the same
square, all BLACK, except every cyan pixel at (r,c) becomes a RED (2) pixel at
(r+1,c). Off-grid all 0. Every in-grid cell is one-hot (black or red), so the
output requires the background channel-0 set everywhere in-grid that isn't red.
This is a per-pixel cross-channel spatial shift: out2(r)=cyan(r-1);
out0(r)=black(r)+cyan(r)-cyan(r-1); all other channels 0.
**Current:** 19.70 pts, pure-param Conv[10,10,2,1] (kh=2, pads=[1,0,0,0]), mem 0, params 200
**Target tier:** mem-0 single-Conv — this is the explicit MEM-0 SINGLE-CONV-AT-FLOOR case.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice cyan + Pad to ch2 (red only) | S | 4350 | 15 | 0 | 0/200 | FAIL: omits bg ch0 (output is full one-hot, not sparse stamp) |
| 2 | colour-index L + Equal->free bool output | detect | 25980 | 29 | 14.83 | 200/200 | correct but ~7 full 30x30 planes |
| 3 | sparse_initializer Conv weight (4 nz) | S | — | — | 0 | 200/200 | runs, but scorer's shape_inference(strict) rejects sparse_tensor(float) Conv W -> "could not be measured" |
| 4 | Constant node sparse_value weight (4 nz) | S | 800 | 4 | 18.31 | 200/200 | scores, but Constant materialises dense [10,10,2,1]=200 elems = 800B intermediate |
| 5 | dense-initializer minimal 4-nz Conv | S | 0 | 200 | 19.70 | 200/200 | = public floor (best achievable) |

## Best achieved
19.70 @ mem 0 params 200 — adopted? N (ties public). Beats prior 19.70? N.

## Irreducible-floor analysis
mem-0 requires conv-in = FREE graph input (10ch) AND conv-out = FREE graph output
(10ch); the +1-row shift forces kh=2; kw=1 is minimal => weight element count is
pinned at 10*10*2*1 = 200, and a dense initializer is the ONLY zero-mem carrier
(initializers don't count as mem). The cross-channel reach (in8 -> both out0 and
out2, far apart) forbids grouping to shrink in-channels. So params=200 is hard.

## OPEN ANGLES (exhausted for sub-floor)
- sparse_initializer: BLOCKED — calculate_memory's onnx.shape_inference with
  strict_mode=True raises on a sparse_tensor(float) Conv weight (0 pts). Also the
  sanitizer never renames sparse_initializers (would break the W name link), but
  shape inference dies first.
- Constant sparse_value: scores (params=4) but the dense weight materialises as a
  node-output intermediate (800B), netting 18.31 < 19.70. No way to keep the
  Constant's output from counting as mem.
- Any 30x30 intermediate (single fp32 plane = 3600B) already caps at 16.8 < 19.70,
  so every non-conv reformulation loses; the conv's double-free-IO is the only
  sub-148-byte structure and it is exactly at 200.

## INSIGHT (transferable)
⭐ SPARSE LEVER IS DEAD FOR Conv WEIGHTS (two independent blocks, both verified):
(1) a **sparse_initializer** makes calculate_memory's `onnx.shape_inference.
infer_shapes(model, strict_mode=True)` throw "W ... unsupported type:
sparse_tensor(float)" -> scorer returns None ("performance could not be
measured"). (2) a **Constant node with sparse_value** scores fine and DOES count
only the stored-nonzero count as params (4), BUT it materialises the full DENSE
tensor as a node-output intermediate that counts as mem (200 elems x 4B = 800B).
So sparse cannot turn a 200-param mem-0 dense-init Conv into a 4-param mem-0 net.
The MEM-0 SINGLE-CONV-AT-FLOOR bail rule stands: a cross-channel spatial-shift
op that must emit the subtractive bg channel-0, with in=out=10 pinned to the free
IO and kh fixed by the footprint, has irreducible params = 100*kh*kw.
