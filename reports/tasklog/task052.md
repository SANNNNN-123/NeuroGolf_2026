# task052 — 25d8a9c8

**Rule:** Input is a 3x3 grid of colours. Output is a 3x3 grid where every cell of row r is gray(5) if input row r is monochrome (all 3 cells equal), else black(0). Value depends only on the row (separable: per-row value broadcast across cols). Off-grid (r>=3 or c>=3) is all background.
**Current:** 17.89 pts (kojimar6275 import), Slice+ReduceSum+Where+Mul×2+Sum+Pad with three 360B fp32 [1,10,3,3] planes, mem 1185, params 43
**Target tier:** A — separable per-row output routed into the FREE/padded output; no flood/detection.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice 3x3, fp16 count, per-row uniform Equal==3, Where channel-onehot, Concat cols, Pad uint8 | A | 729 | 38 | 18.36 | 200/200 | ADOPT (+0.47) |

## Best achieved
18.36 @ mem 729 params 38 — adopted? Y. Beats prior 17.89? Y (+0.47).

## Irreducible-floor analysis
Dominant intermediates after ORT PrecisionFreeCast upcasts the fp16 working planes to fp32:
- Slice [1,10,3,3] fp32 = 360B (the active-block entry plane; cannot shrink — fp16/uint8 entry planes are upcast in the trace, and reducing the full input over cols first costs a 1200B [1,10,30,1] plane).
- ReduceSum [1,10,3,1] fp32 = 120B (per-row per-channel count).
- Concat [1,10,3,3] uint8 = 90B (the only 10-ch expansion plane; Pad output is FREE).
The v1 net's 1185B came from three full fp32 [1,10,3,3] planes (two Mul + one Sum). Killing those by building the channel one-hot at [1,10,3,1] (Where) and broadcasting cols via Concat (Mul rejects uint8) eliminated two of the three big planes.

## OPEN ANGLES (re-attack backlog)
- The 360B fp32 slice is the floor for this 10-ch active-block approach. To go lower would need to avoid materializing any 10-ch 3x3 plane — e.g. derive uniformity from a per-row scalar without slicing all 10 channels. Row-uniform iff max_c(value) == min_c(value) over the colour-index plane, but a colour-index plane is itself >= the slice cost here, so no clear win. Likely near the practical floor (~18.4) for a 3x3 grid task.

## INSIGHT (transferable)
For tiny fixed-size grids (3x3), the v1-style net often materializes multiple full 10-ch planes (Mul/Sum to combine colour one-hots). Replace with: compute the per-row/per-cell selector as a narrow [1,K,h,1] one-hot via Where, broadcast the remaining axis with Concat (Mul/Add reject uint8 but Concat and Where accept it), then Pad uint8 to 30x30 (Pad output is FREE). uint8 whole-pipeline + Concat-broadcast avoids the fp32 Mul/Sum 10-ch plane army. ORT still upcasts the fp16/uint8 ENTRY slice to fp32 in the trace, so the active-block slice (elems×4B) is the residual floor.
