# task298 — bda2d7a6

**Rule:** A size×size grid (size=2·half, half∈{3,4}) is 4-fold mirror-symmetric. Cell (r,c)
belongs to ring m=min(r,c). Three DISTINCT colors c0,c1,c2 (=colors[0..2], black allowed)
are assigned by ring: input ring m = colors[m%3], output ring m = colors[(m+2)%3]. So at
every cell output_color = the color one step BACK in the 3-cycle: input c0→c2, c1→c0, c2→c1.
A pure per-instance color permutation. The whole grid is filled (no interior padding); ring
order is recoverable spatially — c0 at (0,0), c1 at (1,1), c2 at (2,2) (positions always
exist since size≥6). Off-grid cells (rows/cols ≥ size) are all-zero and must stay all-zero.
**Current:** 15.39 pts, ext:biohack_new, mem 14378, params 534
**Target tier:** S/recolor — output is a per-cell recolor of the input (no new geometry), so
the 10-ch expansion routes into the FREE padded output; cost floor = the active-region slice.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | nested Where(mask, color_onehot, …) on full 30×30 | S | 85632 | 75 | 13.64 | 200/200 | works but Where chain materializes 2 fp32 [1,10,30,30] planes |
| 2 | single Lout index plane + Equal→bool output, full 30×30 | S | 36024 | 49 | 14.51 | 200/200 | many fp16 30×30 planes dominate |
| 3 | slice to 8×8, Lout8, Pad→30×30 fp16, Equal→bool output | S | 6944 | 67 | 16.14 | 200/200 | Lout30 fp16 1800B is the big plane |
| 4 | slice 8×8, Equal on 8×8 → Cast uint8 → Pad→output | S | 6424 | 67 | 16.22 | 200/200 | best; avoids the 30×30 fp16 plane |
| 5 | cast inp slice to fp16 first | S | 7302 | 67 | 16.09 | 200/200 | WORSE — adds a plane (PrecisionFreeCast upcasts) |
| 6 | #4 but Cast colf→fp16 after Conv (rest fp16) | S | 6406 | 67 | 16.22 | 200/200, 500/500 | BEST |

## Best achieved
16.225 @ mem 6406 params 67 — adopted? N (orchestrator gates). Beats prior 15.39 by +0.83 → YES.
Fresh 200/200 and 500/500, all 267 stored examples pass.

## Irreducible-floor analysis
Dominant intermediate: the `inp` fp32 slice [1,10,8,8]=2560B (Slice preserves the fp32 input
dtype; casting it to fp16 ADDS a plane so it nets bigger). Next: the two 8×8 one-hot planes
oh8 bool [1,10,8,8]=640B + oh8u uint8 640B (needed to expand+pad into the 30×30 output; Pad
rejects bool so the uint8 carrier is required). Everything else is [1,1,8,8] (128B fp16) or
scalar. The output [1,10,30,30] uint8 is FREE (it is "output"). The full 10-ch slice is the
floor for any recolor that must read all color channels over the active region.

## OPEN ANGLES (re-attack backlog)
- Avoid the 10-ch `inp` slice: could colf come from MatMul contracting the channel axis of the
  FREE full input directly (per-channel batched matvec lever) then slice 8×8? Would also need
  occ; getting occ without a 10-ch tensor is the blocker (black cells need ch0). Est ≤0.1 pts.
- Could the two 640B one-hot planes collapse to one if a uint8-producing Equal existed (it
  doesn't in opset-11) — marginal.

## INSIGHT (transferable)
⭐ A per-instance CYCLIC RECOLOR (3 colors permuted by a fixed cycle, ring/region-determined
order) is Tier-S closed-form: recover each color as a scalar from a FIXED position, build a
SINGLE output color-index plane `Lout = Σ maskX · cyc(cX)` over disjoint per-color masks
(Equal(colf,cXs) gated by an in-grid ReduceMax-occupancy so a BLACK color can't false-match
off-grid colf=0), then one-hot via Equal(Lout, arange). Operate on the generator-bounded
active canvas (here 8×8) and Pad the uint8 one-hot to 30×30 as the FREE output — the 10-ch
slice is the only real cost. Casting the input slice to fp16 BACKFIRES (extra plane + ORT
PrecisionFreeCast upcast), but casting the post-Conv 1-ch colf to fp16 is free and shrinks
all downstream work planes.
