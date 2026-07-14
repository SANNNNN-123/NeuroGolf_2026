# task342 — d89b689b

**Rule:** Grid is exactly 10x10. A 2x2 cyan(8) box sits at (brow,bcol)..(brow+1,bcol+1),
brow,bcol in [2,7]. Four single colored pixels (colors != cyan, all distinct) are scattered,
one in each quadrant relative to the box: top-left (row<brow,col<bcol)=colors[0],
top-right (row<brow,col>bcol+1)=colors[1], bottom-left (row>brow+1,col<bcol)=colors[2],
bottom-right (row>brow+1,col>bcol+1)=colors[3]. Output is all background(0) except the
2x2 box cells, each filled by its quadrant color: out[brow][bcol]=colors[0],
out[brow][bcol+1]=colors[1], out[brow+1][bcol]=colors[2], out[brow+1][bcol+1]=colors[3].
**Current:** 15.64 pts (public ext:kojimar6275)
**Target tier:** detection/scatter (data-dependent gather of 4 quadrant colors onto the box) —
not S/A/B because output color depends on a non-local pixel in the matching quadrant.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | quad masks x4 fp32 2D planes + corner planes | scatter | 14412 | 73 | 15.42 | — | works but heavy |
| 2 | collapse to top/bot col-profiles (1-D), fp16 working planes, outer-product L | scatter | 7684 | 73 | 16.04 | 200/200 | WIN +0.40 |

## Best achieved
16.04 @ mem 7684 params 73 — adopted? N (main adopts). Beats prior 15.64? Y (+0.40, ≥+0.3).

## Irreducible-floor analysis
- `in10` = the fp32 Slice of input to [1,10,10,10] = **4000B** dominates. All 10 channels are
  needed (colorV is a weighted channel sum) and colored pixels span all rows/cols 0-9, so the
  read can't be narrowed below 10x10. Slice preserves fp32 — casting to fp16 would add a second
  2000B plane on top of the 4000B slice, net worse.
- L pad [1,1,30,30] uint8 = **900B** is the standard label-map floor (output must be 30x30; in-grid
  bg must be channel-0-on, off-grid must be all-off via sentinel 10).
- Remaining ~2700B is fp16 10x10 working planes (colorV16, top, bot, Lt, Lb) — already halved.

## OPEN ANGLES
- Single Conv emitting a combined plane P (colored=digit, cyan=sentinel 100) to drop the separate
  cyan Slice — but P30 is still 3600B before slicing, so no net win over the 4000B in10 path.
- A 2-channel conv (colorV + cyan onehot) is 7200B, worse than the 4000B all-channel slice.
- The 4000B input read appears to be the practical floor for this task.

## INSIGHT (transferable)
⭐ Quadrant-color scatter collapses cheaply: when each quadrant holds exactly one colored pixel
and the col-bands of the two same-row pixels are disjoint, reduce the row-band 2D plane to a 1-D
col-profile (ReduceMax over rows) FIRST, then split left/right by 1-D col masks — turns 4×400B
masked-2D-plane reductions into 2×400B + tiny [1,1,1,W] tensors. Build the 2x2 output label as an
outer product rEQ[1,1,W,1] ⊗ rowvec[1,1,1,W] (color-per-column) instead of 4 separate corner planes.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 1510/17.63; ScatterND int64 idx machinery, all 16 scatter points semantically required; channels-first avoids 36kB transpose. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 1093 -> 1084 (+0.008); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].