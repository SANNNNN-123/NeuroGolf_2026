# task244 — 9f236235

**Rule:** Output is a `size×size` grid (size∈{3,4}) of pixels (colors 1..9, 0=bg).
`create_linegrid(output, magnifier, linecolor)` blows each output cell `(r,c)` into a
`mag×mag` solid block (mag=magnifier∈2..5) separated by full gridlines of `linecolor`;
spacing `sp=mag+1`, `actual_size = size*sp − 1`. The whole grid is then flipped
horizontally. So `output[r][c] = input[r*sp][actual_size−1 − c*sp]`. Both `sp` and `size`
vary per instance and must be recovered from the input.

**Current:** 14.63 pts, gen:biohack_new, mem 31685, params 85
**Target tier:** A (data-dependent double-Gather). Tier-S (single fixed-stride Gather)
is BLOCKED — stride `sp=mag+1` and grid `size` both vary per instance, so the gather
indices must be computed from the input, not baked as constants.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | recover sp/size scalars → double-Gather one-hot input (axis3 then axis2) → mask r,c≥size → Pad to output | A | 7949 | 171 | 16.00 | 200/200 (also 500/500) | KEEP |

## Best achieved
16.00 @ mem 7949 params 171 — adopted? N (build-only). Beats prior 14.63? Y (+1.37).

## Recovery details (all exact, validated 8000 fresh instances)
- Off-grid cells are ALL-ZERO one-hot (NOT channel-0=1). `colprof = ReduceMax(input,
  axes=[1,2])` = 1 iff column is inside the contiguous grid → `actual_size−1 = max col
  index with colprof=1` (via `ReduceMax(colprof * arange30)`).
- `actual_size` uniquely fixes `(size,sp)` for all values EXCEPT `actual_size==11`,
  which is `(3,4)` or `(4,3)`. Disambiguate: row index 2 is a full horizontal gridline
  iff `sp==3`; detected as "max over colors 1..9 of count-in-row-2 == actual_size".
- `(size,sp)` recovered via a length-24 lookup table indexed by `actual_size`, with the
  slot-11 correction `+ disamb*delta`.
- `rowidx = arange(4)*sp`, `colidx = (actual−1) − arange(4)*sp` (negatives/overshoot are
  fine — those cells are zeroed by the `(r<size)&(c<size)` mask before Pad).

## Irreducible-floor analysis
Dominant intermediate: the column-gathered plane `g1 [1,10,30,4]` = 1200 elem fp32 =
4800B. Inherent to a two-axis data-dependent gather — the first Gather collapses one
axis to 4 but leaves the other at 30. Second-largest: `row2` slice `[1,10,1,30]` (1200B,
for the disamb histogram). Everything else ≤640B.

## OPEN ANGLES (re-attack backlog)
- Replace the two sequential Gathers with a single GatherND (paired (R,C) indices) to
  drop the 4800B `g1` plane → `[1,10,4,4]` directly. Needs an index tensor `[4,4,2]`
  built per-instance (can't be an initializer since it's data-dependent); with batch_dims
  the index must be replicated over the 10 channels (`[1,10,4,4,2]`=320 int64=2560B),
  which is bigger than the 4800B saved unless done on a squeezed `[30,30]` view per
  channel. Estimated payoff: ~16.0 → ~17 if the index build stays small.
- Compute the disamb histogram from a smaller slice (only need cols 0..10 at actual=11)
  to shrink `row2` from 1200B; minor (~0.1 pt).
- Cast the gathered blocks to uint8 before Pad to halve the small tail planes; tiny.

## INSIGHT (transferable)
⭐ Off-grid padding in the one-hot input is ALL-ZERO, NOT channel-0=1 — so grid-extent /
`actual_size` detection must use `ReduceMax over ALL channels` (any-cell-occupied), never
`1 − channel0`. Variable-stride "un-magnify" tasks are Tier-A: recover the stride+size as
scalars (lookup table keyed on the easily-detected `actual_size`, with a one-bit
disambiguator only where the table is ambiguous), then build gather index vectors
arithmetically and double-Gather the FREE one-hot input. Out-of-range / negative gather
indices for the unused tail cells are harmless when masked before the final Pad-to-output.
