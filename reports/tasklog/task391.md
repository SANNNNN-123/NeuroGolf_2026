# task391 — f8b3ba0a

**Rule:** INPUT renders a width×height bitmap onto a (2h+1)×(3w+1) canvas (cell (r,c) = a
2-px horizontal pair at [2r+1][3c+1..2], rest = bg 0). The bitmap uses exactly 4 distinct
colours (all 1..9): one dominant colour fills the grid, three others are sprinkled with
DISTINCT counts from {1,2,3,4}; the dominant count is always strictly largest. OUTPUT is a
3×1 column = the three NON-dominant colours sorted by DESCENDING count.
**Current:** 15.79 pts, ext:kojimar6275, mem 9992, params 29
**Target tier:** A/S — output is 3 sparse one-hot cells whose colour is a pure count-rank
function of per-channel reductions; no per-cell plane needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | count-rank, route into FREE output, forgot to exclude ch0 | A | 1000 | 68 | 0 | — | fail (canvas-bg ch0 outranks bitmap-bg, ranks shift by 1) |
| 2 | mask ch0 to 0 before ranking | A | 1040 | 78 | **17.98** | 200/200, 500/500 | **WIN** |

## Best achieved
17.98 @ mem 1040 params 78 — adopted? N (build-agent does not adopt). Beats prior 15.79? Y (+2.19)

## Irreducible-floor analysis
Not at floor; this is essentially tier-A. The only intermediates are tiny: cnt [1,10,1,1],
a [10,10] Greater + its fp32 cast (400B, the dominant term), rank [10,1], and an H[1,10,30,1]
bool. The [10,10] pairwise-rank cast dominates but is already trivial. Could shave by ranking
without the full Greater matrix, but pts are already >18-adjacent — no benefit.

## OPEN ANGLES
- Could compute rank with ArgSort-free banded accumulation, but no payoff (already ~18).

## INSIGHT (transferable)
⭐ "Sort the K rare colours by count" with a UNIQUE-count guarantee = pure rank function with
ZERO per-cell planes: cnt_k=ReduceSum(input,[2,3]); rank_k=ReduceSum(Greater(cnt_j,cnt_k)) over
a [K,K] matrix; route into the FREE bool output as Equal(rank,target_ramp) ∧ colmask, where the
target ramp uses a SENTINEL (99) for padding rows so unused/excluded ranks never match. MUST
mask channel 0 to 0 first — the CANVAS background (ch0) has the largest pixel count of all and
silently steals rank 0 from the true bitmap-background, shifting every label by one. Each input
cell being a 2-px pair is irrelevant (constant factor preserves order).
