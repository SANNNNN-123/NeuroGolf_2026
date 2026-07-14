# task398 — feca6190

**Rule:** Input is a 1x5 colour row with n nonzero colours (n=1..5). Output is
an s x s grid, s = 5n. Each input column c (colour v, including zeros) draws the
anti-diagonal ray output[r][s-1+c-r]=v for r in c..s-1. Equivalently the output
value at (r,j) depends only on the anti-diagonal t=r+j plus the grid extent:
column index c = t-(s-1); inside the s x s box show input column c when
0<=c<=4 (else black/colour-0); outside the box the canvas is all-zero.
**Current:** 16.16 pts, custom:task398 (two-level int32 gather table), mem 4428, params 2459
**Target tier:** A — value plane is data-dependent (colours copied from input) so not Tier-S copy; single fp16 plane routed into the free bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: int32 idx [30,30] via ROWMAP/ROWS dedup tables | A | 4428 | 2459 | 16.16 | — | baseline |
| 1 | KEY[30,30]+BIGVEC[51,5] -> per-tuple VALUE vec -> fp16 V plane -> Equal into free bool output | A | 2624 | 1201 | 16.75 | 200/200 | ADOPTED (+0.59) |
| 2 | same gather factorization but `data6/vtab/V/ARANGE` as uint8 | A | **1666** | **1193** | **17.04** | 1000/1000 | ADOPTED |

## Best achieved
17.04 @ mem 1666 params 1193 — beats prior live 16.75 by +0.289. Y.

## Irreducible-floor analysis
The single canvas-sized intermediate is the value plane V [30,30] uint8 = 900B.
It is the gather output (vtab indexed by KEY) and feeds the final Equal that
routes the 10-channel one-hot into the FREE bool output. The colour at each cell
is data-dependent (copied from input columns) so this cannot be a Tier-S
zero-mem copy. Everything else is sub-300B (bvec [51] i32, vtab [51] u8, z
scalar). Params dominated by KEY 900 + BIGVEC 255.

## OPEN ANGLES (re-attack backlog)
- KEY[30,30] (900 params) is separable KEY=MASTER[rcls[r],ccls[col]] (rcls/ccls
  len-30, MASTER 26x26 -> 736 params) but building it needs extra gather planes;
  net loss. A cheaper closed-form KEY (function of r+col and max(r,col) bands)
  could shave ~600 params -> ~+0.1.
- V is now uint8 900B; remaining wins require params/KEY factorization or eliminating the
  full value plane entirely, not dtype changes.

## INSIGHT (transferable)
⭐ Variable-output-size "draw rays / structured fill keyed on a small scalar
(here z = #zero-colours -> grid size)": factor the [Z,30,30] structural index
into a CONSTANT per-cell tuple-id plane KEY[30,30] + a small [Ntuples,Z] table
(BIGVEC[k,z]=idx), then DATA-DEPENDENT-map the structural index to actual output
VALUES *per tuple* (a length-Ntuples fp16 vector) BEFORE the big gather, so the
single canvas plane is the fp16 VALUE plane (1800B) gathered by the constant KEY
— route to the free bool output via Equal(V, arange). This both halves the plane
(fp16 vs int32, needs opset 11 for fp16 Equal — scorer checks domain not version)
AND collapses the row-dedup table (2280) to a tuple table (255). The per-cell
"tuple id over the scalar parameter" factorization beats row/col dedup whenever
the family of output planes is indexed by one small discrete parameter.
⭐ 2026-06-28 dtype update: Gather value planes feeding final `Equal` do not need fp16
when values are colour labels/sentinel. Use uint8 `data/vtab/V/arange` and halve the
only full plane.
