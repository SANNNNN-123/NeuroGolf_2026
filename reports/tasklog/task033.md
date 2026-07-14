# task033 — 1e32b0e9

**Rule:** 17x17 grid = 3x3 of 5x5 cells split by full `linecolor` lines at rows/cols 5,11
(hollywood_squares). Inside each cell a small shape of `color` pixels sits at inner offsets
r,c in {1,2,3}. The reference shape P = the `color` pixels of the TOP-LEFT cell (megacell 0,0),
at canvas rows/cols 1..3. OUTPUT = INPUT plus P stamped (in `linecolor`) into ALL 9 cells (drawn
first), with the original `color` pixels overlaid last (color wins on overlap). Equivalently: the
ONLY delta vs input is that background pixels at stamp positions that are not color pixels flip
0 -> linecolor. color channel of output == color channel of input, unchanged.

**Current (prior):** ~14.08 pts, tier A label-map.
**Target tier:** A (separable per-cell), but the tiling of the top-left cell into all 9 cells is a
non-local copy → realized as a boolean double-MatMul; the only 10-channel tensor is routed into the
FREE output via Where, so effective cost is one [1,1,30,30] plane (B-floor-break territory).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Srow@P@ScolT tile + Where(cond, lc_onehot, input); fp16 planes; pad fp16 | A/fb | 7484 | 425 | 15.99 | 200/200 | works |
| 2 | remove unused line_mask init | A/fb | 7484 | 425 | 16.02 | — | params 714->425 |
| 3 | uint8 pad (half bytes vs fp16 for the 30x30 plane) | A/fb | 6873 | 425 | 16.11 | — | smaller |
| 4 | newpix = Greater(M - color_plane, 0) (drop notcolor + Mul) | A/fb | 6584 | 426 | 16.15 | 200/200 | BEST |

## Best achieved
16.145 @ mem 6584 params 426 — adopted? N (orchestrator gates). Beats prior 14.08 by +2.07 → YES.
Fresh 200/200, fully generalizes (train+test+arc-gen all pass, fresh held-out 200/200).

## Irreducible-floor analysis
Dominant intermediates: ch0 fp32 slice 1156B (forced — reading the fp32 input), newpix30 uint8
[1,1,30,30] 900B + cond bool 900B (the 30x30 plane the Where condition must match), and a short chain
of [1,1,17,17] fp16 work planes (578B each: ch0_16, nonbg, color_plane, M, diff). The 10-channel
output expansion is FREE (Where output is "output"); linecolor is recovered as a [1,10,1,1] one-hot
slice off a guaranteed line pixel (row 5, col 0) so no per-cell color plane is materialized. The
fp32 input read + one 30x30 condition plane are the floor for any input->output edit of this shape.

## OPEN ANGLES (re-attack backlog)
- Could the 17x17 work planes be collapsed further by folding nonbg/color_plane into the MatMul
  operands (contract the input slice directly)? Marginal (~0.05 pts), not pursued.
- The fp32 ch0 slice (1156B) is the single largest tensor — no obvious removal without casting the
  whole input (more expensive).

## INSIGHT (transferable)
⭐ "Stamp a recovered sub-pattern into a regular tiling of cells" = boolean DOUBLE-MatMul
`M = Srow @ P @ ScolT` where Srow[R,dr] = (R % stride == dr+offset) places the K pattern rows into
each cell block and is automatically 0 on gaps/lines/off-grid (no separate in-grid mask). When the
edit only ADDS one dynamic color at masked positions, do NOT build a [1,10,H,W] delta: use
`output = Where(cond[1,1,30,30], color_onehot[1,10,1,1], input)` — the broadcast happens inside Where
to produce the FREE output, so the dominant intermediate collapses to a single 30x30 condition plane.
Recover the dynamic color as a [1,10,1,1] one-hot by slicing input at a position guaranteed to hold it
(here a line pixel). Pad in uint8 not fp16 (half the bytes) since the plane is {0,1}; Pad rejects bool.
