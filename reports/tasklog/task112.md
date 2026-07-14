# task112 — 4938f0c2 (reflect the red pattern around the green box)

**Rule:** A 2x2 green(3) box sits at (brow,bcol)..(brow+1,bcol+1) on a 0 background. A
small red(2) arm-pattern is stamped into the FOUR quadrants around the box, reflected
across the box centre: a red arm pixel at relative (r,c) lands at row (brow-1)+dr*r,
col (bcol-1)+dc*c for each quadrant (dr,dc) in {(-1,-1),(-1,+1),(+1,-1),(+1,+1)}.
Algebraically row R reflects about brow+0.5 (R -> 2*brow+1-R) and col C about bcol+0.5
(C -> 2*bcol+1-C). The INPUT may show only the top-left quadrant (showall=0, 3/4 of the
time) or all four (showall=1); the OUTPUT always shows all four. So
output = symmetrize(input): OR of {red, flipRows, flipCols, flipBoth}, green box copied.
**Current (prior):** ~14.01 pts, tier ~A label.
**Target tier:** B (data-dependent 2-D reflection = boolean double-MatMul). NOT Tier A:
the symmetrized red is an arbitrary L-shaped arm pattern, 2-D coupled (each input pixel
maps to 4 output cells) — it is NOT a row⊗col separable rectangle, so a rowcond⊗colcond
cannot reproduce it (cross-pixel false positives). The reflection axis is data-dependent
(brow/bcol vary across the full grid) so no fixed Conv/permute → S/A blocked.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | red+green fp32 slices, two reflection MatMul matrices, 4-way OR via 3 Adds, anyc=ReduceMax(input,[1]) in-grid, greenb full plane | B | 35004 | 157 | 14.53 | 200/200 | works |
| 2 | Sum op for 4-way OR (−2 planes); in-grid as separable rowocc⊗colocc 1-D profiles (−3600) | B | 28104 | 157 | 14.75 | 200/200 | trim |
| 3 | drop green fp32 slice: green row/col occupancy via per-channel ReduceMax([3])/([2]) + ch3 Slice (1-D); separable green 2x2 box mask from (brow,bcol); reuse profiles for in-grid | B | **27444** | **163** | **14.77** | **300/300** | FINAL |

## Best achieved
**14.77 pts @ mem 27444, params 163 — 266/266 stored, fresh 300/300.** Adopted? **N**
(orchestrator gates adoption). Beats prior ~14.01? **Y (+0.76).** GENERALIZES.

## Irreducible-floor analysis
Dominant intermediates: the SEVEN fp16 [1,1,30,30] planes (= 12600 B) that realize the
data-dependent 2-D reflection — two reflection permutation matrices (mat_R, mat_C) and the
four MatMul results (red16, rB=R@red, rC=red@C, rD=R@red@C) plus the Sum (rsum). These are
irreducible: MatMul requires float operands (fp16 is the smallest ORT-accepted dtype; uint8
MatMul NOT_IMPLEMENTED), and the reflection couples r&c in 2-D so it cannot collapse to 1-D
profiles like the in-grid/green machinery did. The red fp32 channel-2 Slice (3600) is the
Cast source for red16 and cannot be obtained in fp16 (Slice preserves fp32). The two
[1,10,30,1]/[1,10,1,30] profiles (1200 each) supply both green box position and in-grid mask.
Everything else (green box, in-grid, label) is already 1-D-separable or ≤900 B.

## OPEN ANGLES (re-attack backlog)
- Shrink the reflection canvas: the figure is centred on the green box but brow/bcol range
  over the whole grid and grid size is variable 10..30, so a data-dependent crop would be a
  Gather (≥100 B + index machinery), likely net-neutral. Untried in detail.
- Fold red16 + one MatMul (row-symmetrize H=red16+R@red16, then H+H@C): same plane count
  (5 fp16), fewer MatMuls but scorer sums ALL intermediates so no memory win — verified.
- Single-matrix reflection if green&red shared one plane: blocked — green box maps onto
  itself so a summed reflection over-counts green (4×3=12) and corrupts the colour index;
  red must be reflected in isolation.

## INSIGHT (transferable)
⭐ A 4-fold REFLECTION SYMMETRIZATION about a data-dependent axis = the same boolean
double-MatMul idiom as task250's clamp-scatter, but with a REFLECTION permutation matrix
Mat[out,in] = Equal(2*b+1 - in_arange, out_arange) (axis = 2*b+1 where b = min colour
row/col), and the four reflections OR'd via Sum(input, R@input, input@C, R@input@C) then
`>0`. Use `Sum` (variadic) instead of chained Adds to collapse the OR to one plane.
⭐ Recover a box/object position AND build its separable mask without ever materializing its
full colour plane: take per-channel ReduceMax over one axis ([1,10,30,1]), Slice the colour
channel to a 1-D [1,1,30,1] profile, derive min-index scalar, then rebuild the 2x2 box as a
separable rowmask⊗colmask from the scalar — kills the 3600 B fp32 colour slice. The same
[1,10,30,1] profile, ReduceMax'd over channels, doubles as the in-grid row occupancy.
