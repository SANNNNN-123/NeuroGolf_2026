# task137 — 5c2c9af4

**Rule:** Input is a size×size grid (size 20..30) with exactly 3 pixels of one
colour at rows {row-s, row, row+s} on a ±1 diagonal (flip picks the diagonal),
spacing s in [2, size//4]. Output draws concentric square ring perimeters around
(row, col): cell (r,c) in-grid is `color` iff max(|r-row|,|c-col|) % s == 0, else
black; outside the (top-left) grid the canvas is all 0.
**Current:** 16.039 pts, custom:task137 (Gather idx-plane), mem 7106, params 686
**Target tier:** A — closed-form chebyshev rings; output colour COPIES the input
colour so a fixed Conv/route works, but it is a 3-state output (off-grid/black/
colour) so a colour-index route is needed (not pure separable bool).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior: Gather(palette, int32 idx[30,30]) + Less plane | A | 7106 | 686 | 16.039 | 200/200 | baseline |
| 2 | bool ring + Where(one-hot) | - | - | - | 0 | - | off-grid black leak + 9000B else-plane |
| 3 | uint8 idx, 10 separate [30,30] planes | A | 11071 | 685 | 15.63 | - | too many planes |
| 4 | per-axis 1-D idx vecs + dlt-select + offgrid override (4 planes) | A | 5791 | 685 | 16.22 | 200/200 | ok but <+0.3 |
| 5 | + fp16 vectors + fp16 fmod (drop int32 Mod) + 4-D (drop squeeze) | A | 4937 | 685 | 16.366 | 200/200 | beats +0.3 |
| 6 | fold off-grid into 1-D vecs by forcing off-axis to dominate (2 planes) | A | 3197 | 686 | **16.736** | 500/500 | ADOPTED |

## Best achieved
16.736 @ mem 3197 params 686 — beats prior 16.039 by **+0.70**. fresh 500/500.

## Irreducible-floor analysis
Only TWO canvas-sized intermediates remain: `dlt = Less(dr2,dc2)` (900B bool) and
`Lidx = Where(dlt, lc, lr)` (900B uint8). Chebyshev distance is genuinely 2-D so
≥1 canvas plane is unavoidable; the comparison is the second. Remaining mem is the
two fp32 Conv outputs (120B each) + small fp16 1-D vectors. Params (686) are now
the single biggest term — dominated by the two 300-element row/col occupancy Convs
[1,10,1,30]/[1,10,30,1] (ch0 weight 0). Halving them by slicing/reducing channels
instead trades 540 params for 2000-3600B of intermediate plane → strictly worse at
this memory scale, so 686 is the efficient choice.

## OPEN ANGLES (re-attack backlog)
- Collapse the two Convs to one if a single kernel could emit both row- and col-
  occupancy (different orientations block this with a plain Conv); a grouped /
  reshaped contraction might shave ~300 params (~+0.07).
- The 99-sentinel "force the off-grid axis to dominate" trick removed the offgrid
  OR plane AND the Lin plane at once — verify it transfers to other square-grid
  off-canvas-zeroing tasks.

## INSIGHT (transferable)
⭐ For a 3-state per-cell output (off-grid→all-zero / background→ch0 / value→ch_k)
route a uint8 index plane into a FREE BOOL output via Equal(Lidx, arange(10)) with
an OUT-OF-RANGE sentinel (10) for off-grid (matches no channel → all-off). To zero
the off-grid region WITHOUT a dedicated offgrid plane, push the sentinel into the
1-D per-axis index vectors and FORCE the off-grid axis to win a max/select by
setting that axis's distance to a huge value (99) on its 1-D bound mask — collapses
both the offgrid OR-plane and the intermediate select-plane, leaving just the
dominating-axis comparison + the index plane (2 canvas planes total).
