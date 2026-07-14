# task141 — 623ea044

**Rule:** Input is a `size x size` grid (odd `size` in 5..21, top-left in the
30x30 canvas) holding exactly ONE coloured pixel at `(row,col)` of colour
`color`; rest background. Output draws `color` on every in-grid cell of either
45-degree diagonal through the pixel: `r+c == row+col` OR `r-c == row-col` (a
full X clipped to the grid). Off-grid cells stay all-zero.
**Current:** 16.04 pts (ext:wguesdon6304), mem 7668, params 89.
**Target tier:** B (label/predicate routed into FREE output). Not A: a single
45-degree diagonal is `r+c==a`, which couples r and c -> NOT a row-cond x
col-cond separable product. Not S: output colour copies a data-dependent input
colour but the cell set is two data-dependent diagonals, not a fixed permutation.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full 3600B ch0 slice for position, 3 bool diag planes | B | 8060 | 167 | 15.98 | — | works, mem high |
| 2 | reduce-one-axis-first (rs/cs [1,10,30,1]) for ch0 profiles | B | 6612 | 90 | 16.19 | — | marginal +0.15 |
| 3 | **fg/extent profiles via two 1x30 / 30x1 Convs (ch0 weight 0)** | B | **3972** | **674** | **16.56** | **200/200** | WIN +0.51 |

## Best achieved
**16.56 @ mem 3972 params 674 (sum 4646) — fresh 200/200.** Beats prior 16.04 by
**+0.51**. Adopted? N (build-only).

## Irreducible-floor analysis
mem 3972 dominated by the THREE 30x30 bool planes `onA`, `onB`, `diag` (900 B
each = 2700 B). These are irreducible: a single 45-degree diagonal is not
row/col separable, so each diagonal needs its own full-canvas Equal, and the
union needs the Or. The diag plane MUST be 30x30 to broadcast against
`input[1,10,30,30]` in the FREE output Where (cropping to 21x21 then padding
adds a plane, net worse). Everything else (position, colour, extent) is a 120 B
or smaller 1-D profile / scalar. params 674 are mostly the two Conv kernels
(300 elems each, in_channels forced to 10).

## OPEN ANGLES (re-attack backlog)
- Collapse `onA`,`onB`,`diag` (2700 B) toward 1800: only if a single Equal could
  express membership in {aCol[c], bCol[c]} per column without a [.,2,.] blow-up —
  no clean form found.
- Trim the two 300-elem Conv kernels: blocked, Conv in_channels must equal the
  input's 10 even though only ch0 vs ch1..9 matters (can't pre-slice channels
  without a full plane).

## INSIGHT (transferable)
⭐⭐ **A full-row / full-column SUM PROFILE of a channel-subset is a single
no-pad Conv, NOT a [1,10,H,1] reduce.** `Conv(input[1,10,30,30],
W[1,10,1,30])` with the kernel spanning the whole row and weighting only the
wanted channels (ch0=0 for foreground, all-ones for occupancy) outputs
`[1,1,30,1]` directly (120 B) — it folds "drop a channel + collapse one spatial
axis" into ONE op and skips the 1200 B `[1,10,30,1]` per-channel intermediate
that `ReduceSum(input,[3])` forces. Here it cut mem 6612->3972 (the pixel
row/col foreground indicators and the grid-extent profiles). Cost: a 300-elem
kernel per axis (params), far cheaper than the saved 1200 B of mem.
⭐ **An apparent "draw the X through a point" detection is closed-form tier-B:**
recover `a=row+col`, `b=row-col` as scalars from the fg row/col profiles, FOLD
the in-grid clip into the 1-D ramps (`Rclip`=row index in-grid / sentinel
off-grid; `aCol`=a-Cramp in-grid / different sentinel off-grid) so each diagonal
`Equal(Rclip[1,1,30,1], aCol[1,1,1,30])` is auto grid-clipped with NO separate
ingrid plane, then `Where(diag, color_onehot, input)` routes into the FREE fp32
output (input itself supplies in-grid bg + off-grid zeros).
