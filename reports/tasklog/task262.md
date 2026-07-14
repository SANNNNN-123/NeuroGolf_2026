# task262 — a85d4709

**Rule:** 3x3 grid. Each row r has exactly one gray(5) pixel at column cols[r] in {0,1,2}.
Output fills every cell of row r (cols 0..2) with colors[cols[r]] where colors=(2,4,3):
col0->2, col1->4, col2->3. Grid sits at canvas top-left; rest stays background. (Harness
one-hot leaves off-grid cells all-zero; in-grid colours are always {2,3,4} so channel 0 is
never set anywhere — no bg channel needed in the output.)
**Current:** 16.82 pts (public).
**Target tier:** S-ish — no full-canvas plane; per-row scalar broadcast into the FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | gray3 slice -> MatMul row colour idx -> Pad -> Equal(sentinel cvals) -> And(colmask) | S-ish | 468 | 61 | 18.729 | 200/200 | ADOPT |

## Best achieved
18.729 @ mem 468 params 61 — beats prior 16.82 by +1.91.

## Irreducible-floor analysis
Not at floor. Dominant intermediate is rowEq [1,10,30,1] bool = 300B; the rest are tiny
(gray3 36B, rowval3 12B, rowval_full 120B). No [1,1,30,30] plane is ever materialised:
the 10-channel expansion AND the 30-column expansion both land in the FREE bool output
(rowEq broadcasts over cols, colmask broadcasts over channels+rows, And -> [1,10,30,30]).

## OPEN ANGLES
- Could shave rowEq to fp-free by Casting rowval_full to a narrower carrier, but 300B bool
  is already tiny; marginal. Score is dominated by ln of a sub-1k total, diminishing returns.

## INSIGHT (transferable)
⭐ Per-row "fill the whole row with a colour selected by one marker pixel" = a per-row
weighted-count MatMul (gray3[1,1,K,K] @ wcol[1,1,K,1] -> rowval[1,1,K,1]) giving the colour
index per row, then route BOTH the channel expansion and the column fill into the FREE bool
output via And(Equal(rowval_full,cvals)[1,10,H,1], colmask[1,1,1,W]). Use a CHANNEL-0 SENTINEL
in the compare values (cvals[0]=-1) so the background channel never matches off-grid/empty
rows — this is what lets a pure And replace a Where(bg) and keeps off-grid all-false to match
the harness one-hot (which leaves off-grid cells all-zero, NOT ch0=1).
