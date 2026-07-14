# task348 — db3e9e38

**Rule:** INPUT is a single vertical ORANGE(7) line at column `col`, rows 0..length-1, on a width×height grid (5..10, top-left anchored). OUTPUT draws a triangular pyramid: cell (r,c) colored iff `r + |c-col| < length`; colour = ORANGE(7) if (c-col) even (same parity as col), else CYAN(8). Rest stays background; off-grid stays all-zero.
**Current:** 16.12 pts (prior public net)
**Target tier:** A (closed-form, one 2-D plane) — the pyramid mask `r+|c-col|<length` is the single genuine 2-D tensor; everything else (col, length, width, parity, colour) is 1-D vectors/scalars.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | scalar (col,length,width) -> thresh[c] vector -> single `colored` plane -> Where(colored, colorOH[1,10,1,30], input) into FREE output | A | 4626 | 85 | 16.54 | 200/200 | ADOPT |

## Best achieved
16.54 @ mem 4626 params 85 — beats prior 16.12 by +0.42. fresh 200/200.

## Irreducible-floor analysis
Dominant intermediates: `colored` [1,1,30,30] bool (the pyramid mask, ~900B inferred) and `colsum_all` [1,10,1,30] fp32 (1200B, the row-reduction used to read length/col from channel 7). The `colored` plane is the one true 2-D object (coupled r+|c-col|, not separable) so it cannot be removed. The 10-ch expansion is FREE: Where's else-branch = input already carries correct in-grid background and off-grid zeros, and the per-column colour one-hot is only [1,10,1,30].

## OPEN ANGLES (re-attack backlog)
- `colsum_all` 1200B could in principle be avoided if length/col were read from a cheaper signal, but ReduceSum(input,axes=2)→[1,10,1,30] is already the minimal full-input reduction; slicing channel 7 first costs 3600B. Net not worth it.

## INSIGHT (transferable)
A row/col-COUPLED region mask `r + |c-col| < length` (pyramid/triangle/diamond) is NOT separable, but it is ONE plane via `Less(rowramp[1,1,30,1], thresh[1,1,1,30])` after folding all per-column geometry (|c-col|, length, in-grid width gate) into the tiny thresh VECTOR. Then a per-COLUMN colour one-hot `[1,10,1,30]` (parity-dependent) as the Where value-branch with input as the else-branch routes the full 10-ch expansion into the FREE output with zero extra full planes — off-grid and background fall out of the unchanged input automatically.
