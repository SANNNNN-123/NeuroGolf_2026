# task081 — 3aa6fb7a

**Rule:** Several L-trominoes on a fixed 7x7 grid: each is a 2x2 block with exactly 3 cyan (8)
cells and one empty corner (bg 0). Shapes are isolated (generator `has_neighbor` guard). Output:
the 3 cyan cells stay cyan; the empty 4th corner becomes blue (1). Local-exact: a bg cell (r,c)
turns blue iff it is the missing corner of a 2x2 block whose other 3 cells are all cyan (4 corner
roles, at most one matches because shapes never touch).
**Current:** 16.85 pts (public net).
**Target tier:** B (closed-form per-cell predicate routed into the FREE bool output; not pure
copy because blue cells are newly synthesized, so not Tier S).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4 corner-role 3x3 convs (==3) OR, AND bg; L=blue+8·cyan; Equal(arange) | B | 2419 | 67 | 17.18 | 200/200 | ADOPT-CANDIDATE |

## Best achieved
17.18 @ mem 2419 params 67 — beats prior 16.85 by +0.33 (Y). Fresh 200/200.

## Irreducible-floor analysis
Canvas bounded to 7x7 by generator (size=7), so every working plane is tiny. Memory is the four
fp16 corner-conv outputs (~98B each at [1,1,7,7]) + the fp32 cyan slice (196B) + fp16/bool masks +
the uint8 [1,1,30,30] padded label (900B) which dominates. The 30x30 pad to land the one-hot in the
FREE output is the largest single tensor; it is needed to materialize the off-grid-zero target.

## OPEN ANGLES (re-attack backlog)
- Replace the 4 separate corner convs with ONE banded 3x3 conv: weight the 8 neighbors so each
  corner pattern lands in a distinct magnitude band, then a single Equal/range read recovers
  "any complete L" — would drop 3 conv outputs + 3 Equals (~mem -300B, ~+0.1pt). Marginal.
- Pad the uint8 label with a smaller carrier or route blue/cyan via separate channel-Where to
  avoid the 30x30 sentinel pad (the 900B dominant). Likely the biggest remaining lever.

## INSIGHT (transferable)
"Fill the missing corner of an L-tromino / complete-the-2x2" is a fully LOCAL closed-form predicate:
a bg cell is the target iff one of its 4 corner-role 3x3 convs sums to 3 (the 3 present cells of
that block). When the generator isolates shapes, the 4 patterns are mutually exclusive so a plain
OR is exact — no flood-fill, no connectivity wall. Tiny fixed canvas (7x7) keeps every plane cheap.
