# task357 — e179c5f4 ("bounce" zig-zag ray)

**Rule:** Grid is HEIGHT=10 by WIDTH=W (W in [2,10]) anchored top-left of the
30x30 canvas. INPUT = all BLACK(0) with a single BLUE(1) pixel at (row 9, col 0).
OUTPUT = all CYAN(8) background with a BLUE(1) "bounce" path: starting at (9,0)
the path moves UP one row each step while its column bounces between 0 and W-1
(triangle wave). Off-grid cells (row>=10 or col>=W) are all-zero. The path column
for row r is `pc(r) = (W-1) - |((9-r) mod 2*(W-1)) - (W-1)|`. The ENTIRE output is
a function of W ONLY (the input pixel is always fixed at (9,0)).
**Current (prior public):** 16.21 pts.
**Target tier:** B (single-channel label plane + final Equal). Not A: the path
`c==pc(r)` couples r and c through the triangle wave (not a row-cond x col-cond
outer product). Not detection: fully closed-form from one recovered scalar W.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-row pc vector (Mod+Abs), 4 full planes (path/ingrid/lab_i/L) | B | 4150 | 78 | 16.65 | 200/200 | win +0.44 |
| 2 | fold cyan/sentinel into tiny [1,1,1,30] colcyan base -> 3 full planes | B | **3280** | **78** | **16.88** | **200/200** | WIN +0.67 |

## Best achieved
**16.88 @ mem 3280 params 78 — fresh 200/200 + exhaustive W=2..10 all exact.**
Beats prior 16.21 by **+0.67**. Adopted? N (build-only).

## Irreducible-floor analysis
Three [1,1,30,30] planes dominate (~2700 of 3280): `path` (bool 900, the
Equal(col,pc) broadcast that couples r&c — irreducible since the triangle wave
is not separable), `csent` (uint8 900, cyan/sentinel base gated to rows<10), `L`
(uint8 900, the chained-Where label feeding the FREE final Equal). The per-row
`pc` work (Mod/Abs/Sub) lives on 30-elem [1,1,30,1] vectors (tiny). W is a scalar.

## OPEN ANGLES (re-attack backlog)
- Drop `csent` (-900B -> ~16.96): would need cyan(8) in rows>=10 suppressed
  without a row-gated full plane. A per-row +100 offset Add achieves the gate but
  is itself a full-plane Add (same plane count). No cheaper route found.
- Only 9 distinct outputs (W=2..10). A Gather of 9 precomputed 30x30 patterns by
  scalar W is an alternative but the pattern table (9*900 init elems) and the
  Gather output plane don't beat the 3-plane closed form.

## INSIGHT (transferable)
A "bounce / zig-zag ray" output that is a pure function of ONE recovered scalar
(here W = #occupied input columns) is closed-form tier-B, NOT a detection wall.
The bounce path column is a per-ROW vector `pc(r)=(W-1)-|((H-1-r) mod 2(W-1))-(W-1)|`
built with fp16 Mod+Abs on [1,1,30,1] (30-elem, exact) — only the final `c==pc`
comparison need ever be a full plane. Folding the cyan/sentinel background into a
tiny [1,1,1,30] colcyan base (then one row-gated Where) keeps the whole net at
just 3 full [1,1,30,30] planes.
