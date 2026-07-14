# task043 — 2281f1f4

**Rule:** Grid is ALWAYS 10x10. Top row (row 0) has gray(5) markers at a set of columns `cols` (0..8); right column (col 9) has gray(5) markers at a set of rows `rows` (1..9). Output = input (gray markers kept) PLUS red(2) at every intersection (r in rows, c in cols). Since rows>=1 and cols<=8, red never coincides with a gray marker.
**Current:** 16.16 pts, public Slice/Mul/Where net (mem heavier), params/mem unrecorded
**Target tier:** A — separable rowmask[r] x colmask[c] routed into the FREE output; no per-cell label plane needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice gray row0 (colmask) + col9 (rowmask), AND broadcast, Pad+Where(red_oh) | A | 2100 | 34 | 17.33 | 200/200 | WIN +1.17 |

## Best achieved
17.33 @ mem 2100 params 34 — adopted? (build-agent does not adopt). Beats prior 16.16? Y (+1.17)

## Irreducible-floor analysis
Dominant intermediate = the [1,1,30,30] padded uint8 red-mask carrier (~900B) plus the
two single-line gray slices and small bool ANDs. The red one-hot lands in the FREE output
via Where, so no [1,10,*,*] plane is materialized. This is the Tier-A separable floor;
going lower (Tier S) would require the output color to be a pure linear/index function of
the input per-cell, but the intersection red is a genuine row-AND-col product, not a copy.

## OPEN ANGLES (re-attack backlog)
- Could drop the Pad by working on a 30x30 mask directly from sliced lines, but the active
  region is 10x10 so Pad on the small canvas is already cheaper than a full-canvas AND.
- Marginal: fold the two single-line slices; they are already minimal (no ReduceMax needed
  since row0/col9 are single lines).

## INSIGHT (transferable)
Crosshair/intersection tasks ("markers on an edge row + edge column -> mark every grid
intersection") are pure Tier-A separable: the two marker lines ARE the row/col masks
directly (single Slice each, no ReduceMax), AND-broadcast to the [1,1,H,W] mask, route the
fixed fill color into the FREE Where output. Confirm the fill never overlaps the markers
(here rows>=1, cols<=8 guarantee disjointness) so a plain Where (no priority chain) suffices.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 10. Flagged `row_basis`/`col_basis` f16 [1,3,30,1]/[1,3,1,30] 180B each carry einsum free-output dims; the input is not an einsum operand axis, so there's no task187-style re-embed carrier. P[10,30]×2 adds +600 params > the 240B saved. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
