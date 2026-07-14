# task180 — 75b8110e

**Rule:** Input is an 8x8 grid (2*size, size=4) split into four 4x4 quadrants by
(row_offset,col_offset): TL=colour 4, TR=colour 5, BL=colour 6, BR=colour 9.
Each quadrant only ever holds its own colour or background 0. The 4x4 output
overlays the four quadrants; the generator paints in order [0,3,2,1] so a later
painter wins, giving per-cell priority TR(5) > BL(6) > BR(9) > TL(4) > 0:
output[r][c] = colour of the highest-priority quadrant with a pixel at (r,c).
**Current:** 17.74 pts, presence-slice label-map + final Equal, mem 1284, params 136
**Target tier:** B (per-cell deterministic 4-way overlay) — colour per cell is a
priority pick among 4 spatially-disjoint fixed-colour quadrant presences; not a
single linear/permutation map (S) nor row⊗col separable (A).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 1x1 Conv colf[1,1,30,30] -> 4 quadrant slices -> Where chain -> Pad30 -> Equal | B | 4948 | 64 | 16.48 | 200/200 | works but Conv forces a 3600B fp32 30x30 colour plane |
| 2 | slice 4 quadrant PRESENCE masks straight from FREE input (own colour channel) -> Where chain w/ constant colour fills -> Pad30 -> Equal | B | 1284 | 136 | 17.74 | 200/200 | WIN — no Conv, no 30x30 colour plane |

## Best achieved
17.74 @ mem 1284 params 136 — adopted? N (orchestrator gates). Beats prior ~16.08? Y (+1.66). Beats P=17.0? Y (+0.74).

## Irreducible-floor analysis
Dominant intermediate is the padded label map Lp uint8 [1,1,30,30] = 900 B,
irreducible because it must be 30x30 to broadcast against the 10 colour channels
in the final Equal that writes the FREE bool output. The four presence slices are
fp32 [1,1,4,4] = 64 B each (256 B total); the Greater/Where chain is bool/uint8
[1,1,4,4] = 16 B each. No [1,1,30,30] colour plane exists because each quadrant's
colour is FIXED — presence alone suffices, so the colour-index Conv was deleted.

## OPEN ANGLES (re-attack backlog)
- The 256 B of fp32 presence slices could in principle be cut by Casting the
  combined presence to bool earlier, but slices already feed Greater directly;
  marginal (<0.1 pt). Not worth chasing past 17.74.
- Tier A is blocked: the overlay is a priority pick across 4 disjoint 4x4 regions,
  not a row⊗col outer product, so it cannot collapse to rowcond⊗colcond.

## INSIGHT (transferable)
When each spatial region (quadrant/band) carries a single FIXED colour, do NOT
recover a colour index with a 1x1 Conv (that materialises a 3600B fp32 30x30
plane). Slice the region's OWN colour channel straight from the FREE input to get
a tiny presence mask, then map presence -> constant colour fill via Where. This
turned task180 from 16.48 -> 17.74 by deleting the colour-index plane entirely.
A fixed-colour-per-region overlay is a priority chain of Where(presence, colour_k,
prev) on small region tensors, with the only large tensor being the 30x30 padded
label map needed for the final Equal broadcast.
