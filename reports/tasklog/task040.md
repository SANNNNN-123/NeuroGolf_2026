# task040 — 2204b7a8

**Rule:** 10x10 grid. Two non-green colours c0=colors[0], c1=colors[1]. Non-transposed:
col 0 = c0 (all rows), col 9 = c1 (all rows); green(3) markers in cols 1..8 each become
c0 if c<5 else c1. Transposed (xpose=1): grid transposed, so row 0 = c0, row 9 = c1, green
at (r,c) -> c0 if r<5 else c1. Borders unchanged; only green cells are recoloured. The
replacement colour is a COPY of a border cell on the same line, so no colour DETECTION is
needed — only orientation (xpose) + per-cell border-copy plane.
**Current:** 16.03 pts (prior), method label-map
**Target tier:** B — colour index must materialise once (3600B Conv entry floor); output colours copy arbitrary input colours so slice+Where colour routing isn't applicable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | V label-map; tgtA/tgtB candidates; green via slice+Greater | B | 6055 | 73 | 16.28 | 200/200 | marginal (+0.25) |
| 2 | green via Equal(V,3) (drop 400B fp32 green slice) | B | 5655 | 65 | 16.35 | 200/200 | beats +0.3 |
| 3 | robust orientation: row0 AND row9 both full (vs row0 only) | B | 5715 | 64 | 16.34 | 500/500 | ADOPTED |

## Best achieved
16.338 @ mem 5715 params 64 — adopted? Y. Beats prior 16.03? Y (+0.308).

## Irreducible-floor analysis
Dominant intermediate is the colour-index Conv output Vbig [1,1,30,30] fp32 = 3600B — the
mandatory 10->1 entry reduction over the 30x30 input (cannot go below fp32 3600B per
FLOOR_RESEARCH). Next is the Pad output L [1,1,30,30] uint8 = 900B (output must be 30x30).
Everything else is tiny 10x10 / 1-D planes on the size-10 active canvas.

## OPEN ANGLES (re-attack backlog)
- Avoid the 900B uint8 Pad by building the label on a separable row⊗col carrier routed
  straight into the free BOOL output — but L10 is a genuine per-cell label (green positions
  are arbitrary), not separable, so this likely doesn't apply here.
- Slicing input to 10x10 before the Conv costs 4000B (10ch×10×10 fp32) > 3600B, so the Conv
  entry stays the floor.

## INSIGHT (transferable)
⭐ When a colour-index plane V already exists, detect a specific colour's cells with
`Equal(V, k_uint8)` (bool 100B) instead of slicing that channel from the fp32 input
(`Slice→Greater`, 400B fp32 plane) — saved 400B / +0.07 pts here. And for a transpose-or-not
orientation flag, require BOTH border lines full (row0 AND row9), not one — a single full
line can be faked by enough markers, but faking two needs more markers than the generator
ever places, making the discriminator bulletproof under fresh generalization.
