# task284 — ARC b7249182

**Rule:** Two seed dots share a line (a row, or a column if the grid is transposed)
at the long-axis coords Cl<Cr, with half=(Cr-Cl+1)/2. Each seed grows a bilateral
"wrench"/cross glyph in its own colour toward the centreline. Left glyph (colour of
Cl seed): horizontal stem on the shared line cols Cl..e0, a 5-tall vertical bar at
e0 (K-2..K+2 on the short axis), and hook cells at e0+1 on the two bar-ends; right
glyph mirrors with e1, Cr where e0=(Cl+Cr-3)/2, e1=(Cl+Cr+3)/2. The whole grid is
optionally transposed (xpose). Off-grid cells stay all-zero (no channel set).
**Current:** 14.9585 pts, custom:task284, mem 22821, params 138
**Target tier:** A — separable rank-1 glyph (sum of 6 rank-1 components) routed into
the free BOOL output; one fp32 colour-index Conv + one fp32 MatMul are the two
heavy planes, both irreducible per the 3600B-plane floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior committed net (2 MatMul planes + per-channel occupancy) | A | 22821 | 138 | 14.96 | n/a | baseline |
| 1 | single combined 6-component MatMul (both glyphs), colour by along-side threshold | A | 23643 | 137 | 14.92 | — | regressed: the side mask was built as an fp32 broadcast Add (3600) |
| 2 | keep side mask in BOOL (Or of two 1-D bools, never fp32) | A | 21664 | 137 | 15.01 | — | +0.05 |
| 3 | replace 6× [1,10,..] occupancy fp32 (7200B) with one colour-index Conv + tiny 1-D profiles; grid extent via ReduceMax(input,[1,3])/[1,2] 120B vectors | A | 20034 | 128 | 15.09 | — | +0.13 |
| 4 | CANONICAL frame (H=cross,W=along): build whole label assuming non-T, then Transpose+Where the finished uint8 label iff isT — kills all factor-orientation mixing (~6×720B) | A | 17664 | 128 | 15.21 | — | +0.25 |
| 5 | shrink cross factor to 3 distinct rows (stem/band5/mid3) instead of 6 | A | 16424 | 119 | 15.29 | 200/200 | adopt-candidate |

## Best achieved
15.286 @ mem 16424 params 119 — beats prior 14.9585 by **+0.328** (≥+0.3 ✓).
Fresh isolated 200/200. adopt-recommend **Y**.

## Irreducible-floor analysis
Two unavoidable fp32 [1,1,30,30] planes dominate (3600B each = 7200B):
- **Conv_0** colour-index plane `Σ_k k·input_k` — the cheapest way to recover the
  two seed colours AND their positions; the alternative (per-channel
  ReduceSum/Max → 6× [1,10,30,1]/[1,10,1,30] fp32) costs 7200B, strictly worse.
- **MatMul_77** the single combined glyph plane (CH[1,1,30,6]@AW[1,1,6,30]); a
  6-component rank-1 sum routed through one MatMul — ORT forces the output to fp32.
Remaining ~5400B is the uint8/bool label chain (Lc0 grid sentinel, Lc glyph fill,
LcT transpose, L orientation-select, glyphM, gridC) — each a 900B [1,1,30,30].
The output Equal is FREE.

## OPEN ANGLES (re-attack backlog)
- The orientation Transpose+Where on the finished uint8 label costs 1800B
  (LcT 900 + L 900). If the glyph MatMul could be made orientation-correct without
  per-factor mixing AND the grid/colour kept consistent, that 1800 could drop —
  but every attempt to mix at the factor level re-introduced ~4320B of [1,1,6,30]
  fp32 factor tensors, so the transpose-the-label trick is the cheaper of the two.
- gridC (900B bool) is the off-grid sentinel = NOT(crossExt⊗alongExt). It is
  separable but a Where needs the materialized condition; folding it via associated
  broadcasts into the final glyph Where might shave ~900B (untried, fiddly).
- The two 3600B planes are the structural floor; ~16.5KB is near the realistic
  minimum for "recover 2 colours + positions + orientation, then stamp a non-rect
  separable glyph". Tier S impossible (output colours copy arbitrary input colours,
  needs a routed plane).

## INSIGHT (transferable)
⭐ For a multi-colour separable stamp, build ONE combined glyph plane via a single
MatMul over ALL components of ALL colours (K = total component count), then colour
each cell by a cheap 1-D ALONG-AXIS side threshold (`along-coord < S/2 ? colL : colR`)
instead of one MatMul-plane per colour — halves the heavy fp32 plane count.
⭐ Handle a global transpose by building the entire uint8 label in a CANONICAL frame
and conditionally transposing the FINISHED label (`Where(isT, L^T, L)`) — this is
far cheaper (one Transpose + one Where = 1800B) than mixing orientation into every
1-D/2-D factor (which re-materializes many [1,1,6,30] fp32 tensors). Pull the grid
extent into the canonical frame by swapping the 1-D row/col extent vectors by isT.
⭐ Any broadcast Add/Mul that lands on [1,1,30,30] is upcast to fp32 (3600B) by ORT
even with fp16 operands — keep boolean selections in BOOL (And/Or, 900B) and never
let an orientation-mix become a 30×30 float Add.
