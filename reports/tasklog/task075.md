# task075 — 363442ee

**Rule:** Grid 9x13 (size=3). Left cols 0..2 rows 0..2 = a 3x3 colour TEMPLATE P;
col 3 = full-height gray(5) separator; cols 4..12 = a 3x3 grid of 3x3 blocks.
Each block (R,C) is marked in the INPUT by a single blue(1) pixel at [3R+1, 3C+5].
OUTPUT keeps the template + separator and copies P into every MARKED block
(unmarked blocks stay background).
**Current:** 15.17 pts (public net), mem unknown
**Target tier:** S/A — every output cell is a spatial COPY of an input cell in cols 0..3, so a single data-dependent Gather suffices (no [1,10,30,30] plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | active 9x13 flat gather, fp16 | S | 16380 | 377 | 15.27 | — | only +0.10 |
| 2 | source = input cols 0..3 only (36 cells), fp16 | S | 10368 | 387 | 15.72 | — | +0.55 |
| 3 | + 2-D gather index [9,13] (drop reshape), int32 idx | S | 7560 | 384 | 16.02 | 200/200 | ADOPT |

## Best achieved
16.02 @ mem 7560 params 384 — adopt? Y. Beats prior 15.17 by +0.85.

## Irreducible-floor analysis
Dominant intermediate = the output-side gather `outact` [1,10,9,13] fp16 = 2340B
(inherent: 117 active output cells x 10 channels, halved by fp16). Next: entry
slice `srcf` [1,10,9,4] fp32 = 1440B (the one mandatory fp32 plane; Slice preserves
input fp32 dtype). Everything else is the cols-0..3 source (720B), the blue marker
slice (~470B), and tiny [117] index tensors. No [1,10,30,30] plane exists anywhere.

## OPEN ANGLES (re-attack backlog)
- bool source for the gather (Cast srcf->BOOL, gather->bool 1170B) — but Pad rejects
  bool, so a Cast(bool->fp16) plane reappears; net wash. Skipped.
- drop the separate fp16 blue cast (`blueh`) by doing marker arithmetic in fp32 and
  casting once — saves ~234B, marginal.
- could fold the gray separator + template into a still-smaller 9x3 source if the
  gray col is reconstructed by a const, shrinking the entry slice ~25%.

## INSIGHT (transferable)
⭐ When EVERY output cell is a copy of an input cell from a small sub-region, the
whole task collapses to ONE data-dependent flat Gather into the FREE padded output.
Shrink the gather SOURCE to the minimal column band that contains all copied content
(here cols 0..3 — template + separator + a background sentinel), not the full active
grid: it scales every downstream plane by the band width. Fold data-dependent masking
into the index itself (`base + marker*coef`, marker gathered from the relevant
channel), pointing masked cells at a guaranteed in-grid background cell — no separate
mask plane. A 2-D Gather index [H,W] on axis=2 yields [1,10,H,W] in one op, killing
the flat->2D reshape plane.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 13 (in/out 9×13). Flagged `scalar30` uint8 [30,30] 900B is the Equal index plane for the free output; Equal is nonlinear so it can't move into the einsum. A 9×13 one-hot + Pad = 1170B > 900B. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
