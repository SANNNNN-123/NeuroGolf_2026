# task290 — b94a9452 ("cookie/creme" square -> cropped, colour-swapped)

**Rule:** Input grid holds ONE solid square of side `size = thicks[0] + 2*thicks[1]`
(thicks[i] in {1,2} -> size in {3,4,5,6}) at (row,col). The whole square is the
"cookie" colour colors[1]; a CENTRED thicks[0]xthicks[0] inner block (the "creme")
is colour colors[0] (since size = thicks[0] + 2*thicks[1], the inner block is exactly
centred, margin thicks[1] on every side). OUTPUT = that square cropped to the top-left
corner (0,0) of a size x size grid with the two colours SWAPPED: output outer =
colors[0] (input inner / rarer colour), output inner = colors[1] (input outer / more-
frequent colour). Verified exact: everything collapses to four scalars read from the
per-channel pixel COUNTS — no spatial plane needed. cnt[c0]=thicks[0]^2 in {1,4},
cnt[c1]=size^2-thicks[0]^2; size=sqrt(cnt[c0]+cnt[c1]), t0=sqrt(cnt[c0]),
t1=(size-t0)/2.
**Current:** 17.48 pts, custom:task290 (count-driven scalars + separable label-map),
mem 1802, params 49. Prior 15.63 (public ext:kojimar6275).
**Target tier:** B (label-map + final Equal), but the spatial work fully collapses to
4 scalars. Not S: output colours c0,c1 are random per instance, so a fixed Conv/permute
cannot route them to the correct output channel (same blocker as task012). Not pure A:
the c0 channel is a FRAME (square minus inner block) = rectangle MINUS rectangle, not a
single row⊗col product; only the c1 (inner) channel is separable. B label-map is the
highest admissible tier — but the geometry is so over-determined that the only real
intermediate is the 900 B label plane; everything else is scalars.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel ReduceSum counts -> recover c0/c1/size/t0/t1 scalars; separable in-square & centred-inner masks on 8x8; Where->c1/c0; Cast u8; sentinel outside; Pad 30x30; Equal | B | 1802 | 49 | 17.48 | 200/200 + 800/800 bulk + 36/36 all-thick edge | ADOPT-candidate, +1.85 |

## Best achieved
**17.48 @ mem 1802 params 49 — fresh 200/200, 800/800 bulk, 36/36 across every
(thicks[0],thicks[1]) combo at corner/center positions.** Beats prior 15.63 by
**+1.85**. Adopted? **N** (build-only; main adopts via `python -m src.adopt 290`).

## Irreducible-floor analysis
The dominant intermediate is the **900 B uint8 [1,1,30,30] Pad** feeding the free final
Equal — irreducible: the output spans 30x30 and Equal must write every cell; the
sentinel-99 Pad is the only way off-square cells become all-channels-off. Next is the
640 B fp32 `cnt` ([1,10,1,1] is only 40 B, but several [1,10,1,1] Where/compare planes
accumulate) plus a handful of 64 B uint8/bool 8x8 working masks. Notably there is NO
30x30 float colour plane at all (the usual 3600 B gateway): the entire rule is recovered
from a 40 B per-channel COUNT vector, which is the key compaction here.

## OPEN ANGLES (re-attack backlog)
- Drop the 900 B Pad by emitting Equal at 8x8 then bool-Pad the output -> 8x8 bool*10
  channels would be 640 B but the output footprint must still be 30x30 for the FREE
  output; padding the bool output costs ~9000 B (worse). Pad-then-Equal is optimal.
- Trim the [1,10,1,1] scalar-recovery planes (a few hundred bytes) by folding the two
  count->index ReduceMax selects into one — sub-0.1 pt, not worth complexity.
- Tier-S long-shot: blocked — per-instance random colours forbid a fixed colour routing.

## INSIGHT (transferable)
⭐⭐ **When a shape is fully PARAMETRIC (here a centred concentric square), recover its
size/thickness from per-channel pixel COUNTS — `ReduceSum(input, axes=[2,3])` is a 40 B
[1,10,1,1] vector — and rebuild the output from scalars. This avoids the usual 3600 B
30x30 colour-index Conv/Slice gateway entirely.** size^2 = total non-bg cells, inner
side = sqrt(rarer-colour count), and `sqrt` is fp32-exact on the small perfect squares
{1,4,9,16,25,36}.
⭐ **A colour-swap-on-crop is NOT a colour map to detect:** the two output colours ARE
the two input colours, identified purely by frequency (inner/creme is the rarer; outer/
cookie the more frequent). c0=argmin-count, c1=argmax-count among channels>=1, recovered
via `Where(cnt==min,chidx,-1)` + ReduceMax.
⭐ **Centred inner block = (size - t0)/2 margin**, so position needs no spatial detection
— the offset is a closed-form scalar. The data-dependent TRANSLATION (square at (row,col)
-> output at (0,0)) likewise vanishes because the output is built directly in its own
frame from scalars; no Gather/shift required.
