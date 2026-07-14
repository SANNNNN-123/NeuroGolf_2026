# task059 — 29623171

**Rule:** 11x11 "hollywood squares" grid = a 3x3 array of 3x3 mini-cells separated by a gray (5)
frame at rows/cols 3 and 7. Each mini-cell holds 0..max coloured pixels in a single per-instance
colour `c` (c != gray). OUTPUT keeps the gray frame; every mini-cell whose coloured-pixel COUNT
equals the maximum count over the 9 cells (ties allowed) is filled SOLID with `c`; all other cells
become background 0.
**Current:** 15.08 pts, ext:kojimar6275, mem 17840, params 2547
**Target tier:** B (per-cell colour-index label routed into the free bool output) — entry needs one
fp32 colour/occupancy plane (10->1 reduction floor); the rest is closed-form & separable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf conv -> 9x9 compact gather, 3x3 block counts, Equal(cnt,max) ties, label+Pad+Equal | B | 8093 | 293 | 15.97 | 200/200 | WIN +0.89 |
| 2 | same, cast colf->fp16 for downstream small planes | B | 8323 | 293 | 15.94 | n/a | WORSE (dup entry plane) |

## Best achieved
15.97 @ mem 8093 params 293 — adopt-recommend Y. Beats prior 15.08? Y (+0.89).

## Irreducible-floor analysis
Dominant intermediates: colf [1,1,30,30] fp32 = 3600B (the 10->1 colour/occupancy reduction — per
FLOOR_RESEARCH this entry plane cannot drop below fp32 3600B via dtype tricks), cr [1,1,9,30] fp32 =
1080B (byproduct of the per-axis interior Gather), and the padded label L [1,1,30,30] uint8 = 900B.
The fp16 cast lever (task377) does NOT apply here: casting colf->fp16 after the conv DUPLICATES the
entry plane (fp32 colf32 3600 + fp16 colf 1800 both counted), and the downstream savings (9x9/3x3
planes are already tiny) can't offset it — mem went 8093->8323. Removing the colf plane is impossible
because both the per-cell occupancy count AND the colour scalar require the 10-channel reduction.

## OPEN ANGLES (re-attack backlog)
- Eliminate the `cr` 1080B intermediate by reshaping colf to a [1,1,3,3,3,3]-style block layout
  directly via Slice+Reshape rather than the two-axis Gather (the frame rows/cols are at fixed
  positions, so a Slice that drops them in one Reshape might avoid the [1,1,9,30] byproduct).
- Remove the uint8 L pad plane (900B) by building the bool output via an associated separable
  Where/And on the 11x11 region — blocked by Pad rejecting bool, so would need a 30x30 const carrier.
- These would cut ~900-1080B -> est ~16.2; the colf 3600B entry is the hard floor (tier-B ceiling).

## INSIGHT (transferable)
Same hollywood-squares grid family as task011 — the gather-interior-{0,1,2,4,5,6,8,9,10}-to-9x9-compact
idiom packs the 9 mini-cells gap-free so a [1,3,3,3,3] reshape + ReduceSum(axes=[2,4]) gives the 9
block counts with no flood-fill. "Fill the max-count block(s), ties allowed" = Equal(cnt, ReduceMax(cnt))
(NOT ArgMax/Less — Equal keeps ties for free, unlike task011's unique `count<5`). Single fixed fill
colour recovers as a scalar ReduceMax over the frame-free interior (no gray there), so the output is a
single colour-index label L = frame?5 : (selected[block]?c:0) routed through Pad(sentinel 10)+Equal(arange)
into the free bool output. CONFIRMED: the post-conv fp16-cast lever backfires when the only big plane is
the entry colour plane itself (the cast adds a second full plane instead of shrinking downstream work).
