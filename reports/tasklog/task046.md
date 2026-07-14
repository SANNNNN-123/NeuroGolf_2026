# task46 — 234bbc79 (RE-PROBE, wall re-confirmed)

**Rule:** Height-3 grids (height ALWAYS 3; width in 8-20 → out 6-16). Generator builds a
horizontal "snake" of 3-4 colored segments (each width 2-4) that turn vertically while
advancing column-by-column (snake-col c = 0,1,2,…). INPUT places segment `idx` shifted
RIGHT by `idx` columns (creating all-zero separator columns) AND shifted vertically by a
per-segment data-dependent offset `randint(-min(srows), height-max(srows)-1)` (offset[0]=0);
junction pixels (where adjacent path cells belong to different segments) are recolored gray(5).
OUTPUT re-packs segments contiguously (separators removed), undoes each segment's vertical
offset to the canonical snake-row, and restores the true segment color (output has ZERO grays).

**Current:** 14.64 pts, mem+params ≈ 31506. Beating +0.3 needs ≤ ~23300 (~26% cut).
**Target tier:** none — re-confirmed BAIL-class wall.

## Attempts (this re-probe)
| # | angle | result |
|---|---|---|
| 1 | pure column-compaction (remove all-zero cols, no row change) | 0/3000 — rows genuinely roll per segment |
| 2 | greedy per-column vertical shift by 4-connectivity (height-3, shift∈{-2..2}) | 17%; smarter min-|s|+touch variant 53% — offset NOT locally recoverable |
| 3 | structural: gray-recolor + segment-color propagation | requires segment identity = traced connected turning path through gray junctions |

## Irreducible-floor analysis
Three independent, non-collapsible data dependencies, all confirmed empirically this pass:
1. **Per-segment vertical roll is global, not local.** The shift that maps input row→canonical
   output row is constant-within-segment but a per-segment `randint`. Greedy 4-connectivity
   reconstruction (the only local signal, and cheap on a height-3 canvas) tops out at 53% on
   POSITIONS ALONE — the offset can only be fixed by integrating segment identity across the
   whole path. This is a per-segment data-dependent vertical GatherND.
2. **Segmentation needs path tracing.** Segments are delimited by gray(5) junction pixels that
   co-occur with content in the same/adjacent columns along a TURNING path; #empty-cols ≠
   #segments, so no column-profile or prefix-OR splits the snake. Recovering segment id is a
   connected-component trace along a snake (banned: Loop/Scan/NonZero; not a separable row⊗col,
   not a prefix/suffix-OR, not a count→fixed-pattern).
3. **Gray recoloring depends on (2).** Output has 0 grays; each gray must be repainted with its
   segment's color, which is only known after segmentation.
The canvas is ALREADY small (3×≤20), so CROP-TO-ACTIVE / small-canvas / fp16 plane levers give
nothing — the existing ≈31.5k net is near the structural floor, and every cheap collapse
(compaction Gather, separable masks, count-pattern) leaves the roll+segmentation untouched.

## Best achieved
None built. Re-probe did NOT break the wall.

## OPEN ANGLES (genuinely exhausted)
- Compaction alone is cheap (drop zero columns), but it is <1/3 of the transform and cannot stand
  alone. No remaining reformulation avoids the path-traced per-segment offset+color.

## INSIGHT (transferable)
⭐ Not every blank/short canvas is a re-probe win: when the canvas is ALREADY small (height fixed,
width ≤20) the dtype/CROP levers are inert and the verdict rests purely on STRUCTURE. The tell for
a true wall here = "per-segment data-dependent vertical roll + segmentation that needs tracing a
connected TURNING path through junction pixels" — greedy local connectivity caps at ~53% on
positions, proving the offset is global. Re-confirmed INFEASIBLE (the prior verdict held; this was
NOT a false positive).
