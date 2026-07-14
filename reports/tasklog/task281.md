# task281 — b548a754

**Rule:** Input has a rectangular box (1px outer-colour border + inner-colour interior) near an edge plus a single cyan(8) dot offset along the box's axis (within the box's perpendicular span). Output stretches the box so its edge reaches the dot's line: the output rectangle = the bounding box of ALL non-background cells (box ∪ dot), drawn as a 1px outer frame with an inner-colour interior; the cyan dot is removed. xpose/flip applied to both grids — "union-bbox → framed rect" is orientation-independent.
**Current:** 16.01 pts, separable bbox + count-based colours, mem 7315, params 745
**Target tier:** A (separable rect/interior masks routed into the FREE Equal output; colours are tiny [1,10,1,1] arithmetic — no 2-D colour/neighbour plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf 13x13 plane + 3x3 deep-conv inner-colour + triangular masks | A | 17688 | 739 | 15.18 | 200/200 | exact but 6760B multi-ch slice killed it |
| 2 | Conv→colf30 fp32, slice 13x13, fp16 downstream | A | 11462 | 752 | 15.59 | 200/200 | 3600B colf30 dominates |
| 3 | drop 2-D colf entirely; colours via cnt==area; 1-D occ | A | 10421 | 806 | 15.67 | 200/200 | bbox Where planes (rlo/rhi/clo/chi) heavy |
| 4 | colours via MIN/MAX count (ring always > interior), fp16 rc/cc | A | 8291 | 745 | 15.89 | 200/200 | beats +0.3 |
| 5 | Conv occupancy on fp32 rc_f directly (drop fp16 rc/cc planes) | A | 7315 | 745 | 16.01 | 500/500 | ADOPTED |

## Best achieved
16.01 @ mem 7315 params 745 — beats prior 15.54 by **+0.47**. 266/266 stored, 500/500 fresh.

## Irreducible-floor analysis
Dominant intermediates are the two per-channel 1-D spatial reductions `rc_f`/`cc_f`
([1,10,30,1] fp32 = 1200B each). They must be fp32 (ReduceSum of the fp32 input) and
feed the [0,1,…,1] channel-Conv that yields non-bg row/col occupancy. Using the 1-D
[1,10,30,1] shape (not a [1,10,30,30] plane) keeps each at 1200B, beating the
"two fp32 reductions ≈15.8" ceiling. The 900B uint8 30×30 label map (Pad target for the
output shape) is next; Pad rejects bool so the carrier must be uint8 before Equal.

## OPEN ANGLES (re-attack backlog)
- Fuse the two occupancy reductions: a single channel-Conv on input gives [1,1,30,30]
  (3600B) serving both axes — WORSE than 2×1200, so not pursued. A genuine 1-D-only
  occupancy that avoids the 30-length per-channel plane would drop ~1200B → ~16.3.
- The five 338B 13×13 fp16 work planes (rect_f/intr_f/Lcol/L_outer/L_diff) could fold
  into fewer ops (~−1000B → ~16.2) but each is cheap.

## INSIGHT (transferable)
⭐ "stretch a bordered box to a marker dot" = the union bounding box of all non-bg
cells, drawn as a framed rect — orientation-free, no flood-fill. For inner-vs-outer
colour of a box, the OUTER ring count > INNER interior count for ALL box sizes in
[3,5]² (perimeter 2(t+w)−4 vs area (t−2)(w−2)), so inner = MIN-count / outer = MAX-count
present non-cyan colour — pure [1,10,1,1] arithmetic, no bbox-area Where planes and no
2-D colour plane at all. Rect mask = non-strict triangular prefix∧suffix-OR; eroded
interior = the SAME with STRICT triangulars (k<r, k>r) — erodes the run 1 cell each end
for free, no separate erosion Conv.
