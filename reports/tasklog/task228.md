# task228 — 952a094c

**Rule:** A hollow `bcolor` rectangle (outline ring, interior all black) sits in the fixed 10×10 grid.
At its 4 interior corners are 4 distinct colours colors[0..3] (TL,TR,BL,BR, reading inside the ring).
Output keeps the SAME hollow ring (interior black, interior corner pixels cleared) and EJECTS the 4
colours to the 4 *outer* diagonal corners, each going to the diagonally-OPPOSITE outer corner
(point-reflection through box centre): outer TL=cBR, outer TR=cBL, outer BL=cTR, outer BR=cTL.
**Current:** 16.32 pts, custom:task228, mem 5833, params 79 (was 15.99, mem 8011 — prior custom net).
**Target tier:** B (label-map + final Equal) — output is per-cell deterministic but NOT separable
(4 isolated outer-corner points + a ring break row/col separability).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior net: 1×1 colour Conv → 30×30 fp32 plane (3600B) → slice/label | B | 8011 | 195 | 15.99 | — | baseline |
| 1 | reconstruct from SCALARS; bounds+corner-colours from 2 per-channel occ reductions; no colour plane | B | 7065 | 235 | 16.10 | 200/200 | bug fixed (rank-3 gather broadcast) |
| 2 | slice occ to WORK=10 bool; fp16 corner extraction; Equal-masks; MatMul corner-label | B | 5833 | 79 | 16.32 | 200/200 | ADOPT-WORTHY |

## Best achieved
16.32 @ mem 5833 params 79 — beats prior 15.99 by **+0.33**. Fresh 200/200 (twice, fresh seeds).

## Irreducible-floor analysis
Dominant intermediates = the two per-channel occupancy reductions ReduceMax(input,[3]) [1,10,30,1]
and ReduceMax(input,[2]) [1,10,1,30] (1200B each, fp32 — ORT ReduceMax emits fp32, the 30-extent is
fixed by the input spatial dims so it can't be sliced before the reduce) + the padded label map
[1,1,30,30] uint8 (900B). That 3300B core is the task132 ~16.8 floor. The remaining ~2.5KB is a long
tail of tiny [1,10,1,1]/[1,1,10,1]/[1,1,1,10] vectors and a few 10×10 working planes; squeezing them
further yields <0.05 pts. Both reductions are genuinely needed: row-occ + col-occ jointly disambiguate
each interior corner colour by (top|bot row-band) × (left|right col-band), and also give the box bounds.

## OPEN ANGLES (re-attack backlog)
- Kill ONE of the two 1200B reductions: the 4 corner colours might be recoverable from a single axis if
  the 4 colours' per-row signature is unique without the per-col cross — not obviously true (2 colours
  share each interior row), so the 2-D cross looks required. ~ -1200B → ~16.9 if it works.
- Tier-A separable route: blocked — the 4 outer-corner points are isolated and the ring's row/col-edge
  union is not a single row⊗col product, so a 30×30 carrier (the padded L) is required either way.

## INSIGHT (transferable)
"Box-with-corner-tags" reconstructions are fully SCALAR-driven: box bounds (r0,r1,c0,c1) AND the per-
corner colours both fall out of the SAME two per-channel occupancy reductions [1,10,30,1]+[1,10,1,30] —
Gather the bcolor channel for bounds, Gather the interior-corner row/col indices for per-channel colour
PRESENCE, then a colour sits in corner (rowband × colband) iff present in both — NO 3600B [1,1,30,30]
colour plane is ever built. ⭐ Collapsing N disjoint single-cell stamps (different colours) into the
label map: build it as ONE MatMul of a [W,N_rows] row-selector by an [N_cols,W] colour-bearing
col-selector instead of an N-deep Where chain (here 2 rows × 2 cols via Concat→MatMul: 3 planes → 1).
fp16 `Equal(ramp, scalar)` builds a single-index mask in ONE op (no Sub/Abs/Greater triple).
