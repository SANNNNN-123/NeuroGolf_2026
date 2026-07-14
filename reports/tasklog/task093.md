# task093 — 4093f84a

**Rule:** size-14 grid with a solid GRAY(5) horizontal "horizon" band (thickness 2-5) spanning the full
width; scattered single coloured pixels (one non-gray colour) sit off the band. Each coloured pixel falls
toward the band along the axis perpendicular to it and STACKS contiguously against the band edge — i.e. per
line perpendicular to the band, count coloured pixels on each side and stack that many GRAY cells against the
band on that side. `flip` (swaps the two sides — symmetric, a no-op for the rule) and `xpose` (rotates the
band horizontal<->vertical) may be applied. Output is always GRAY: band + stacked cells -> 5, else 0.
**Current:** 15.354 pts, ext:kojimar6275, params 58
**Target tier:** A (closed-form separable broadcast; the per-side stack is a per-line distance vs per-line
count comparison — no flood-fill, no scan).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 two-branch (h/v) + fp Where select | A | 21209 | 72 | 15.03 | 200/200 | working, below P |
| 2 | fp16 downstream + bool select (And/Or) + uint8 Where output | A | 14592 | 73 | 15.41 | 200/200 | improved |
| 3 | fold dist<=count into direct Less (kill 4 fp16 Sub planes) | A | 13136 | 73 | 15.51 | 200/200 | improved |
| 4 | per-line counts via masked-sum MatMul (kill 4 fp16 product planes) | A | 11568 | 101 | 15.64 | 200/200 | improved |
| 5 | orientation/band detect via ReduceMin(V) (kill fp16 G plane) | A | 11176 | 100 | **15.67** | 500/500 | BEST |

## Attempts (RE-GOLF session 2026-06-19, occupancy-only + equivariance)
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 6 | orientation-equivariance: canonicalise V via transpose, solve 1 branch | A | 10247 | 72 | 15.76 | — | kept |
| 7 | fp16 Equal gray mask (1 op); occupancy-direct count (no colored mask) | A | 9463 | 72 | 15.84 | — | kept |
| 8 | CONTIGUOUS-SPAN: gray = r0-na <= r <= r1+nb (above+band+below is one run) | A | 8651 | 71 | 15.93 | — | kept |
| 9 | occupancy via Min(Vh,1) (1 fp16 op, no bool occ plane) | A | 8455 | 72 | 15.95 | — | kept |
| 10 | OCCUPANCY-ONLY: drop value plane entirely, canonicalise occupancy, band=fullocc-row | A | 8259 | 72 | 15.97 | — | kept |
| 11 | orientation via ReduceMax(rowOccV) (drop rowFull cast chain) | A | 8217 | 72 | **15.977** | 200/200 | **BEST** |

## Best achieved
15.977 @ mem 8217 params 72 — adopted? N (build-agent does not adopt). Beats deployed 15.670 by **+0.307**.
(Prior session best was 15.670 @ 11176; this session re-golfed it.)

### RE-GOLF floor analysis
Same 3 structural floors (3600 entry Conv + 784 fp32 crop + 900 uint8 output carrier = 5284B). The win came
from eliminating ~3000B of working planes: (a) OCCUPANCY-ONLY — the gray value is irrelevant, so the whole
pipeline runs on `(V>0.5)` presence (band = the only fully-occupied INPUT line; counts = occupied pixels),
killing every value-comparison mask and letting band detection reuse the occupancy reduction (rowSum==CW);
(b) CONTIGUOUS-SPAN — above-stack ∪ band ∪ below-stack is ONE contiguous run `lo<=r<=hi` (lo=r0-na, hi=r1+nb),
collapsing the 7-plane band/above/below OR-chain into 2 compares + 1 And; (c) single transpose-canonicalised
branch (task341 equivariance) instead of the two h/v branches. Remaining working set: 3 fp16 occupancy planes
(occ, occ^T, C) + 7 bool/uint8 14x14 planes + tiny per-line vectors. ⚠️ Two-branch and bool-select
canonicalisations both measured equal-or-worse — the orientation transpose tax (~1764B) resists further cuts.

---
### (prior session below)

## Best achieved (prior session)
15.670 @ mem 11176 params 100 — adopted? N (build-agent does not adopt). Beats prior 15.354? Y (+0.316).

## Irreducible-floor analysis
Dominant intermediates: V32 = the value-plane Conv output [1,1,30,30] fp32 = 3600 B (the irreducible 10->1
colour-index entry floor); L = padded uint8 label [1,1,30,30] = 900 B (the 30x30 output carrier into the
free Equal); Vc = fp32 14x14 crop before the fp16 cast = 784 B. Everything else is fp16 [1,1,14,14]=392 B
(V, C) or bool 196 B planes. The 3600 B entry and 900 B carrier are the two structural floors; the 784 B Vc
is a fp32 crop that can only be removed by casting before crop (which makes a 1800 B fp16 30x30 plane — net
worse), so it stays.

## OPEN ANGLES (re-attack backlog)
- Kill the 784 B Vc fp32 crop: if a fp16 30x30 plane could be avoided (e.g. derive gray/coloured masks on the
  30x30 fp32 directly as bool then crop the bool), Vc disappears — but bool 30x30 = 900 B each (two masks =
  worse). No clean win found.
- Unify h/v branches via a single transpose-canonicalised branch (Where(horiz,V,V^T)) to drop the duplicated
  vertical bool plumbing — small bool savings (~5x196 B), modest score bump, not worth the added Transpose
  planes at this margin.

## INSIGHT (transferable)
"pixels fall toward a full-line barrier and stack" is closed-form tier-A, NOT a gravity/flood BAIL: per line
perpendicular to the barrier, the stacked-cell mask is `1 <= dist_from_edge <= count_on_that_side`, a pure
broadcast of a per-line DISTANCE vector [1,1,K,1] against a per-line COUNT vector [1,1,1,K] — the count comes
from a **masked-sum MatMul** (contract the perpendicular axis: `MatMul(rowIndicator[1,1,1,K], C[1,1,K,K])`)
which produces the per-line count directly with NO 2-D masked-product plane. ⭐ Fold `dist <= count` into a
single `Less(dist, count+0.5)` (precompute count+0.5 as a tiny vector) so the comparison's [1,1,K,K] output
is born as a bool 196 B plane, never a fp Sub plane. ⭐ Detect a FULL-of-one-colour line cheaply by reusing
the value plane: `ReduceMin(V, perp-axis) == colour` (a non-full line has a bg(0) cell -> min<colour),
removing a dedicated fp16 gray-count plane entirely. The flip symmetry of a two-sided barrier makes flip a
free no-op; only the horizontal/vertical orientation needs a scalar branch select done with bool And/Or
(ORT Where is NOT implemented for bool, so select via `(horiz AND h) OR (NOT horiz AND v)`).
