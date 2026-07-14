# task115 — 4be741c5

**Rule:** Input is a small grid (height 8..16) tiled left→right by K consecutive
vertical colour BANDS (K = 3 or 4 distinct non-bg colours). Band boundaries are
noisy (the "gap" columns mix the two neighbouring band colours per row), but every
row contains all K colours in the SAME left-to-right order `colors[0..K-1]`. If
`xpose=1` the figure is transposed (bands run top→bottom). OUTPUT = just that colour
sequence: a 1×K row (non-xpose) or a K×1 column (xpose); all other cells (incl
off-grid) are all-zero, NOT background ch0.
**Current (prior public):** 16.68 pts, ArgMax/TopK/ScatterElements reads of two
directional input slices, mem 4042, params 77.
**Target tier:** B — output colours COPY input colours but band ORDER is a
data-dependent rank + orientation is data-dependent, so no fixed Conv/permute (tier S
blocked); routing is separable (rank into a tiny grid → Pad), escaping the 3600B
plane floor — but the floor is set by the two directional input reductions.

## Attempts (this session — re-attack with new levers)
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1..6 | (prior session) centroid-rank, 4×4 routing+Pad | B | 3932 | 100 | 16.70 | 500/500 | prior best |
| 7 | rebuild: 2× ReduceSum profiles, MatMul centroids (scalars), centroid-spread orient, pairwise rank, 4×4 Pad | B | 4044 | 101 | 16.67 | — | baseline |
| 8 | fold present-gate INTO rank (absent→rank+100, kills a [1,10,4,4] And plane) | B | 3884 | 101 | 16.71 | — | |
| 9 | fuse band-position + zero-axis into ONE selectable bgrid → single Equal (kills the And) | B | 3699 | 117 | 16.75 | 200/200 | **best** |

## Best achieved
**16.75 @ mem 3699 params 117** — adopted? N (do-not-adopt). Beats prior 16.68
by **+0.07** → **MARGINAL** (bar is +0.30). ISOLATED fresh 200/200.

## Irreducible-floor analysis (re-confirmed against THIS session's levers)
Dominant intermediates: the TWO fp32 directional input reductions —
`colcount [1,10,1,30]` and `rowcount [1,10,30,1]`, **1200 B each = 2400 B**. This
session tried hard to drop one; it is mathematically required:
  * ⭐ NEW finding: per-channel **counts are TRANSPOSE-INVARIANT** (transpose
    preserves every colour's pixel count), so orientation carries ZERO signal in the
    free `ReduceSum(input,[2,3])` count vector (40 B) — orientation is purely SPATIAL
    (which axis the bands run along) and needs a per-axis profile on BOTH axes.
  * ⭐ NEW finding: orientation IS derivable from the column profile alone
    (avg #colours per occupied column < 2 ⇒ non-xpose, 0/8000) — but that only gives
    ORIENTATION, not the xpose ROW ORDER. The xpose vertical band order collapses
    under any column reduction, so the row profile is still required. Two profiles.
  * fp32 is mandatory for the moment: the count-weighted centroid (the only ordering
    that survives variable band widths — moment-only ranking fails 48% of the time)
    reaches ~14 000 (>2048), outside fp16's integer-exact range, so the moment MatMul
    must run on the fp32 count profile.
  * The min-COORDINATE alternative (first-occupied col/row, values 0..29, fp16-exact,
    order+orientation both 0/8000) would let everything downstream be fp16 — but the
    occupancy it needs (`ReduceMax(input,[2])`) is the SAME fp32 1200 B profile, so it
    saves nothing on the wall and ADDS [1,10,1,30] Where planes (600 B each). Rejected.
  * uint8 reductions rejected by ORT; the input cannot be cast to fp16 (18 000 B).
Everything downstream is already near minimal: centroids/orientation are [1,10,1,1]
scalars (~40 B), rank is a [1,10,1,10] pairwise compare (100 B bool + 200 B fp16 cast
+ 20 B reduce — fp16 is the floor, uint8 ReduceSum is rejected), the whole output is
routed in a [1,10,4,4] uint8 block (one Equal 160 B + one Cast 160 B) then Pad(0) to
30×30. mem+params = 3816 ⇒ 16.75. Theoretical minimum for this structure ≈ 2400 +
~700 (rank+routing+orient) + ~100 params ≈ 3200 ⇒ ~16.9, STILL below the +0.30 bar
(needs mem+params ≤ 3045). The +0.3 win is structurally unreachable.

## OPEN ANGLES (re-attack backlog)
- Eliminate one directional reduction: PROVEN impossible this session — orientation is
  transpose-variant (needs both axes' spatial profiles) while counts are
  transpose-invariant; xpose band order cannot be recovered from a column reduction.
- Tier-S impossible (data-dependent order + orientation). Tier-A blocked (output is a
  rank-permutation, not a fixed row⊗col rectangle).

## INSIGHT (transferable)
⭐ TRANSPOSE-INVARIANCE TEST FOR "IS ORIENTATION FREE?": before assuming a
data-dependent xpose flag needs two full per-axis reductions, ask whether the cheap
signal (per-channel COUNT vector, 40 B) is transpose-invariant. For band/stripe tasks
it IS (counts don't move under transpose) ⇒ orientation needs a SPATIAL per-axis
profile on BOTH axes ⇒ the 2×1200 B fp32-reduction wall is REAL and caps the score
~16.8. (Contrast: tasks whose orientation shows in a count asymmetry are free.)
⭐ FOLD THE PRESENT-GATE AND THE ZERO-AXIS CONSTRAINT INTO SCALARS/CONSTS, not
[1,10,k,k] And planes: (1) absent colours → rank+BIG (a [1,10,1,1] Where) so they
never match a band position 0..k−1 — removes a full present-And block; (2) bake "band
index AT the zero-axis cell, sentinel elsewhere" into the routing grid itself
(`bgrid[1,1,4,4]` selected by orientation), so ONE `Equal(rank, bgrid)` yields the
final mask with no separate zero-axis And. Took 16.67→16.75 by deleting two
[1,10,4,4] (160 B) planes.
⚠️ Two fp32 directional input reductions (one per axis) is a hard ~2400 B wall for any
rule whose answer needs a per-axis weighted profile on BOTH axes when orientation is
transpose-variant. Caps the score ~16.8 — same wall the public net hits. Report
MARGINAL once both profiles are confirmed necessary by the transpose-invariance test.
