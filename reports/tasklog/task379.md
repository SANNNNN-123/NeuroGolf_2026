# task379 — ecdecbb3

**Rule:** 1-2 full-width horizontal CYAN(8) lines; each column has ≤1 RED(2) dot.
Each dot shoots a ray (painting red along its column) toward the NEAREST line
above AND the NEAREST line below it (a line strictly between blocks the farther
one). Where a ray reaches a line at (L,c): paint the inclusive column segment
[dot..L] red, stamp a 3×3 CYAN box centred at (L,c), then set the box centre RED.
Paint priority: cyan-lines < ray-red < box-cyan < centre-red. `xpose` flips the
whole figure (lines become vertical).
**Current (public):** 13.82 pts, mem ~70k.
**Target tier:** B (closed-form masks; per-cell reconstruction routed into the
free BOOL output) — full Tier S impossible (output colours are fixed but the
geometry is a per-column ray + stamp, not a pure copy/permutation of input cells).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | dual-branch fp32 closed-form | B | 140k | 53 | 13.15 | — | correct, too big |
| 2 | single branch (select inputs by xpose flag) | B | 92.8k | 53 | 13.56 | — | correct |
| 3 | + fp16 whole pipeline | B | 67.2k | 54 | 13.88 | — | +0.06 |
| 4 | + CROP-TO-ACTIVE 20×20 + pad-back | B | 41.7k | 61 | 14.36 | — | +0.54 |
| 5 | + uint8 nested-Where label (no fp32 L planes) | B | 38.5k | 64 | 14.44 | — | |
| 6 | + combined-channel Slice + bool in-grid | B | 29.3k | 68 | 14.71 | — | |
| 7 | + ArgMax dotrow (no rred plane) | B | 28.7k | 68 | 14.73 | — | |
| 8 | + scalar line-rows (no dval/uval full planes) | B | 26.5k | 68 | 14.81 | 200/200 | prior best |
| 9 | PROFILE rewrite: dot/line via no-pad collapse Convs (NO red/cyan full plane) | B | 11385 | 1851 | 15.51 | — | breaks 11.4k floor |
| 10 | + merge count into dot conv via (r+1) offset (6→4 convs) | B | 10865 | 1252 | 15.60 | — | |
| 11 | + per-line SEPARABLE box (row-band ⊗ widened-reach-col), drop fp16 MaxPool | B | 10633 | 1252 | 15.62 | — | no full fp16 plane |
| 12 | + band-pack 2 profiles/conv (val=(dot+1)+64·cyancnt), 4→2 convs | B | 11189 | 654 | 15.62 | — | params 1252→654 |
| 13 | + box rows = {L-1,L+1} ONLY → centre red falls out of line<ray<box (drop 4 planes) | B | 9469 | 653 | **15.78** | 3000/3000 | **BEST — beats deployed 15.66 by +0.12** |

## Best achieved
**15.7775 @ mem 9469 params 653 (total 10122)** — beats deployed kojimar 15.658
(mem 10134 + params 1274 = 11408) by **+0.12 EXACT**. fresh 3000/3000 + isolated 200/200.

## Irreducible-floor analysis
Dominant survivors (all 1600 B = fp32 20×20): 3 entry colour slices (red/cyan/bg,
free input → slice is counted), the fp16 casts of the red/cyan masks needed for
the float ReduceSum (cyan-per-row line detect) / ReduceMax (red-per-col presence),
and the box dilation (bool→fp16 cast + 3×3 MaxPool — MaxPool needs float). The
orientation transpose/select are now uint8 (400 B). ReduceSum/ReduceMax reject
uint8/bool so a colour count needs ≥1 fp16 full plane per axis — that's the
remaining structural cost, plus the unavoidable 3 entry slices.

## KEY BREAKTHROUGHS (this session, 26.5k→9.5k)
1. **PROFILE-not-plane**: never slice red/cyan to a full plane. Recover dot row
   and cyan-line via no-pad collapse Convs (kernel [1,10,30,1]/[1,10,1,30]) that
   emit 1-D [1,1,1,30]/[1,1,30,1] vectors. Kills all the fp32 entry slices.
2. **BAND-PACK 2 profiles per conv**: one collapse-direction conv carries BOTH
   dot position (weight r+1 on red) AND cyan count (weight 64 on cyan) — decode
   by floor/mod 64 (ints<2048 fp16-exact). 4 convs → 2 ⇒ params 1252→654.
3. **SEPARABLE per-line box** kills the fp16 3×3 MaxPool: box = (row-band ⊗
   widened-reach-col), widen the 1-D reach profile with a tiny 1×3 MaxPool, AND
   the per-row band — all bool planes, zero full fp16 plane.
4. ⭐ **BOX ROWS = {L-1, L+1} ONLY (drop the line row L)**: the line row is
   already cyan from lineB and the ray already reds (L, dotcol). With box never
   covering row L, the priority `line(8) < ray(2) < box(8)` reproduces the RED
   box centre for FREE — eliminates the entire centre layer (cen_min/cen_max/bcB
   + 1 compose Where = 4 full planes, ~1.7k). This was the win that crossed the
   floor.

## OPEN ANGLES (untried, total now 10122)
- The ~3.4k of tiny 1-D decode planes (Div/Floor/Slice/Where chains, 120B each)
  could likely shave a few hundred B by selecting the PACKED value first then
  decoding once, instead of decoding both orientations then selecting.
- Ray still costs 4 full bool planes (lt_lo/gt_hi/out_r/rayB); a single
  between-test would help but the empty-range (no-reach) sentinel blocks the
  product trick.

## INSIGHT (transferable)
⭐ A "ray + iterative stop-on-cyan + 3×3 stamp" generator that LOOKS like a flood
is fully CLOSED-FORM: the stop-on-cyan reduces to "reach the NEAREST line in each
direction" → per-column scalar `Ldown=min line>dot`, `Lup=max line<dot`. With
≤2 lines these come from a tiny `Where(lineB, rowidx, ±BIG)` ReduceMin/Max over
the **[1,1,WK,1] line vector** (NO WK×WK candidate plane) plus a 2-level Where
chain on the [1,1,1,WK] dot-row vector. Ray = row-vs-{dot,L} range masks; box =
3×3 MaxPool of the line-intersection mask; compose by nested **uint8** Where
(priority order) → no fp32 colour-value plane. xpose handled by uint8
transpose+select of the masks once (uint8 Where works; bool Where does NOT).
⭐ ArgMax works on uint8 (ReduceSum/ReduceMax do not) — use it for "row index of
the unique marker per column" to avoid a full coord×mask product plane.
