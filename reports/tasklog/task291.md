# task291 — b9b7f026

**Rule:** An H×W grid (12..18 each) holds 4..7 disjoint solid axis-aligned
rectangles, each a distinct colour. The FIRST box (colour colors[0]) has a
rectangular "donut hole" of black cells carved out of its STRICT interior
(drow/dcol/dwide/dtall chosen so the hole never touches the box edge — the
colour-0 ring stays solid all the way round). Every other box is a perfectly
solid rectangle. OUTPUT = a 1×1 grid holding colors[0] (the holed box's colour).
**Current:** 16.63 pts (public net), prior.
**Target tier:** A-ish (scalar discriminator + separable one-hot output, no
30×30 carrier). Output is a single cell, so the whole task is "recover ONE
scalar colour" — no per-cell label map needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bbox-area via min/max ramps + Pad(99)→Equal | B | 16124 | 102 | 15.31 | 200/200 | correct but heavy (ramps + fp32 carrier) |
| 2 | area = nrows·ncols (occupancy sums), fp16 occ, fp32 Pad | A | 7444 | 39 | 16.08 | 200/200 | dropped ramps; fp32 Pad still 3600 |
| 3 | uint8 Pad(99) carrier | A | 4745 | 39 | 16.53 | 200/200 | 3600→900 carrier |
| 4 | drop redundant fp16 casts (fp32-only, all <2^24 exact) | A | 3565 | 39 | 16.81 | 200/200 | fp16 cast was net-adding a plane |
| 5 | SEPARABLE one-hot output (no 30×30 carrier) | A | **2974** | 90 | **16.97** | **200/200** | **best** |

## Best achieved
**16.97 @ mem 2974 params 90 — adopted? N (orchestrator gates).**
Beats prior 16.63 by **+0.34** (≥+0.3 ✓). Generalizes: stored 265/265 + fresh 200/200.

## Key construction
- Discriminator: a SOLID rect has pixel-count = (#occupied rows)·(#occupied cols).
  The donut hole is STRICTLY interior, so the holed box still occupies every row
  and col of its bbox, yet count < nrows·ncols. The donut colour is the unique
  k≥1 with `cnt_k < nrows_k·ncols_k`. No min/max ramps, no argmax, no flood-fill.
- All quantities are per-channel reductions of the FREE input:
  cnt=ReduceSum[2,3]; rowoc=ReduceMax[3]; coloc=ReduceMax[2];
  nrows=ReduceSum(rowoc,[2]); ncols=ReduceSum(coloc,[3]); area=nrows·ncols.
  fp32 throughout — every value ≤900 so exact (<2^24).
- donut=(cnt<area)∧(k≥1); colidx=Σ_k k·donut (scalar). All <2^24 exact in fp32.
- Output (1×1 at cell (0,0)) built SEPARABLY: chsel=Equal(colidx,arange)[1,10,1,1];
  rowsel/colsel = const (index==0); associate And(chsel,colsel)=[1,10,1,30] (300B)
  then And(·,rowsel)→[1,10,30,30] = FREE output. No 30×30 carrier plane at all.

## Irreducible-floor analysis
Dominant intermediates = the two fp32 occupancy planes rowoc[1,10,30,1] and
coloc[1,10,1,30] = 1200B each (2400B total). They are the direct ReduceMax
outputs over the 10-channel one-hot input; per-channel row/col occupancy is
genuinely needed for both nrows AND ncols (area = product). ReduceMax must emit
fp32 (input is fp32) and a following fp16 cast NET-ADDS bytes (keeps both the
1200 fp32 + 600 fp16) so fp32-only is leaner here. Slicing the input to the
≤18×18 active region would shrink the occupancy axis but forces a [1,10,18,18]
≈13KB slice first — strictly worse. 2400B occupancy is the floor for this form.

## OPEN ANGLES (re-attack backlog)
- Fuse the two occupancy planes into one reduction (no opset-11 op yields both a
  per-row and per-col occupancy from a single pass) — would need a custom Conv
  trick that still preserves per-channel counts. ~1200B payoff if found.
- Replace rowsel/colsel const 30-vectors (60 params) with a cheaper (r==0,c==0)
  selector — minor params trim, no memory effect.

## INSIGHT (transferable)
⭐ "Which solid rect has a hole" = `cnt < nrows·ncols` per colour, where
nrows/ncols are ReduceSum of ReduceMax-occupancy — NOT a flood-fill/enclosure
wall and NOT a bbox-min/max-ramp computation. A strictly-interior hole keeps the
box occupying every row & col of its bbox, so #occupied-rows × #occupied-cols
equals the bbox area for free (no min/max ramps). Exact integer arithmetic in
fp32 (<2^24).
⭐ For a 1×1 (single-scalar) output, build the one-hot SEPARABLY
(And(chsel[1,10,1,1], colsel[1,1,1,30], rowsel[1,1,30,1])) instead of
Pad-a-scalar→Equal — kills the [1,1,30,30] carrier entirely (largest intermediate
becomes a [1,10,1,30] 300B bool). +0.16 here vs a uint8 Pad carrier.
