# task333 — d43fd935

**Rule:** 10x10 grid, ONE green(3) 2x2 box at (boxrow,boxcol) in [2,7]. 1..2 distinct
scatter colours (never green). For each coloured pixel: if its row is a box-row
(boxrow/boxrow+1) it paints a horizontal ray TOWARD the box (filling up to, not incl, the
green edge); if its col is a box-col it paints a vertical ray. Other pixels stay as dots.
Generator allows at most ONE pixel per box side (left/right/up/down); box-rows/box-cols only
ever hold ray pixels, so ray paths cross only background and never conflict.
**Current (pre):** 15.76 pts, scalar-recovery label-map, mem 10068, params 260.
**Target tier:** B (single colour-index label-map -> Equal). Directional fill toward a box
is closed-form (triangular prefix/suffix MatMul) — NOT a flood-fill wall.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | (pre-existing) scalar boxrow/boxcol recovery + region masks | B | 10068 | 260 | 15.757 | 200/200 | baseline |
| 1 | green-gate (full-plane G@tri gates) on colour plane C | B | 10100 | 448 | 15.74 | 200/200 | no gain (batch of full gate planes) |
| 2 | drop strict matrices, reuse Utri/Ltri inclusive | B | 10100 | 248 | 15.76 | 200/200 | params down, mem flat |
| 3 | fold green-removal (Cn) + double-Where green restore | B | 9900 | 247 | 15.78 | 200/200 | -200B |
| 4 | SEPARABLE gates (row⊗col-side vectors, no full-plane gate MatMul) | B | 9280 | 247 | 15.84 | 200/200 | -620B |
| 5 | beam = disjoint Where-CHAIN (no Add planes) | B | 8680 | 247 | 15.90 | 200/200 | -600B |
| 6 | 4D throughout (drop Squeeze + Reshape planes) | B | 8380 | 243 | 15.94 | 200/200 | -300B |
| 7 | fold "beam>0?beam:C" by using C as Where-chain base case | B | 8080 | 243 | 15.97 | 200/200 | -300B |
| 8 | drop Cn entirely (green never enters the gated prefix sums) | B | 7880 | 243 | **15.998** | 500/500 | (prior probe, never committed) |
| 9 | feed RAW C into the 4 triangular MatMuls; box-row/col restriction moves into the Where condition (drop Crows/Ccols pre-mask) | B | 8128 | 264 | 15.96 | 500/500 | -2 Mul planes |
| 10 | drop the 4 `fill>0` Greater planes: on a box-line the only coloured cell in a side region IS the beam, so cond = (side AND box-line) suffices | B | 7728 | 264 | **16.014** | 500/500 | **NEW BEST** |

## Best achieved
**16.014 @ mem 7728 params 264** (=16.014 pts). Beats prior committed 15.757 by **+0.257**.
Beats the un-committed 15.998 probe by +0.016. ISOLATED FRESH 500/500. **MARGINAL** (+0.257 < +0.3)
but per adopt.py rule a generalizing gain is a WIN — rebuilt and checked into src/custom/task333.py.

### Two structural collapses past the prior 15.998 probe (both eliminate full [1,1,10,10] fp16 planes):
1. **RAW-C triangular fills.** The directional fill MatMuls (`C@Utri`, `C@Ltri`, `Ltri@C`,
   `Utri@C`) can run on the unmasked colour plane C — no box-row/box-col pre-mask Mul needed.
   Any carry that spills onto a non-box row/col is never SELECTED, because the combine
   Where only writes a beam where (side AND box-line) holds. Removes Crows+Ccols (−400B, +2 ops).
2. **No `fill>0` gate.** On a box row the ONLY coloured cells in the left region belong to the
   left beam (generator allows ≤1 dot/side, stray dots live on non-box lines), so where fill==0
   the colour plane C is also 0 → the Where condition needs only `(sidemask AND box-linemask)`,
   not a separate `Greater(fill,0)` plane. Removes 4 bool [1,1,10,10] planes (−400B).

## Irreducible-floor analysis
Fixed costs pin the floor at ~16.0:
- **3600B** fp32 colour-index Conv output [1,1,30,30]. The 10->1 colour reduction must be
  fp32 and the input is 30x30, so this is the irreducible entry plane (FLOOR_RESEARCH).
  Slice-then-reduce ([1,10,10,10]=4000B) is strictly worse.
- **400B** fp32 crop [1,1,10,10] of that Conv (everything downstream runs on the 10x10 region).
- **900B** uint8 padded label [1,1,30,30] feeding the FREE Equal->output (sentinel 10 zeroes
  off-grid). Padding a bool one-hot instead is worse (Pad rejects bool; uint8 [1,10,10,10]=1000).
- Remaining ~2980B = the minimal fill working set: C(200)+G(200)+G_b(100)+four carry MatMuls
  (800)+four gate Ands(400)+the 5-step Where chain(1000)+L10(100)+tiny gate vectors.
Total fixed+minimal ~7800-7900B => pts ~15.99-16.02. Reaching +0.3 (16.06 => mem+params<=7639)
would require cutting the entry/output planes, which are structurally irreducible.

## OPEN ANGLES (re-attack backlog)
- Eliminate the 4 separate carry MatMuls: a single combined triangular can't encode the
  data-dependent box-position gate, so this seems blocked — but a 2-direction reuse (pack
  prefix+suffix into one wider MatMul + a sign-split) is untried.
- Cut the G fp16 plane (200B): ReduceMax rejects bool/uint8, so a float green plane is needed
  for row/col presence; deriving rowg/colg without it (e.g. from the carry edge column) untried.
- Tier-A per-channel route (route fill into FREE output, skip the 3600 colour plane): blocked
  because batching the 8 colour channels makes every working plane 8x (1800B vs 200B) — net worse.

## INSIGHT (transferable)
⭐ "Directional ray-fill from a pixel TOWARD a barrier (here a green box), stopping at the
barrier" is closed-form and NOT a flood-fill wall: carry the colour with a triangular
prefix/suffix MatMul of the colour plane, and let the BARRIER itself supply every gate.
Because the barrier sits at a single position, the side-gate threshold is GLOBAL, so each
2-D gate is SEPARABLE into a tiny row-presence (x) column-side vector pair (ReduceMax + one
[1,N] triangular MatMul) — NO full-plane gate MatMul. The carry can use the raw colour plane
(no green-removal) since the gated cells lie strictly BEFORE the barrier along the scan, so the
barrier value never enters their prefix sum. Combine disjoint directional fills with a single
priority Where-CHAIN whose BASE CASE is the original colour plane C (a gated cell before its ray
has carry 0 == background C) — this folds the "beam>0?beam:C" select and all the Adds into the
chain for free. These four levers cut a 10068B scalar-recovery label-map to 7880B (15.76->16.0).
The residual ~16.0 floor = 3600B fp32 colour-Conv + 900B uint8 output-route, both irreducible.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2792/16.92; full_idx 900B uint8 30x30 label carrier for free Equal->output, idx_f32 400B forced-fp32 crop, spatial_select [10,30] shape-locked. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 3227 -> 3223 (+0.001); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].