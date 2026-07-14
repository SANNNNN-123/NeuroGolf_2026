# task388 — f5b8619d ("2x2 tile a sparse grid + cyan lines through pixel-columns")

**Rule:** Input is a size x size grid (size in 2..6) with 1..size colored pixels of a
single colour `color` (cyan excluded) on a 0 background. Output is 2*size x 2*size:
(1) every input column c that holds >=1 colored pixel paints FULL vertical cyan(8) lines
at output cols c AND c+size (all 2*size rows); (2) each input pixel (r,c)=color is tiled
2x2 into {(r,c),(r,c+size),(r+size,c),(r+size,c+size)}=color, drawn AFTER cyan so colour
overwrites. Per output cell: color if src[r%size][c%size] is a pixel, else cyan if column
(c%size) has a pixel, else 0.
**Current:** 16.23 pts (public net). Now **16.74 pts custom, mem 3785, params 78.**
**Target tier:** B — output colour per cell needs the data-dependent 2x2 TILING of an
arbitrary sparse pixel set (not row⊗col separable: a single pixel binds both r and c),
so a label plane is required; cyan part alone is separable but the colour tile is not.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | W=12 canvas; colf via Mul+ReduceSum on [1,10,12,12]; double-MatMul tile + separable cyan; off-output sentinel | B | 17125 | 66 | 15.25 | 200/200 | correct but two [1,10,12,12] fp32 planes (5760 each) dominate |
| 2 | drop colf plane: src=(bg-channel==0)&in-grid from a 1-CH bg slice [1,1,12,12]; k from cnt=ReduceSum[2,3]; size from ReduceMax[1,3] profile | B | 5693 | 66 | 16.34 | 200/200 | killed both 5760 planes |
| 3 | split canvases IW=6 (input/src) vs W=12 (output); bg slice + R/CT matrices shrink to 6-wide | B | **3785** | 78 | **16.74** | **200/200** | FINAL |

## Best achieved
**16.74 @ mem 3785 params 78 — 266/266 stored, fresh 200/200 (isolated).**
Adopted? N (build agent). Beats prior 16.23? **Y (+0.51).**

## Irreducible-floor analysis
Dominant intermediate is the **900 B Pad** ([1,1,30,30] uint8 label feeding the FREE
final Equal). The colour tile is a tiling of an arbitrary sparse set → NOT separable, so
a per-cell label map is unavoidable, and the Equal must span the full 30x30 output → the
900 B uint8 Pad is the canonical label-map floor. The rest is the minimal label/mask
chain on the 12x12 output (a handful of 144 B bool/uint8 [1,1,12,12] planes) + one 288 B
fp16 tilef. The 30x30 colour-collapse plane (normally 3600 B) was ELIMINATED entirely:
colored occupancy comes from a single-channel bg slice cropped to 6x6 (144 B fp32) + the
in-grid mask, and the colour value k comes from the cnt=ReduceSum(input,[2,3]) [1,10,1,1]
vector — no per-cell colour plane is ever built.

## OPEN ANGLES (re-attack backlog)
- The 900 Pad is the label floor for a non-separable per-cell output; only a separable
  route (impossible here — the tile is sparse) would beat it. Considered done.
- Could fuse L1/Lin/Lw Where chain (3x 144 B uint8) but the Pad floor dominates; ~negligible.

## INSIGHT (transferable)
⭐ For a "tile the grid 2x2 by a data-dependent shift `size`" rule, the 2x2 tiling is the
double-MatMul `R @ src @ C^T` with R[Rout,rin]=(Rout==rin)|(Rout==rin+size) (and C^T
symmetric) — built from the scalar `size` via Equal on an output-ramp vs an input-ramp(+size),
NO Mod and NO Kronecker index plane. ⭐ When the only foreground is "non-background"
(every fg cell shares one colour and bg is channel 0), the colored-pixel occupancy is
`(bg_channel==0) AND in-grid` from a SINGLE-channel bg slice cropped to the active region
(144 B) — this dodges the 3600 B colour-collapse plane entirely; recover the colour value
separately as a scalar from cnt=ReduceSum(input,[2,3]). Split the input vs output active
canvases (IW=6 vs W=12) so the MatMul matrices stay rectangular-small.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 2215 -> 2191 (+0.011); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].