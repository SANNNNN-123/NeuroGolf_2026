# task205 — 8731374e (confettibox)

**Rule:** A solid `tall x wide` rectangle of `boxcolor` (tall,wide in [6,10]) sits at
offset (rowoffset,coloffset) in a full random-noise grid, with 1-3 interior special
pixels of a second `color` at strictly-interior box positions (rows 1..tall-2, cols
1..wide-2).  Output is the `tall x wide` box crop; for each special pixel at box-relative
(row,col) the WHOLE output row `row` and WHOLE output column `col` are flooded with
`color`.  So out[i][j] = color if (i is a cross-row OR j is a cross-col) else boxcolor;
cells outside the box are off (all channels false).

**Current:** 14.257 pts, custom:task205, mem 46074, params 239 (was 13.81 / mem 69514 / params 2882)
**Target tier:** detection (run-based box localisation) feeding a separable Tier-A tail —
the output IS row-cond ⊗ col-cond separable, so all memory lives in box DETECTION, not the tail.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior adopted (Dr/Dc MatMul shift + IDX sentinel) | det | 69514 | 2882 | 13.81 | — | baseline |
| 1 | Gather shift (idx=arange+r0) vs 2x 30x30 MatMul; coord ramp vs 900-IDX | det | 67114 | 245 | 13.88 | — | params 2882->245 |
| 2 | sentinel via single Where(in_grid,G,coord) (fused gm1/Gm/Gms) | det | 63514 | 245 | 13.94 | — | -3600 |
| 3 | drop redundant `solid AND gm` (sentinel already kills exterior runs) | det | 62614 | 245 | 13.95 | 200/200 | -900 |
| 4 | in-grid mask = separable 1-D ReduceMax(input,[1,3])/[1,2] (no occ Conv) | det | 57514 | 236 | 14.04 | — | -5100 |
| 5 | equal-neighbour via Equal(diff,0) (Gms ints exact in fp16; drop Abs) | det | 46074 | 239 | 14.26 | 500/500 | -11440 cumulative |
| 6 | boxcolor via 2 scalar Gathers G[r0,c0] (corner always boxcolour); +1 Conv-bias so ONE Where(cross,G,0) plane gives colour AND occupancy; notbox via fp16 Equal | det | 46074 | 239 | 14.26 | 500/500 | best |

## Best achieved
14.257 @ mem 46074 params 239 — adopted? N (build-only). Beats prior 13.81 by **+0.45** (>= +0.3). 266/266 stored, 500/500 fresh under ORT_DISABLE_ALL (scorer-exact).

## 2026-06-29 live-frontier refresh and rejected compression

Current live/source is far ahead of this older note: **15.359697 pts @ mem
14734 params 638** (`teacher:urad7174_top15_public_probe`, source-owned exact
builder).  Mem profile is dominated by `cgrid` 3600B fp32, `bmh` 1800B fp16,
two 6-run Conv outputs at 1500B each, and the 10x10 output one-hot crop.

Tried replacing the box-mask run detector
`Cast(bm)->fp16; Conv(kh/kv); GreaterOrEqual(..., six_fp16)` with an opset-17
uint8 route:
`Cast(bm)->uint8; QLinearConv(kh_i8/kv_i8, scales=1,zp=0); GreaterOrEqual(..., six_u8)`.
Stored result improved to **15.533855 pts @ mem 12274 params 641** (266/266),
saving 2460B.  However a stronger fresh run failed at **934/935**, and comparing
against the incumbent public/live graph showed the candidate output was identical
to incumbent on the failure (`candidate == old`, old also wrong).  Because this
session requires stored + fresh success before adoption, the compression was
rejected and source/network/manifest were restored to the incumbent.

Reusable negative: uint8/QLinear run-sum compression can be stored-equivalent and
score-higher locally, but do not adopt it on tasks whose incumbent already has
rare generator failures unless the candidate passes the agreed fresh gate or the
project explicitly switches to an equivalence-to-incumbent compression policy.

## Irreducible-floor analysis
Remaining memory is the box-DETECTION pipeline run at full 30x30: Gf (3600 fp32 colour Conv,
unavoidable Conv output), G/coord/Gms (sentinel-shifted grid, 5400), two neighbour-diff Convs +
their fp16 Eh/Ev (~7000), two run Convs + run-start masks (~6000), two dilation Convs hcov/vcov +
solid (~6300), Gc colour-plane (1800), and the assorted bool masks.  The dilation is REQUIRED for
robustness: a single >=6 run occurs in noise at ~1e-5/position (~0.7% of grids), so the box must be
the 2-D coincidence solid = hcov AND vcov (p ~ 1e-10) — using horizontal-runs for rows and
vertical-runs for cols SEPARATELY corrupts the bbox on ~1% of fresh grids (tried, rejected).  The
detection is inherently full-grid because the box location is unknown a priori, so the working
canvas cannot be cropped first.

## OPEN ANGLES (not yet tried)
- Fuse hcov/vcov: a single combined dilation/threshold could drop ~1 plane (~1800).
- coord (1800) + Gms (1800): the exterior sentinel. A parity-only sentinel still needs a full plane;
  building the unique-negative coordinate into the Conv (extra weight channel) might remove the
  separate Sub plane.
- run-start to bbox without the full `solid` float plane: rowocc/colocc could perhaps come from
  ANDing hcov/vcov 1-D profiles if the box-rectangle assumption is exploited (separability).
- Tier-A tail is already minimal (separable uint8 label -> Equal->BOOL output, no 10-ch plane).

## INSIGHT (transferable)
⭐ Two reusable levers landed here and generalise broadly:
(1) **Exact fp16 `Equal` replaces Sub+Abs+threshold whenever the operands are integers** (colour
indices, neighbour-difference == 0, "is colour X" tests).  fp16 is exact for ints < 2048, and ORT
runs fp16 Equal fine under DISABLE_ALL.  This cut ~6k bytes here by collapsing every
difference-then-threshold chain to one bool op.  (CAUTION: fp16 `Min`/`Max` triggers ORT's
InsertedPrecisionFreeCast crash under DISABLE_ALL — do index clipping in fp32.)
(2) **A +1 Conv bias on the colour-index plane lets ONE value-carrying Where plane serve both the
colour scalar (global ReduceMax) AND occupancy profiles (max>0.5)** even when the relevant colour
is 0 — it removes the otherwise-separate {0,1} mask plane.  Pair with reading anchor scalars
(boxcolor) via cheap corner Gathers instead of a full mask*value reduction plane.
Net: an over-engineered detection net (69.5KB/2882p) shrank to 46KB/239p with NO algorithm change,
purely by (a) separable 1-D in-grid mask off the free input, (b) Gather-shift vs MatMul-matrix,
(c) integer fp16-Equal collapses, (d) the +1-bias single-plane colour/occupancy merge.

## S8 (2026-07-02) — reverse-ArgMax → select_last_index (+0.012) ADOPTED, div 0
Same idiom as task319 (Slice-reverse variant).


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 11209 -> 9982 (+0.116); gate inc/cand=4/4 (safe). See [[neurogolf-urad-7225-bundle-vein]].

## S16 (2026-07-09) — Not+Where replaces Cast+Sub for no-marker mask (+0.174) ADOPTED, 266/266
Replace Cast(bool→float)+Sub(1,_) with Not(bool), then Gather on bool directly, then Where(bool_mask, float_in_box, 0.0) instead of Mul. Eliminates safe_name_48,49 (float32 intermediates, 120B each); converts safe_name_50,51,58,59,60,61 from float32 to bool (saves 90B each × 6 = 540B). Total: 780B saved, mem 4846→4066, score 16.505→16.679 (+0.174). Node count 70→68.
**Transferable:** Any Cast(bool→float)+Sub(1,_)+Gather+Mul chain can use Not+Gather(bool)+Where.