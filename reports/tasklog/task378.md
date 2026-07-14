# task378 — ec883f72 ("antenna man": add 4 diagonal rays to a bullseye)

**Rule:** Input is a concentric bullseye — OUTER ring of colors[1] (rows
[row-2,row+height+1], cols [col-2,col+width+1]), a BLACK middle ring, and a
solid INNER rectangle of colors[0] (rows [row,row+height-1], cols
[col,col+width-1]). OUTPUT = INPUT plus 4 diagonal rays of colors[0] emanating
OUTWARD from the 4 corners of the OUTER ring: from (r0,c0)=TL dir(-1,-1) on
r-c==r0-c0 with r<r0; (r0,c1)=TR dir(-1,+1) on r+c==r0+c1 with r<r0;
(r1,c0)=BL dir(+1,-1) on r+c==r1+c0 with r>r1; (r1,c1)=BR dir(+1,+1) on
r-c==r1-c1 with r>r1. Rays clipped to the size x size grid. size in [6,12].
**Current:** stored 14.43 (untriaged). Custom net = 15.90 pts, mem 8938, params 52.
**Target tier:** B (geometric reconstruction → label/one-hot fold into FREE Where
output). Not S/A: each ray is a data-dependent 45° DIAGONAL (r±c==const) clipped
by a data-dependent half-plane — not a fixed per-cell linear map (S) and a single
diagonal is not a row-cond⊗col-cond product (not separable → not A).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bbox-of-all-nonbg ±0, center-Gather colour, fp32 30×30 | B | 23928 | 78 | — | 115/200 | WRONG: clipped outer bbox + center lands on black middle ring |
| 2 | inner-rect (colors[0]) bbox ±2 = true outer corners; argmin-span inner id | B | 33772 | 85 | 14.57 | 200/200 | correct, too heavy |
| 3 | drop colf; fp16 RmC/RpC + per-channel; Where-based rmin/rmax | B | 18826 | 75 | 15.15 | 200/200 | +0.72 |
| 4 | W=12 active canvas for all 2-D planes; Pad mask to 30×30 | B | 10174 | 112 | 15.76 | 200/200 | +1.33 |
| 5 | slice per-channel presence to W (bool slice); reuse ramps | B | **8938** | **52** | **15.90** | **500/500** | WIN +1.47 |

## Best achieved
**15.90 @ mem 8938 params 52 — fresh 500/500.** Beats stored 14.43 by **+1.47**.
Adopted? **N** (build-only).

## Irreducible-floor analysis
Dominant intermediates: two `ReduceMax(input,axes=[3/2])` per-channel presence
planes `chrow32`/`chcol32` [1,10,30,1] fp32 = 1200B each (ReduceMax outputs fp32
from the fp32 input and keeps the un-reduced 30-axis — can't be fp16 or W-sized
without slicing the 10-ch input, which costs far more). The 30×30 carrier pair
`mask30_u8` (Pad out) + `mask` (bool for Where) = 900+900: a 30×30 BOOL cond is
required to broadcast against the FREE [1,10,30,30] output, and Pad rejects bool
so a uint8 pad + bool cast are both needed (~1800B near-minimal).

## OPEN ANGLES (re-attack backlog)
- Eliminate one of chrow32/chcol32 (2400B): identify inner colour + bbox without
  full per-channel reductions (e.g. detect colors[0] as the non-bg colour absent
  from the content bbox border, recovered from a single colf plane). Unclear it
  beats 2×1200.
- The 1800B 30×30 carrier: only escapable if Where could take a uint8 cond (this
  ORT build rejects it) or a [1,1,W,W] cond could broadcast to [1,10,30,30]
  (shape mismatch — can't).

## INSIGHT (transferable)
⭐⭐ **Clipped concentric figures: recover geometry from the INNERMOST shape, not
the all-content bbox.** A bullseye/frame clipped at a grid edge has a WRONG outer
bbox, but the inner rectangle is clipped strictly LATER (further in), so on every
on-grid side the visible inner bbox edge = the true edge, and outer corners =
inner-bbox ± constant. Clipping automatically pushes any off-grid corner's
diagonal fully off-grid, so the in-grid mask (rowin⊗colin) suppresses phantom
rays with no special-casing — exact across all clip configurations.
⭐ **Inner vs outer of nested shapes = smaller bbox SPAN** (outer ⊇ inner ⇒ strict
inequality, never ties): pack per-channel span into [1,10,1,1], +BIG for absent,
+2·BIG for bg ch0, Equal-to-ReduceMin → inner-colour one-hot in one shot, and the
SAME one-hot doubles as the Where fill colour (Cast→fp32) and as the channel
selector for the inner bbox edges (Σ_k edge_k·is_inner_k).
⭐ **45°-ray reconstruction = union of two diagonals (r-c==a) and two
anti-diagonals (r+c==b) through bbox corners, each gated by a 1-D half-plane**;
keep RmC/RpC fp16 (values in [-29,58] exact) and build everything on the W=12
active canvas (size≤12 lives top-left), then Pad the W×W mask to 30×30 once.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 4449 -> 4432 (+0.004); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].