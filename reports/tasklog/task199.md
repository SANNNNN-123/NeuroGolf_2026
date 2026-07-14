# task199 — 834ec97d

**Rule:** Input is a size×size grid (size 3..15) with ONE non-yellow pixel of colour `cc` at
(row,col); row∈[0,size-2], col∈[0,size-1]. Output: the pixel moves DOWN one row →
output[row+1][col]=cc; yellow(4) fills every cell (r,c) with r≤row AND c%2==col%2 (same column
parity as col); rest is background (0), off-grid all-zero. The colour point (row+1) sits outside
the yellow region (r≤row) so the two paints never overlap.
**Current:** 16.70 pts, label-map + final Equal on 15×15 canvas, mem 3298, params 743.
**Target tier:** B (label map → Equal). Yellow region IS separable (r≤row ⊗ parity(c)), but the
moved single pixel at (row+1,col) is a point that must be stamped, so a single label map is the
clean form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: per-channel [1,10,30,1]+[1,10,1,30] marginals (2×1200B) | B | 5848 | 172 | 16.30 | 266 stored | baseline |
| 1 | replace marginals with ONE no-pad Conv W[2,10,30,1] → [1,2,1,30] (240B) | B | 3298 | 743 | 16.70 | 200/200 | ADOPT-WORTHY |

## Best achieved
16.70 @ mem 3298 params 743 — beats prior 16.30 by **+0.40**. Fresh 200/200.

## Irreducible-floor analysis
After the Conv swap the dominant intermediates are the label-map family: padded L (uint8 [1,1,30,30]
=900B) plus ~7 bool/uint8 15×15 masks (225B each ≈1575B). These are intrinsic to routing a per-cell
colour-index into the FREE BOOL output via Equal. The Conv output is only [1,2,1,30]=240B. The 600
conv-weight params (2×10×30, kh=30 to collapse the full grid height; row∈0..13 but kernel must span
all 30 input rows for a no-pad height-1 collapse) are the only large param block; net trade is
strongly positive (saved ~2160B memory for ~570 params).

## OPEN ANGLES (re-attack backlog)
- Tier-A separable form for the yellow comb alone (rowcond[1,1,15,1]⊗colcond[1,1,1,15] routed into a
  yellow-channel slice of output) + a separate single-pixel stamp into the cc channel — could drop the
  900B L plane, but the priority overlay of two free-output writes is awkward without a label map.
- Shrink the conv: a kw=1 kernel that emits ONLY a presence vector (10×30=300 params) and recover row
  by a second tiny op — unclear it beats the packed 2-channel kernel.

## INSIGHT (transferable)
⭐ To recover a single coloured pixel's (row, col, presence) as scalars WITHOUT the two 1200B
per-channel spatial marginals [1,10,30,1]+[1,10,1,30], use ONE no-pad Conv whose kernel
W[out=2,in=10,kh=30,kw=1] has weight 0 on ch0 (drops background) and packs presence (w=k≥1) and a
row-ramp (w=(k≥1)·r) into two output channels → [1,2,1,30] (240B). col=ReduceSum(presence·colramp),
row=ReduceSum(rowvec). The Conv contracts the channel axis (excluding bg) AND the row axis in one op —
something Reduce* can't do because every in-grid row carries a background pixel, so no full-one-hot
reduction can locate the colour pixel. Cost ~600 params but saves ~2160B memory (net +0.4 pts here).

## S16 (2026-07-06) — public bit-identical golf (franksunp) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
