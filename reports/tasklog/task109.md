# task109 — 47c1f68c

**Rule:** Input is a (2s+1)x(2s+1) grid (s = `size` in [3,6]) with a full `linecolor`
cross at row=s and col=s, and a small `color` sprite in the top-left s x s quadrant
(rows,cols < s; the cross and other three quadrants are linecolor/background). Output
is a 2s x 2s grid that is the 4-fold reflection symmetrization of the sprite MASK,
painted in `linecolor` (the sprite's own colour and the cross are dropped):
for each sprite pixel (r,c) set (r,c),(r,2s-1-c),(2s-1-r,c),(2s-1-r,2s-1-c)=linecolor.
**Current:** 16.33 pts (public net), mem ~5395, params ~403
**Target tier:** B — data-dependent reflection of a non-separable sprite shape; needs
one per-channel spatial reduction (1200B) + one 30x30 output plane (900B), both at floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior draft: E@M@E fold-matrix double-MatMul on 12x12 fp16 + uint8 label pad | B | 5395 | 403 | 16.33 | 200/200 | = P (no gain) |
| 2 | SEPARABLE mirror via two data-dependent Gathers idx[i]=clip(min(i,t-i),0,5) on a uint8 6x6 sprite (replaces fold matrix + 2 fp16 12x12 MatMuls) | B | 3975 | 108 | 16.685 | 200/200 | ADOPTED (+0.355) |

## Best achieved
16.685 @ mem 3975 params 108 — adopted? Y. Beats prior 16.33? Y (+0.355).

## Irreducible-floor analysis
Two intermediates dominate and both resist removal:
- **1200B** `Xc = ReduceSum(input, axes=[2])` [1,10,1,30] fp32 — the per-channel column
  profile. Serves BOTH width (n=2s+1 via a nonzero-conv col-sum) AND linecolor id
  (the unique k>=1 whose max per-channel column-count == n; a sprite of width<=s-1 can
  never fill a full column, so this is the only robust discriminator). Any per-channel
  spatial extraction is 10ch x 30 = 1200B; a Gather of the centre column/last row, a
  ReduceMin-over-rows, or a MatMul double-contraction all force a [1,10,30,1]/[1,10,1,30]
  middle tensor = 1200B. Total-count (== 2n-1) would be 40B but COLLIDES with a full 5x5
  sprite at size=6 (sprite count can hit 25 = 2n-1), so it is unsafe.
- **900B** padded uint8 label map L [1,1,30,30] for the final `Equal(L, arange)`. The
  output is 30x30 and in-grid background must emit channel-0=True, so the And-broadcast
  trick is unusable; one 30x30 carrier plane (mirror=linecolor, in-grid-bg=0,
  off-grid=sentinel-10) is the leanest single-plane route. Irreducible (output is 30x30).
The 12x12 work planes are now uint8/bool 144B each (was fp16 288B), so the mirror chain
is no longer a cost driver.

## OPEN ANGLES (re-attack backlog)
- Linecolor from a fixed-small centre window input[:,:,3:7,3:7] (640B) instead of 1200B —
  the cross centre (size,size) always lies in [3,6]^2, but for size>=4 the window also
  catches sprite cols 3..size-1, so isolating "the colour forming the in-window full
  line" cleanly is unproven. Payoff only ~+0.13.

## INSIGHT (transferable)
⭐ A data-dependent 4-fold reflection symmetrization is SEPARABLE and need NOT use the
task112 fold-matrix double-MatMul: out[r,c] = sprite[min(r,t-r), min(c,t-c)] with
t=axisconst, realised as TWO Gathers (idx[i]=clip(min(i,t-i),0,SRC-1)) on a **uint8**
sprite — keeps every work plane uint8/bool (half of fp16) and drops the E matrix, the
Greater, and 4 fp16 planes. The fold index min(i,t-i) auto-restricts to the source
quadrant; AND the mirror with the in-grid rect AFTER gathering so the WORK-canvas
reflection (about WORK/2) does not leak past the true 2s output region.
