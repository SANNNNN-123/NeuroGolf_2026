# task159 — 6b9890af

**Rule:** A 3x3 conway sprite S (every row & col has >=1 on-cell; diagonally
connected) is drawn at NATIVE size in one colour `color` (!= red 2) somewhere; a
hollow red box of side `outsize = 3*magnifier + 2` (magnifier in 1..4 -> outsize
in {5,8,11,14}) is drawn elsewhere. OUTPUT is `outsize x outsize`: the perimeter
is all red (2); the interior (inner 3m x 3m block at offset (1,1)) is the sprite
magnified by m — out[r*m+dr+1][c*m+dc+1] = color iff S[r][c], else bg 0.

**Current:** 15.49 pts, custom:task159 (prior draft), mem 13331, params 166
**Target tier:** B — output is a deterministic per-cell label on a <=14x14 canvas;
the only forced full-grid float is the colour-footprint plane (sprite occupancy
is non-separable so needs a 2-D read), capped near ~16.8 by that 3600B plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior: footprint Conv, gather 14x14 sprite@TL, magnify via 2 more 14x14 gathers, 14x14 masks | B | 13331 | 166 | 15.49 | (stored) | the 14x14 footprint copy (1680 fp32 [1,1,14,30]) + 3x 784 fp32 14x14 magnify planes bloat mem |
| 2 | gather EXACT 3x3 sprite from fpf (3 rows+3 cols), magnify via flat-index gather | B | 11580 | 157 | 15.63 | — | int64 [1,1,14,14] sidx (1568) + fp32 14x14 (784) still heavy |
| 3 | magnify via two small gathers S[gidx[i],gidx[j]] (no index plane) | B | 9354 | 177 | 15.84 | 200/200 | kills the int64+fp32 14x14 planes |
| 4 | sprite-first label order (drop interiorB); 1-D separable border/in-grid masks | B | 8874 | 177 | 15.89 | — | removes RintB/CintB/interiorB |
| 5 | rmin/cmin via Where(present,ramp,99)+ReduceMin (task195 idiom) | B | 7914 | 177 | 16.00 | — | drops Mul/Sub/Mul/Add chains + Casts |
| 6 | drop magB Reshape — keep [14,14] rank-2, broadcasts against [1,1,14,14] masks | B | 7718 | 173 | **16.03** | 500/500 | final |

## Best achieved
**16.0265 @ mem 7718 params 173** — adopted? N (orchestrator gates). Beats prior
15.49 by **+0.54**. 265/265 stored, 500/500 fresh isolated.

Key exact pieces:
- m = (#red cells - 4) / 12  (red ring perimeter = 12m+4) — a scalar from one
  ReduceSum + Gather(ch2), NO red full plane.
- color = the single present channel with k!=0,2 (sprite is the only non-bg
  non-red object) -> scalar uint8.
- S[3,3] gathered DIRECTLY from the footprint Conv plane at the bbox top-left
  (sprite covers all 3 rows/cols so bbox is exactly 3x3); never a 14x14 copy.
- magnify = out[i,j] = S[gidx[i], gidx[j]], gidx[i]=clip(floor((i-1)/m),0,2),
  via Gather(S,axis=0) then Gather(.,axis=1) -> [14,14] bool. No int64 index
  plane, no fp32 14x14 plane (this was the single biggest mem win, -2.2KB).
- label priority offgrid>border>sprite>bg via 3 chained Where on the 14x14
  uint8, Pad to 30x30 (sentinel 10), Equal(L, arange) into free BOOL output.

## Irreducible-floor analysis
mem 7718 = fpf Conv 3600 (the one fp32 per-cell reduction of the 10-ch input;
sprite occupancy is a NON-separable 2-D pattern so it needs a genuine 2-D read —
the canonical 3600B floor, confirmed by FLOOR_RESEARCH: dtype tricks can't shrink
it and the sprite can sit anywhere in 30x30 so no fixed small crop) + Pad 900
(30x30 uint8 output-shaping label) + occr 360 ([1,1,3,30] row-gather, the
unavoidable cost of pulling 3 rows from a 30-wide plane) + ~2.9KB small 14x14
working masks/labels and 1-D recovery vectors. fpf alone caps this construction
near pts ~16.8.

## OPEN ANGLES (re-attack backlog)
- Shave occr 360: gather the 9 flat sprite positions in ONE Gather from a
  reshaped fpf — blocked, the [900] reshape of fpf is itself 3600B (worse).
- Trim the 3 uint8 14x14 Where (588B): fold sprite+border into a banded value
  then one threshold — likely <0.05 pts, not worth the readability cost.
- The fpf 3600 floor is the ~16.8 cap; only a data-dependent crop of the sprite
  region could break it, and the sprite position is unbounded in 30x30 (the
  task134/task011-class colour-plane wall). Treat 7718 as near-floor.

## INSIGHT (transferable)
⭐ For an UPSCALE/magnify of a small recovered KxK pattern by a data-dependent
factor m, build the output via `out[i,j] = S[gidx[i], gidx[j]]` with TWO small
gathers (Gather(S, gidx, axis=0) then Gather(., gidx, axis=1)) where
gidx[i]=clip(floor((i-1)/m),0,k-1). This avoids BOTH the int64 flat-index plane
(r*k+c over the full canvas, 8B/cell) AND the fp32 index-arithmetic plane —
the two-gather form keeps only [canvas,k] and [canvas,canvas] bool intermediates.
It generalises task195's const-index Kronecker to a RUNTIME magnifier and beat a
naive flat-Gather by ~2.2KB here. Also: recover the magnifier as a cheap scalar
from the red-ring PERIMETER count (12m+4) rather than a row-span reduction — no
red full plane needed. Keep the magnify result rank-2 [K,K] and let it broadcast
against the [1,1,K,K] geometry masks (skip the Reshape, saves one K*K plane).
