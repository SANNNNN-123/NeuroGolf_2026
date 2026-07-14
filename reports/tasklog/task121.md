# task121 — 5117e062

**Rule:** A 13x13 grid holds 3-4 non-overlapping 3x3 "conway" sprites, each a
solid-ish subset of a 3x3 block drawn in its own colour `colors[idx]` (colours
sampled from 1..9 excluding cyan=8). Sprite 0 is special: the CENTER cell of its
3x3 block, (brows[0]+1, bcols[0]+1), is overwritten with cyan (8) as a marker.
The 3x3 output is sprite 0's shape painted in sprite 0's colour, with the center
forced to that colour too (every occupied OR center cell -> colors[0], else 0).
Spacing>=1 guarantees no other sprite intrudes into sprite 0's 3x3 block.

**Current (prior stored):** ~16.22 pts, tier B
**Target tier:** B — needs one data-dependent spatial scan (marker location) +
one window extraction; not Tier A (output colour is the data-dependent colors[0],
shape is non-separable conway pattern) but well above detection floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | two full planes (cyplane slice 3600 + colf 3600) | B | 9156 | 102 | 15.87 | (passed train) | works, heavy |
| 2 | single colf plane w/ MARK=1000 cyan weight; reduce for marker | B | 5856 | 99 | 16.31 | — | one big plane |
| 3 | one-hot at 3x3 then Pad straight into free output (kill 900B L) | B | 5136 | 98 | 16.44 | — | — |
| 4 | ArgMax marker (kill Greater/Cast/Mul/ReduceSum tail) | B | 4564 | 36 | 16.57 | 200/200 | ADOPT candidate |

## Best achieved
16.566 @ mem 4564 params 36 — beats prior ~16.22 by +0.35. Fresh isolated 200/200.

## Irreducible-floor analysis
Dominant intermediate = `colf` [1,1,30,30] f32 = 3600B (the only full-grid plane).
It cannot be removed: locating the data-dependent cyan marker requires a full
spatial scan, and reading the 3x3 sprite window requires the per-cell value plane.
fp16 does NOT shrink it (ORT upcasts full planes to fp32 in the trace). Remaining
tail ~960B = the [1,1,3,30] row-gather window (360B) + two [1,1,30,1]/[1,1,1,30]
ReduceMax profiles (120B each) + tiny scalars. Single MARK=1000 cyan-weighted Conv
collapses 10ch->1 AND tags the marker so marker-locate + value-read + colour-recover
all read from ONE plane.

## OPEN ANGLES (re-attack backlog)
- Shrink the 360B window row-gather: a fused 3x3 GatherND would leave a 9-elem
  tensor (~36B) instead of the 90-elem [1,1,3,30] step — payoff ~+0.05, marginal.
- The two ReduceMax profiles could in principle be one op if a single reduction
  produced both row & col argmax — not expressible in opset 11; ~+0.05.
- Pushing past ~16.7 needs eliminating the full colf plane, which is blocked by
  the data-dependent marker scan. Likely at the structural floor for tier B.

## INSIGHT (transferable)
⭐ "Locate a unique sentinel marker AND read a window around it" collapses to ONE
full plane: a single 1x1 Conv `sum_k w_k*input_k` with an OUTSIZED weight on the
marker channel (w_marker=1000, w_k=k) does triple duty — (a) per-row/col ReduceMax
> MARK/2 (or ArgMax) locates the marker as scalars from cheap 120B profiles,
(b) a 3x3 Gather window around it reads the local pattern, (c) the underlying
sprite colour recovers as ReduceMax(window with marker->0). Prefer ArgMax over
Greater->Cast->Mul-ramp->ReduceSum to get the marker index: it removes ~6 small
intermediates and ~60 ramp params in one op. And one-hot at the small (3x3) label
then Pad-into-free-output beats padding the label to 30x30 first (saves the 900B
uint8 plane). When the marker sits at a guaranteed interior position (center in
[1,11]), the +/-1 window needs no Clip.
