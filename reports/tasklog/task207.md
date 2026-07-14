# task207 — 88a62173

**Rule:** A 5x5 grid holds four 2x2 sprites with top-left corners (0,0),(0,3),(3,0),(3,3) (a 3-stride 2x2 layout). Three sprites share one pattern ("same"); exactly one ("diff") has a different pattern. All coloured pixels use a single colour `color`. The 2x2 OUTPUT is the diff sprite's pattern in that colour.
**Current:** 17.46 pts, ext:kojimar6275, mem 1840, params 35
**Target tier:** B (closed-form per-channel count threshold; output is the FREE final Pad)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel count over 4 blocks, mask=(cnt==1)\|(cnt==4), Pad to output | B | 1000 | 35 | 18.06 | 200/200 | ADOPT-worthy |

## Best achieved
18.06 @ mem 1000 params 35 — beats prior 17.46 by +0.60 (Y). Generalizes isolated fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate = the four fp32 [1,10,2,2] sprite slices (~160B each, ~640B
total), plus cnt (160B) and the bool masks (120B) and the fp16 cast carrier (80B).
The 30x30 result is the FREE output of the final Pad, so no full-canvas plane is
ever materialised. Slices come straight off the fp32 input so they enter as fp32;
casting them to fp16 would ADD a plane rather than shrink (slice is already fp32).
Everything is on a 2x2 footprint so the net is tiny regardless.

## Key derivation (why no argmax/gather is needed)
Stack the four blocks; per channel per cell let cnt = sum of the four blocks.
The 3 identical "same" blocks contribute 0 or 3; the diff block contributes 0 or 1,
so cnt in {0,1,3,4} (never 2). The diff block's value is 1 exactly when cnt in
{1,4}: cnt=4 all agree incl diff; cnt=1 only diff sets it; cnt=3 the three same set
it but diff does not; cnt=0 none. This holds for EVERY channel incl background ch0,
so (cnt==1)|(cnt==4) reconstructs the diff sprite's one-hot directly — no block
identification, no ArgMax, no Gather (the prior kojimar net used majority + ArgMax +
Gather over [1,10,2,2] fp16 windows, mem 1840).

## OPEN ANGLES (re-attack backlog)
- Replace 4 Slices with one Gather(rows=[0,1,3,4]) + Gather(cols=[0,1,3,4]) -> [1,10,4,4],
  reshape (2,2,2,2), ReduceSum the block axes -> cnt. Removes 3 slice-start/end inits
  (params ~35->~20) but the [1,10,4,4] tensor (640B) + reshape view may not lower mem.
  Score delta from any of these is <0.1 (ln of ~1000), so low priority.

## INSIGHT (transferable)
⭐ "k-of-n identical + 1 odd-one-out, output the odd one" is closed-form with NO
argmax/gather: per-channel COUNT over the n candidate blocks lands in a fixed small
set, and the odd block's value is recovered by a MAGNITUDE-BAND test on the count
(here cnt in {1,4}). Because the test is per-channel, it reconstructs the full
one-hot of the odd block directly, routed into the FREE Pad output. Generalises to
any "majority vs minority over a fixed candidate count" reconstruction.
