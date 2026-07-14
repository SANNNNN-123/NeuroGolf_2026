# task065 — 2dc579da

**Rule:** A (2s+1)x(2s+1) grid (s in 1..7) is filled with background `b`, then a full
horizontal line (row=s) and full vertical line (col=s) of `linecolor` split it into four
s x s quadrants. A single `dotcolor` pixel sits at (row,col) in one quadrant (never on a
line). Output is an s x s grid, all `b`, with `dotcolor` at the dot's quadrant-local
position (lr=row if row<s else row-(s+1); lc likewise). Cells outside s x s are all-bg
(every one-hot channel off). The three colours line/dot/b are distinct.
**Current:** 16.17 pts (prior stored), label-map + Equal, mem ~6758.
**Target tier:** B (floor-break) — scalars only + one small label plane routed into the
FREE one-hot output; data-dependent s x s shape via Pad-sentinel + Equal (task121 idiom).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: counts + Mul/ReduceSum masked marginals (4x1200B fp32) | B | 6758 | 108 | 16.166 | 200/200 | =P, no beat |
| 1 | GATHER dot channel out of the two raw spatial marginals (drop the 2 masked-copy marginals) | B | 4366 | 109 | 16.594 | 500/500 | BEST |

## Best achieved
16.594 @ mem 4366 params 109 — adopted? N (orchestrator gates). Beats prior 16.17 by +0.42 → YES.
Fresh 500/500, fully generalizes (evaluate 266/266 stored + held-out 500/500).

## Irreducible-floor analysis
Dominant intermediates after the win: two fp32 per-channel marginals `rc`=ReduceSum(input,[3])
[1,10,30,1] 1200B and `cc_`=ReduceSum(input,[2]) [1,10,1,30] 1200B, plus the padded uint8 label
plane [1,1,30,30] 900B (the carrier for the final Equal -> free output). The marginals are
reductions of the FREE fp32 input so they MUST be fp32 and span all 10 channels (the dotcolour
channel is selected AFTER, by a 120B Gather). The 900B uint8 plane is the minimal carrier for a
[1,10,30,30] one-hot output. ~3300B of 4366B is these three; the rest is scalar derivation.

## OPEN ANGLES (re-attack backlog)
- Collapse the two 1200B marginals: a single MatMul contracting the channel axis with dotvec would
  need a transpose that re-materialises a 1200B tensor (no net gain). Gathering the dot channel from
  `input` first is a 3600B plane (worse). The 2x1200B looks like the per-channel-position floor.
- Position without per-channel marginals (e.g. from a colf index plane) costs a 3600B plane — worse.

## INSIGHT (transferable)
⭐ When you need ONE channel's 1-D row/col profile and you already ArgMax'd that channel's index
as a scalar, do NOT build masked-copy marginals (Mul by a [1,10,1,1] one-hot then ReduceSum over
channels = two full [1,10,30,1] tensors). Instead reduce the free input over the spatial axis ONCE
([1,10,30,1]) and `Gather(marginal, chan_idx, axis=1)` -> [1,1,30,1] (120B). This halved the marginal
cost here (4x1200B -> 2x1200B) for +0.43 pts. Generalises to any "select-then-profile-one-channel".
