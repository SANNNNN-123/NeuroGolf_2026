# task334 — d4469b4b

**Rule:** Input is a size×size grid (default 5×5) with random pixels of ONE colour c∈{1,2,3}; the pixel POSITIONS are irrelevant. The OUTPUT is always a fixed 3×3 gray(5)/black(0) pattern keyed ONLY on c: c=1→[[0,5,0],[5,5,5],[0,5,0]], c=2→[[5,5,5],[0,5,0],[0,5,0]], c=3→[[0,0,5],[0,0,5],[5,5,5]]. Pure COUNT→FIXED-PATTERN (whole output determined by ONE scalar).
**Current:** 18.59 pts, ext:kojimar6275, mem 572, params 37
**Target tier:** B (count→fixed-pattern) — output is a const 3×3 per recovered colour scalar; cheapest tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | ReduceSum→slice 1:4→ArgMax→MatMul table→Where ch0/ch5→Concat(zero4)→Pad | B | 236 | 103 | 19.17 | 200/200 | ok |
| 2 | reuse single zero3 const ×4 in Concat (drop zero4=36) | B | 236 | 67 | 19.29 | 200/200 | ok |
| 3 | replace MatMul-table+Reshape with Gather([3,3,3] u8 table) by ci | B | 167 | 63 | 19.56 | 500/500 | ADOPTED |

## Best achieved
19.56 @ mem 167 params 63 — adopted? Y. Beats prior 18.59? Y (+0.97).

## Irreducible-floor analysis
Dominant intermediates: oneh6 [1,6,3,3] uint8 = 54B (needed to place gray at colour channel 5 and black at channel 0 before the free Pad) and cnts [1,10,1,1] fp32 = 40B (ReduceSum must run on the full 10-ch input; ReduceSum rejects uint8). The 6-ch one-hot is the cost of routing into the free output without materialising a 30×30 carrier — alternative (label plane Padded to [1,1,30,30] then Equal) costs 3600B, far worse. Not at a hard floor but already deep tier-B.

## OPEN ANGLES (re-attack backlog)
- Drop cnt3 (12B) by excluding ch0 from cnts without a Slice (e.g. zero ch0 via a [1,10,1,1] weight in the ReduceSum path) — marginal ~12B.
- Collapse ch0+ch5 build into the Gather: a [3,6,3,3] table Gathered by ci gives oneh6 directly (drops the two Where + Greater + Reshape) but the table grows to 54 params and gather output is still 54B — net likely a wash; worth a measure.

## INSIGHT (transferable)
⭐ When output content is a fixed small pattern keyed on a recovered scalar index, a Gather of a tiny uint8 [K,h,w] table by that scalar beats a one-hot⊗MatMul-table: it skips the f32 one-hot, the f32 MatMul result, AND the Reshape (two ~36B planes), and the gathered uint8 plane doubles as the indicator directly. Combine with the COUNT→FIXED-PATTERN free-Pad route. Position-robust colour recovery: ReduceSum(input,[2,3]) → Slice to the candidate colour channels → ArgMax (excludes background ch0 by slicing, exact when exactly one colour is present).
