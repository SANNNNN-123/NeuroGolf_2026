# task238 — 9aec4887

**Rule:** Input has two separated objects on a bg grid: a CYAN(8) conway sprite (LxL blob,
bbox EXACTLY LxL, L∈{3,4,5}) and a hollow box frame (L+2)x(L+2) drawn with 4 distinct non-cyan
colours (top=colors[0], right=colors[1], bottom=colors[2], left=colors[3], each a length-L run,
corners empty). Output = the (L+2)x(L+2) box frame with the cyan sprite copied into the interior
(+1,+1), each cyan cell recoloured by its sprite-coord quadrant (d0=r−c, d1=(L−1)−r−c: top/right/
bottom/left by the signs of d0,d1; diagonal stays cyan). Empty interior cells → bg.
**Current (prior):** 15.34 pts, closed-form sprite+box, mem 15612, params 129
**Target tier:** A (closed-form scalar recovery + 7x7 work-canvas; the only 2-D plane is the
sprite gather, which is geometrically required → not Tier S).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp16 7x7 work canvas + ArgMax(occ)→min, Mul-ramp→max | A | 13712 | 127 | 15.46 | — | partial |
| 2 | drop rmax: classify H/V by colcount, L from cmax | A | 12432 | 127 | 15.56 | — | partial |
| 3 | colcount via ReduceSum(colocc): kills the 1200B cmaxsrc Mul plane | A | 11192 | 69 | 15.67 | 200/200 | win |
| 4 | fold interior+sprite masks into one Where | A | 11143 | 69 | 15.6753 | 200/200 | **ADOPT** |

## Best achieved
15.6753 @ mem 11143 params 69 — beats prior 15.34 by **+0.34**. 266/0 stored, 200/200 fresh.

## Irreducible-floor analysis
Dominant intermediate: **cyplane [1,1,30,30] fp32 = 3600B** — the channel-8 slice from which the
7x7 sprite window is gathered. The sprite SHAPE is genuinely 2-D (which interior cells are cyan),
so it cannot come from 1-D row/col profiles; the 30x30→7x7 gather MUST pass through a single-channel
30x30 plane (gathering all 10 channels first is 8400B; casting to fp16 adds an 1800B cast plane →
strictly worse). Next: rowocc/colocc [1,10,30,1]/[1,10,1,30] fp32 = 1200B each — the per-channel
occupancy profiles; ReduceMax(input) is fp32-pinned and both axes are needed (rmin/cmin via ArgMax,
colcnt via ReduceSum). Lfull uint8 30x30 = 900B is the Pad carrier into the Equal→output (Pad rejects
bool even at opset 11, so the uint8 Pad→Equal order is forced). These three (cyplane + 2 profiles +
Lfull) are structural; remaining ~2.5KB is the 7x7 fp16/bool work canvas.

## OPEN ANGLES (re-attack backlog)
- cyplane 3600B: only escape would be a single GatherND [7,7,2] reading both spatial dims at once
  (kills the 840B Vr intermediate) but the data-dependent int64 index plane likely costs ≥784B param +
  build planes — est net ~wash, untried.
- Vr 840B fp32: the row-gather intermediate; col-first gather is the same 840B. Unavoidable two-step.

## INSIGHT (transferable)
⭐ **Per-channel COLUMN COUNT (ReduceSum of the {0,1} ReduceMax occupancy profile, axes=[3]) → [1,10,1,1]
40B scalar** classifies box-side ORIENTATION (1=vertical Lx1 run, L=horizontal/blob) AND recovers a
blob side-length L in ONE cheap reduction — eliminating a 1200B max-index Mul-ramp plane. Whenever a
task needs "how many distinct rows/cols a channel spans" (run orientation, side length, bbox dims),
ReduceSum-of-occupancy beats Mul(occ,ramp)+ReduceMax. Also: bbox MIN-index = ArgMax(occupancy profile)
directly (first-True, fp32 only — ArgMax rejects bool), no where-source plane needed. And once scalars
are recovered, run the whole small (≤7x7) value-assembly canvas in fp16 (half mem, values 0-10 exact;
use range tests not Equal since ORT fp16 Equal is unsupported).
