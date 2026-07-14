# task038 — 1fad071e

**Rule:** 9x9 grid of red(2)/blue(1) pixels; some pixels form 2x2 SOLID boxes, others are
isolated singletons. Boxes never overlap (gap>=1) and singletons never touch a same-colour
pixel, so any 2x2 all-blue block is exactly one genuine blue box. `big_blue` = number of 2x2
BLUE boxes (range 1..5). Output is a 1-row x 5-col grid: first `big_blue` cells blue(1), rest
background(0).
**Current:** 16.88 pts (public net)
**Target tier:** A (closed-form count + separable output, no full 30x30 plane)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice blue->9x9, 2x2 ones-Conv ==4 count, sentinel value-row + Equal + row-gate And | A | 1744 | 86 | 17.49 | 500/500 | ADOPT |

## Best achieved
17.49 @ mem 1744 params 86 — beats prior 16.88 by +0.61.

## Irreducible-floor analysis
Dominant intermediate is the sliced blue channel `input[:,1:2,0:9,0:9]` = [1,1,9,9] fp32 = 324B,
not a full 30x30 plane. Cropping the slice to the 9x9 active grid (generator size bound) kept the
conv response at [1,1,8,8] (256B) instead of the [1,1,29,29] (3364B) a full-canvas conv forces.
Everything downstream is scalar/[1,1,1,30] tiny. mem 1744 is the sum across the few small planes.

## OPEN ANGLES
- Could drop the 9x9-blue slice (324B, the single biggest tensor) if a Conv-over-full-input with a
  channel-1-only 2x2 kernel could be IMMEDIATELY 1-D reduced before materializing 29x29 — but the
  conv response itself is the 29x29 plane, so slicing first is strictly cheaper here. Not worth it.

## INSIGHT (transferable)
"Count fixed-size (2x2) solid blocks of one colour" with isolation guarantees (gap>=1, no same-colour
touch) = a single all-ones KxK no-pad Conv, threshold ==K^2, ReduceSum the hits — NO flood-fill, NO
argmax. Crop the colour-channel slice to the generator's small active grid FIRST so the conv response
stays tiny ([1,1,8,8]) instead of [1,1,29,29]. Route a count-parametric 1-row output via a sentinel
value-row VL=blue+ingrid-1 ({1,0,-1}) -> Equal(arange) -> row-gate And, never building a 30x30 plane.
