# task095 — 4258a5f9

**Rule:** size=9 grid. Input holds up to 9 isolated GRAY(5) dots at (r,c) with
r,c in [1,7] and non-overlapping length-3 boxes (=> centres >=3 apart, never
8-adjacent). Output: each gray dot becomes a 3x3 BLUE(1) box with the centre
cell kept GRAY(5); BLACK(0) background elsewhere; off-grid all-zero.
**Current (public):** 18.20 pts, single `Conv(input, W[10,10,3,3], pad=1)` -> output (FLOAT), mem 0, params 900.
**Target tier:** S/A — pure spatial dilation; but the public net is ALREADY the
optimal single mem=0 Conv for this structure.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | crop gray ch5 -> 9x9, banded ring-Conv (centre 5, ring 1) -> L, Cast u8, Pad sentinel 10, Equal->BOOL | B | 1629 | 37 | 17.58 | 200/200 | CORRECT but BELOW P (worse than 18.20) |

## Best achieved
17.58 @ mem 1629 params 37 — adopted? N. Beats prior 18.20? N (worse).

## Irreducible-floor analysis (why beating 18.20 is INFEASIBLE)
The public net is a SINGLE `Conv(input[1,10,30,30], W[10,10,3,3], pads=1)` whose
output IS the graph output => **mem 0**, params = 10*10*9 = 900, score
25-ln(900) = 18.20. To beat by +0.3 needs mem+params < 665.

Two structural escapes, both blocked:
1. **Shrink the single Conv's params (keep mem 0).** The weight element count is
   fixed by shape [O,I,kh,kw]. O must be 10: output channel 0 (background) is a
   REQUIRED one-hot channel (target sets ch0=1 on every in-grid bg cell, scored
   per-channel out>0), and it is NOT a simple copy — at a blue-stamped cell the
   input had ch0=1 but the target needs ch0=0, so the Conv must SUBTRACT the
   dilated gray from the bg-passthrough; only a full cross-channel Conv does this
   in one op. I must equal input channels (10): grouping cannot reduce I because
   output ch1 (blue) is computed from input ch5 (gray) — a 5->1 cross-channel
   map that no `group` partition of {0..9} keeps in the same group. Kernel must
   be 3x3 (blue is the full 3x3 ring footprint). => 10*10*9 = 900 is forced.
   fp16 weight does NOT help (params count ELEMENTS, not bytes).
2. **Decompose into >=2 ops (crop / route into free output).** Any second op
   makes a 30x30 intermediate. The Where(blue_mask, blue_onehot, input) idiom
   needs a 30x30 BOOL cond (900B) AND a float source to threshold it from
   (another 30x30 plane, or a cropped pipe that still pads back to a 900B 30x30
   plane). Best cropped pipeline (attempt 1) lands mem 1629 -> 17.58. The
   theoretical best 2-op graph is ONE 900B bool plane + ~10 params =>
   25-ln(~940) = 18.15, STILL below 18.20. A 30x30 intermediate is strictly
   worse than the mem-0 single Conv.

Conclusion: the dilation's cross-channel 3x3 footprint over the full 10-channel
one-hot, plus the required subtractive background channel, pin this to a single
900-param mem-0 Conv. No decomposition beats mem 0.

## OPEN ANGLES (exhausted)
- group/depthwise Conv: blocked by the 5->1 (gray->blue) cross-channel map.
- fp16 weight: irrelevant (params = element count).
- Where/Equal route-to-free-output: pays a >=900B 30x30 intermediate -> <=18.15.

## INSIGHT (transferable)
⭐ A memory-0 single `Conv(input, W[10,10,k,k])` -> output is at HARD floor when
the rule is a genuine cross-channel spatial dilation that must also emit the
SUBTRACTIVE background channel 0: O=10 (bg is a required, non-copy one-hot
channel), I=10 (cross-channel target->source map breaks any `group` partition),
k forced by the stamp footprint => params = 100*k^2 is irreducible, and ANY
decomposition pays a >=900B 30x30 intermediate that beats mem 0 only below the
existing score. Same family as the GRIDSAMPLE-at-floor BAIL: when the public net
is one mem-0 op whose param count equals the irreducible rule dimension, BAIL.
