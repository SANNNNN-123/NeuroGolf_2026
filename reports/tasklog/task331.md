# task331 — d364b489

**Rule:** Up to 10 blue(1) seed pixels on a fixed 10×10 grid, pairwise separated by
≥3 in row OR ≥3 in col (so the radius-1 crosses never collide and a neighbour never
lands on another seed's centre). INPUT shows only the blue seeds. OUTPUT stamps a
fixed coloured plus around each seed (r,c): (r−1,c)=red(2), (r+1,c)=cyan(8),
(r,c−1)=orange(7), (r,c+1)=pink(6), centre=blue(1); edge-clipped
when a neighbour falls off-grid. Each lit output cell carries a unique colour by its
offset to the nearest seed; all lit cells are disjoint.
**Current:** 18.19 pts, conv3x3+b, mem 0, params 910.
**Target tier:** MEM-0 single-conv-at-floor — dense 3×3 neighbourhood op that also
emits the subtractive background channel-0.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | full-input 3×3 Conv→fp32 L30, Slice 10×10, Cast u8, Pad99→Equal | B | 5000 | 115 | 16.46 | — | below P |
| 1 | Slice ch1+10×10 in ONE Slice, 3×3 Conv→L10, Cast u8, Pad→Equal | B | 1800 | 37 | 17.48 | 200/200 | below P |
| 2 | cast blue10 + Conv in fp16 | — | — | — | — | — | ORT rejects fp16 Conv under DISABLE_ALL (type-bind error) |

## Best achieved
17.48 @ mem 1800 params 37 — adopted? **N**. Beats prior 18.19? **NO** (−0.71).

## Irreducible-floor analysis
The stored net is a SINGLE `Conv(input[1,10,30,30], W[10,10,3,3]) + bias`, output IS
the graph output ⇒ **mem = 0**, params = 900 + 10 = 910, pts = 25 − ln(910) = 18.19.
Inspected the weight: non-zero only in input channels {0,1} — exactly the minimal
read. ch1 carries the coloured cross (centre + 4 orthogonal offsets → the 5 colours);
ch0 (the one-hot background channel, =1 in-grid / 0 off-grid) gates the bg output
channel-0 (= input ch0 − cross-indicator), which is what makes off-grid correctly
all-zero. So the rule is a GENUINE dense 3×3 cross-channel neighbourhood predicate
that must also emit the subtractive bg ch0 — the canonical MEM-0 single-conv floor.

Why params can't drop below 910 at mem 0: element count is locked at 10·10·3·3 = 900
regardless of the zeros (only inputs 0,1 used). Slicing the input to its 2 used
channels to shrink the weight to [10,2,3,3]=180 costs a 30×30×2 fp32 plane (7200B mem)
— catastrophic. A separable plus (3×1 ⊕ 1×3) needs two 30×30 conv intermediates
(3600B each) before the free Add. Any decomposition that routes the 10-ch one-hot
through a label plane pays a 30×30 uint8 Equal carrier (900B) PLUS the fp32 conv plane,
flooring my best at 1800B = 17.48 < 18.19. exp(25 − 18.49) = 672, so beating P by +0.3
requires mem+params < 672, and the 900B label-carrier alone exceeds that.

## OPEN ANGLES (re-attack backlog)
- None viable. The single-conv mem-0 form dominates every decomposition: the colour
  cross is a true neighbour read (not per-cell / not separable), and the bg ch0 is a
  required non-copy channel, so all three structure-escapes (spatial-copy / separable
  free-output / small-active-canvas) still pay ≥900B and land below 18.19.

## INSIGHT (transferable)
⭐ "Fixed multi-colour STAMP around each seed" (coloured plus / cross) is NOT golfable
when the public net is already a mem-0 single Conv reading exactly input ch0 (bg gate)
+ the seed channel: it is the textbook MEM-0 single-conv-at-floor (dense 3×3 neighbour
op emitting subtractive bg ch0). The colour-per-offset makes it a genuine neighbourhood
read, not per-pixel/separable logic. Distinguish from task371 (single green plus at a
midpoint of two dots, no bg-ch0 conv) which WAS golfable to 16.69 — there the net
wasn't a mem-0 conv and the stamp colour was a constant routed via Where. Here the conv
already buys mem 0 + params 910; ANY intermediate plane (label carrier 900B + fp32 conv
plane) pushes mem above the 672B budget needed to beat +0.3. Also: ORT rejects an fp16
Conv under ORT_DISABLE_ALL (T-binding float16/float error) — keep Conv in fp32.
