# task022 — 137eaa0f

**Rule:** An 11x11 input has 4 well-separated (>=4 apart) "centre" cells painted GRAY(5)
(or left background when their colour is 0). Around each centre, the 8 non-centre cells of
a 3x3 window are partly painted with that centre's colour; across the 4 centres the 8 output
positions cover the 3x3-minus-centre exactly once. The OUTPUT is the 3x3 reconstruction:
centre = GRAY(5), and output[1+dr][1+dc] = the colour sitting at offset (dr,dc) from some
centre. Coloured pixels are never gray (random_colors with 5->0). Because the four windows
are pairwise disjoint, this is exactly the correlation
`out[1+dr][1+dc] = sum_{i,j} G[i,j]*colf[i+dr,j+dc]` where G=(input==gray),
colf=sum_{k!=5} k*input_k. Verified 0/3000 fresh.
**Current (prior):** 15.17 pts
**Target tier:** A (closed-form correlation, no flood-fill / no global argmax).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 9-window Slice stack + Mul + ReduceSum, fp32, fp32 30x30 L pad | A | 24268 | 105 | 14.90 | 266pass | works, below P |
| 2 | + fp16 work planes, uint8 L pad, Conv-full colf | A | 13067 | 108 | 15.51 | 266 | beats P |
| 3 | per-window mul+reduce (drop 9-ch stack+prod) | A | 10907 | 108 | 15.69 | 266 | better |
| 4 | im2col 9 shifts via ONE Conv + MatMul correlate | A | 10793 | 147 | 15.70 | 266 | marginal |
| 5 | Conv shifts + Mul/ReduceSum (drop reshape copy) | A | 10551 | 143 | 15.72 | 266 | better |
| 6 | 8 NON-centre shifts only, insert GRAY centre via Concat | A | 9985 | 129 | 15.78 | 200/200 | ADOPT |

## Best achieved
15.78 @ mem 9985 params 129 — beats prior 15.17 by +0.61. Fresh 200/200 isolated.

## Irreducible-floor analysis
Dominant intermediate = the fp32 `colf` entry plane [1,1,30,30] = 3600B. It is the
channel-collapse (1x1 Conv, sum_k k*input_k) run on the FREE [1,10,30,30] input, so the
output is full-canvas. Cropping the 10 input channels to 11x11 first to shrink it would cost
4840B (>3600) — Conv-on-free + crop is cheaper. This is the "pay one fp32 entry plane" floor.
Everything after is fp16: shifted/prod [1,8,11,11] = 1936B each; Lu8 30x30 = 900B.

## OPEN ANGLES
- Combine channel-collapse + 8-shift im2col into ONE Conv(10->8, 3x3) on the free input —
  blocked because the output would be [1,8,30,30] full-canvas (16200B); only worth it if a
  data-dependent spatial crop of the input to ~11x11 were free (it is not).
- Drop the colf crop+cast pair (484+242B) by Conv'ing the 11x11 region directly — needs a
  cheap 11x11 multi-channel slice (>=3600B), no net win.
- Lower bound ~ 3600 (entry) + ~1936 (one fp16 work plane) + 900 (Lu8) ~ 6.5KB -> ~16.3 pts
  ceiling; current 9985 leaves the second fp16 work plane (prod) as the only further target.

## INSIGHT (transferable)
⭐ A "stamp the 3x3 neighbourhoods of K data-dependent marker cells into a fixed K-size
output" task is a CLOSED-FORM CORRELATION, not a scatter: out[offset] =
sum over grid of marker_mask * shift(value_plane, offset). Implement the offset im2col with
ONE Conv whose output channels are one-hot shift kernels (K[o,dr+1,dc+1]=1, pad=1), then
multiply by the marker mask (broadcast) and ReduceSum over space — no Gather/scatter, no
data-dependent indexing. Fold the value-weight into that same Conv kernel ONLY if the output
stays small (here it would have gone full-canvas, so keep the channel-collapse separate).
Drop any output cell that is later overwritten (centre -> GRAY) to shave a Conv channel.
