# task327 — d13f3404 (diagonal down-right ray propagation)

**Rule:** Input is a 3x3 grid with up to 3 coloured pixels (colours 1..9), each on
a DISTINCT diagonal `c-r` (the generator skips repeated diagonals). The 6x6 output
draws from each coloured pixel at (r,c) a down-right 45 degree ray
`output[r+idx][c+idx]=colour` for idx in `range(2*size-max(r,c))` (i.e. until it
leaves the 6x6 grid). Background stays 0; cells outside the 6x6 grid are empty.
size=3 is fixed (generate() takes no varying args), so the active region is a
fixed 3x3 input -> 6x6 output — no symbolic dims.

**Current (prior):** 16.07 pts (public net).
**Target tier:** A (diagonal prefix-sum = one bounded Conv; routed to free bool out).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice [1,10,6,6] -> colidx Conv -> 6x6 diag Conv -> Pad+Equal | A | 2664 | 71 | 17.09 | 200/200 | WIN +1.02 |
| 2 | slice the 3x3 SOURCE corner only ([1,10,3,3]=360B), Pad colidx to 6x6 | A | **1620** | 80 | **17.56** | **200/200** | WIN +1.49 |

## Best achieved
**17.56 @ mem 1620 params 80 — fresh 200/200, stored 265/265.** Beats prior 16.07
by **+1.49**. Adopted? N (build-only).

## Irreducible-floor analysis
mem 1620 dominated by the 900B uint8 [1,1,30,30] padded label `L` feeding the FREE
final Equal (irreducible — Equal must write all 30x30). Remaining ~720B: the
[1,10,3,3] input slice (360B fp32, the entry colour read), the two [1,1,6,6] f32
working planes (Vin 144B, Ldiag 144B), the [1,1,6,6] uint8 Lu (36B). All working
planes are already on the tiny 6x6 / 3x3 active canvas.

## OPEN ANGLES
- The 900B padded-L plane is the floor for any "label -> Equal" route; a pure
  spatial-COPY (Tier S, mem 0) is impossible here because each output cell is a
  data-dependent colour from an up-left source, not a fixed permutation of inputs.
- Could fold the colour-index Conv and diagonal Conv into one Conv on the sliced
  input (depthwise diagonal then weight-sum) — same byte count, more ops, no gain.

## INSIGHT (transferable)
⭐ A "draw a 45 degree ray from each marker" task where each diagonal carries AT MOST
ONE source is a DIAGONAL PREFIX-SUM, not a detection/flood wall: `out[R][C] =
sum_{j>=0} V[R-j][C-j]` = ONE Conv with a diagonal-of-ones kernel (size = grid
side) and top/left padding = side-1. The single-source-per-diagonal guarantee makes
the sum reproduce the colour exactly (no double-count), so it runs on the collapsed
1-channel colour-index plane and routes straight into the free bool output via
Pad(sentinel)+Equal. Slice to the SOURCE region (3x3) not the output region (6x6)
before the colour Conv, then Pad the colour-index plane up — quarters the entry slice.
