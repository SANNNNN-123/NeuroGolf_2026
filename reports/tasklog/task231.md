# task231 — 963e52fc

**Rule:** Input is a 5xW grid (W in 6..10). A periodic stripe pattern fills `tall`
(1..2) consecutive rows starting at `offset` (1..2); horizontally the pattern repeats
with period `wide` (2..3) and spans the whole width: cell(r,c)=colors[(r%tall)*wide + c%wide]
within the pattern rows. OUTPUT is 5x(2W): the SAME periodic pattern extended to double
width. Since the input already holds a full period, out(r,c)=input(r, c mod wide).
**Current:** 14.59 pts, gen:biohack_new, mem 33113, params 101
**Target tier:** A (closed-form column gather; the whole output is one Gather, FREE)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | output = Gather(input, src, axis=3), src = period-tiled col index | A | 2555 | 193 | 17.08 | 500/500 | PASS |

## Best achieved
17.08 @ mem 2555 params 193 — adopted? N (build-only). Beats prior 14.59? Y (+2.49).

## Irreducible-floor analysis
The final Gather IS the output (free). Dominant intermediate is the row-contraction
MatMul `rowwt` [1,10,1,30] fp32 = 1200B, used to form an injective per-column colour
signature for period (2 vs 3) detection. Everything else is [30] / [1,1,1,30] vectors.
Could shave by detecting period from only 2 specific column comparisons, but slicing
two [1,10,30,1] columns is comparably sized; 1200B already gives ~17.1.

## OPEN ANGLES (re-attack backlog)
- Replace the 1200B MatMul signature with a cheaper period probe: e.g. compare only
  columns (0,2) and (1,3) via two narrow column slices — might reach ~17.5 if it
  stays under ~600B. Risk: row-arrangement collisions need the full per-row weights.

## INSIGHT (transferable)
⭐ "Horizontally tile a periodic pattern to 2x width" is Tier-A closed-form, NOT a
re-synthesis task: out(r,c)=input(r, c mod period), so the ENTIRE output is a single
`Gather(input, srcIdx, axis=3)` (free). Pad columns (c>=2W) are zeroed for free by
pointing srcIdx at a guaranteed-empty pad column (29); rows>=H are zeroed for free
because Gather copies whole columns and the padded input's extra rows are all-zero.
Period (2 vs 3) is recovered from a per-column injective signature
colsig[c]=sum_r rowW[r]*colorindex(r,c) (distinct row weights => column-tuple
recoverable) and "colsig[c]==colsig[c+2] on every occupied pair". A genuinely period-3
pattern only fakes period-2 when each row is constant, in which case period-2 tiling
reproduces the identical grid -> the mis-detection is harmless.
2W extent recovered offset-free: keep[c]=colocc[floor(c/2)] (Gather by const
[0,0,1,1,...]) since c<2W iff floor(c/2)<W.
