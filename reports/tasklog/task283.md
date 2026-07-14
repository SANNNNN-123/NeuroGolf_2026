# task283 — b6afb2da

**Rule:** Fixed 10x10 grid (top-left of the 30x30 canvas) with two solid axis-aligned
GRAY (5) rectangles, each >=3 wide and >=3 tall. Recolour each gray cell by its place
in its own rect: corner -> blue(1), edge non-corner -> yellow(4), interior -> red(2);
non-gray cells stay black(0). Optional transpose applied to both input and output
(orientation-equivariant). Classification is a PURE local 3x3 function of the gray
channel: with O=#gray orthogonal neighbours, D=#gray diagonal neighbours (off-grid =
non-gray), interior=(O4,D4), edge=(O3,D2), corner=(O2,D1); a single linear functional
of (centre, orth-sum, diag-sum) thresholded at >0 isolates each colour. Background
ch0 = a 1x1 copy of input ch0 (keeps the off-grid border all-zero).
**Current:** 18.19 pts, conv3x3+b, mem 0, params 910
**Target tier:** mem-0 single-Conv floor — irreducible (see analysis)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | direct Conv[10,10,3,3] (rebuilt from rule) | floor | 0 | 910 | 18.19 | 200/200 | exact, == current |
| 2 | grouped Conv[10,5,3,3] (group=2) | — | — | 460 | — | — | INFEASIBLE coupling |
| 3 | decompose: slice ch5 10x10 + classify + Pad->Equal | — | >=1800 | small | <17.3 | — | worse (full-canvas plane) |

## Best achieved
18.19 @ mem 0 params 910 — adopted? N (do-not-adopt). Beats prior 18.19? N — equals floor.

## Irreducible-floor analysis
Mem-0 single Conv whose output IS the graph output, so the entire cost is its 910
params; there are no intermediates to shrink.
- GROUPED-CONV escape FAILS: the cross-channel coupling is gray(in ch5) ->
  {bg ch0, blue ch1, red ch2, yellow ch4}. Group=2 blocks {0-4}/{5-9} put in-5 with
  out-{5..9}; group=5 blocks {0,1}{2,3}{4,5}{6,7}{8,9} put in-5 only with out-{4,5}.
  No equal group dividing 10 co-locates input index 5 with output indices 0,1,2,4
  (the coupling spans index 0 and index 5). So grouped-conv cannot carry it.
- DECOMPOSE escape FAILS: output[cell] depends on its 3x3 neighbourhood (corner/edge/
  interior needs orthogonal+diagonal neighbours), so the per-cell/separable-half
  closed-form escape does not apply, and the two rectangles are not row(x)col separable.
  Any op decomposition must feed the full 30x30 ten-channel output via at least one
  full-canvas intermediate. opset-10 Pad accepts only fp16/fp32 (not bool/uint8) and
  Equal only int32/bool, so the cheapest full-canvas feeder is a 1800B fp16 (or 3600B
  int32) 30x30 plane, PLUS the working masks/slice -- far above the 910 params of the
  direct conv (mem 0), and nowhere near the ~671 needed for a +0.3 gain.

## OPEN ANGLES (re-attack backlog)
- None with positive payoff. The only structural lever (block-localised grouped conv)
  is blocked by the in5->out{0,1,2,4} coupling crossing the group boundary; the
  decomposition lever is blocked by neighbourhood-dependence + full-canvas non-separable
  background forcing an fp16 30x30 feeder >= 1800B.

## INSIGHT (transferable)
⭐ A mem-0 Conv[10,10,k,k] is at the HARD floor exactly when the source channel and the
required output channels straddle a group boundary that no equal divisor of 10 can
contain — here input ch5 (gray) must feed output ch0 (bg). The "bg channel = 1x1 copy
of input ch0" sub-pattern is the load-bearing detail that lets a single conv keep the
off-grid border zero (a positive bias on bg would light the whole 30x30 border and
fail the scorer); recovering this is the difference between a passing and failing conv.
