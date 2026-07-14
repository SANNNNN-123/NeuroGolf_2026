# task149 — 6773b310

**Rule:** 11x11 "hollywood_squares" board = 3x3 arrangement of 3x3 mini-grids (minisize=3)
separated by single cyan(8) gridlines at rows/cols 3,7. Each mini-block (r,c) gets exactly
1 or 2 pink(6) pixels. Output is a 3x3 grid: blue(1) iff that block holds >=2 pink, else
black(0). Cells outside the 3x3 output are all-zero (no channel set).
**Current:** 19.54 pts, Conv(neg-pad,bias=-1.5)->Neg->Concat->Pad, mem 144, params 91
**Target tier:** A — count-collapse to 9 scalars then sign-one-hot; conv weight (10ch->1, 3x3)
is the hard param floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30 conv->slice 3x3, Greater/Where/Equal one-hot[1,10], Pad uint8 | A | 439 | 118 | 18.68 | - | worse |
| 2 | slice ch6+11x11 (484B) + conv[1,1,3,3] | A | 727 | 40 | 18.36 | - | slice too costly |
| 3 | full conv->slice, 2-ch one-hot + channel-pad (Equal/Where) | A | 295 | 110 | 19.00 | - | better |
| 4 | replicate existing: neg-pad conv crop->3x3, bias=-1.5, Neg+Concat fp32 | A | 144 | 91 | 19.54 | - | = existing |
| 5 | **#4 but Cast conv->fp16 before Neg/Concat; legacy attr-Pad (0 param)** | A | 108 | 91 | 19.707 | 200/200 | **best** |

## Best achieved
19.707 @ mem 108 params 91 — adopted? N (caller decides). Beats prior 19.54? +0.167 → MARGINAL (<+0.3).

## Irreducible-floor analysis
- params 91 = conv weight [1,10,3,3]=90 + bias 1. The op is "select channel 6, sum each 3x3
  block" — a 10ch->1ch 3x3 Conv = 90 elements, irreducible (params count elements, dtype free).
  Any single-channel pre-slice to dodge the 10-ch weight costs a >=484B fp32 plane (slice
  [1,1,11,11]), strictly worse than 90 params + the 36B conv plane.
- mem 108 = conv out fp32 36 (Conv FORCES fp32 out, irreducible) + cntH fp16 18 + negH fp16 18
  + oh[1,2,3,3] fp16 36. The fp16 cast of the tiny downstream planes is the ONLY win over the
  existing net (saved 36B mem, 144->108). Negative conv pads ([0,0,-19,-19]) crop directly to
  3x3 (no 7x7=196B plane). bias=-1.5 folds the >=2 threshold; Neg+Concat builds the 2-colour
  one-hot by SIGN (out>0 scoring), so no Equal/Where/threshold consts. Legacy opset-10 attr-Pad
  adds the 8 trailing channels + spatial border at 0 params (opset-11 pads-as-input would cost 8).
- Total floor ~199 -> 19.71. To reach 19.84 (total<175) you must cut 24+ more from an already
  near-minimal graph; the 90-param weight + 36B fp32 conv (=126) plus a >=2-channel output
  one-hot leave no slack. Structurally at-floor.

## OPEN ANGLES (re-attack backlog)
- 2-channel conv weight [2,10,3,3] (ch0=-sum, ch1=+sum, bias [1.5,-1.5]) removes Neg+Concat
  (mem -> ~72) but DOUBLES params to 182 — net worse. Confirmed dead.
- No MatMul/Reduce path avoids the 90-elem channel-select-and-block-sum cost.

## INSIGHT (transferable)
⭐ After an unavoidable fp32 entry plane (Conv output is forced fp32), CAST it to fp16 and do
all downstream sign/Neg/Concat on half-width planes — drops a tiny-plane net's mem ~25% for free
(here 144->108, +0.17). ⭐ Legacy opset-10 Pad takes pads/value as ATTRIBUTES (0 params) vs the
opset-11 pads-as-input form (an 8-elem initializer) — use attr-Pad for fixed pads to save params.
⭐ Negative Conv pads crop the conv output spatially (here -19 -> exactly 3x3), avoiding a strided
7x7=196B intermediate; fold a count threshold into the conv bias and emit a 2-colour one-hot by
SIGN (out>0) with no Equal/Where. This task is at its structural floor: count-collapse + 10ch->1
3x3 conv (90 params) caps it ~19.7.
