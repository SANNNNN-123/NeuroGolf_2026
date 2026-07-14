# task55 — 272f95fa

**Rule:** The grid is partitioned into a 3x3 arrangement of variable-size blocks (block sizes random 1..10) by two full horizontal cyan(8) lines and two full vertical cyan(8) lines (all four always present). The input already contains the cyan cross. Output keeps the input cyan lines and fills 5 blocks (a plus shape) with FIXED colours by (rowband,colband): (0,1)=red2, (1,0)=yellow4, (1,1)=magenta6, (1,2)=green3, (2,1)=blue1. Corner blocks + off-grid stay background 0.

**Current:** 14.83 pts (prior net)
**Target tier:** A — separable rows×cols band partition + fixed colour LUT, routed into FREE BOOL output; no flood-fill, no NonZero.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | CumSum band + double-MatMul LUT, fp32 planes, occ ReduceSum gate | A | 18900 | 51 | 15.15 | — | passes 263/263 |
| 2 | fp16 downstream + separable rowin/colin gate | A | 9660 | 46 | 15.82 | — | passes |
| 3 | drop uint8 cast, fp16 Equal directly into output | A | 8760 | 46 | 15.92 | 200/200 | FINAL |
| 4 | QLinearMatMul uint8 LUT path (scale=1/zp=0), uint8 label overlays | A | 5790 | 48 | **16.33** | 1000/1000 | **ADOPTED, LB 7171.00** |

## Best achieved
16.33 @ mem 5790 params 48 — adopted as `custom:task055`. Beats prior live 15.92 by **+0.411 local**.
Submission `54127206` completed with **publicScore 7171.00**, up from 7170.59.

## Irreducible-floor analysis
The old fp16 version was NOT at floor. Three [1,1,30,30] fp16 value planes
(Lband, Lg, L = 1800 B each = 5400) dominated, but the LUT selection is integer-valued.
Replacing the two fp16 MatMuls with `QLinearMatMul` on uint8 one-hots and a uint8 LUT
(all scales=1, zero-points=0) keeps the same exact bilinear selection while shrinking
the three full label planes to 900 B each. Current dominant full planes are:
Lband/Lg/L uint8 + isline/ingrid bool = 5 × 900 B. The line and off-grid overrides still
materialise real 2-D masks, but the label dtype is no longer forced to fp16.

## OPEN ANGLES (re-attack backlog)
- Fold the line override into the double-MatMul by augmenting the band one-hots with hline/vline basis vectors. Blocked: a line cell still carries its band one-hot, so the bilinear term adds LUT[k][cb] that cannot be cancelled to a clean 8 — would need the band term zeroed, which the band one-hot does not provide. Could try a 5-component row/col factor where the line component DOMINATES additively (e.g. line weight 1000) then read back by magnitude bands in the final Equal threshold, collapsing Lg into Lband (~−1800 B, ~+0.2 pts).
- Merge offgrid+line into a single override plane (disjoint regions) to drop one Where — but building `8*isline + 10*offgrid` costs as many full planes as it saves.

## 2026-06-30 category-LUT rewrite adopted

The old open angle was too pessimistic because the row/col one-hots do not need
to keep the original band component active on separator or off-grid cells.  Build
five mutually-exclusive row categories and five column categories instead:

- row/col bands `0..2`, gated by `in-grid && not separator`;
- separator line category;
- off-grid category.

Then a single `5x5` uint8 `QLinearMatMul` LUT emits the normal block colour,
cyan line colour `8`, or off-grid sentinel `10` directly.  This removes the
full-canvas `isline`, `Lg`, `ingrid`, and second overlay `Where` path, replacing
them with only small row/column category vectors.

Adopted result:

- previous: `points=16.32877535559045`, `memory=5790`, `params=48`;
- new: `points=16.963426590292688`, `memory=3030`, `params=62`, stored `263/263`;
- delta: `+0.634651` points, `-2760` memory, `+14` params.

Reusable mechanism: when a separable row×column LUT has full-canvas override
planes for separator/off-grid/sentinel regions, first try promoting those
regions to explicit row/column categories and folding the overrides into the
LUT.  This is profitable when the override predicate is row-separable,
column-separable, or an OR/product of the two.

## INSIGHT (transferable)
A data-dependent rows×cols block grid drawn by FULL separator lines is a fully separable partition: the per-axis band index is the EXCLUSIVE CumSum of the line indicator sampled from the line colour along the first column/row (which is never itself a separator). A non-rank-1 (rowband,colband)->colour map is then the double-MatMul LUT idiom (Ronehot @ LUT3x3 @ Conehot), and preserving the input separator lines is free: just overlay the line colour (8) where the line indicator is set before the final Equal. No flood-fill, no NonZero — closed-form Tier A. ⭐ Reusable for any "fill specific cells of a line-delimited grid with fixed colours" task.
⭐ New transferable mechanism: if a MatMul/MatMul LUT path selects small integer labels from
one-hot factors and then feeds `Equal(label, channel_ids)`, try `QLinearMatMul` with uint8
inputs/LUT and scale=1/zp=0. It preserves exact integer selection and can halve every full
label plane versus fp16. This is directly relevant to other row×col LUT or packed-label
builders, but not to true weighted sums that exceed uint8 or need fractional values.
⭐ Category-augmented row/column LUT: line/off-grid overrides that look like
post-LUT full-canvas `Where` planes can often be folded into the LUT by making
the row/column factors mutually exclusive over `{bands, separator, off-grid}`.
This turns multiple 30x30 override planes into small 1-D category vectors.
