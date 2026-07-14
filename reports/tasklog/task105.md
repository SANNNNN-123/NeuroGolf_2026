# task105 — 4612dd53

**Rule:** A rectangular OUTLINE (perimeter of a bbox) is drawn, optionally with ONE full
interior "cutline" — either a horizontal row (`horiz`) or a vertical column (`vert`) spanning
the box interior. Every figure cell is colored blue(1) or red(2) (each red w.p. ~1/4). In the
INPUT only the blue cells appear; the red cells are erased to background(0). In the OUTPUT
every figure cell appears (blue unchanged, erased cells become red). So output = input with
every figure-cell that is currently empty repainted red(2). Verified bbox(blue)==bbox(figure)
(corners always part of the outline), so the box geometry is fully recoverable from blue.
Cutline orientation recovered from per-row vs per-col interior-blue COUNT (Rmax vs Cmax).

**Current:** 16.14 pts (prior), method = unknown public net.
**Target tier:** A (separable row⊗col rectangle + cutline routed into the FREE Where output;
not S because output color set is fixed but the figure is a 2-D structure, not a pure spatial
copy of input cells).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bbox-as-mask (tri-matmul) + perim + cutline, W=14 sq, fp16 planes | A | 8901 | 450 | 15.86 | 200/200 | works, too heavy |
| 2 | + cutline counts via MatMul (drop interior/int_blue WxW); 2 outer-prod figure | A | 6717 | 450 | 16.12 | — | leaner |
| 3 | + drop bg slice (red=figure∧¬blue); single Less for notblue | A | 6129 | 444 | 16.21 | — | |
| 4 | + rectangular canvas HR=13×WC=11 (figure spans rows1-12,cols2-10) | A | 5088 | 634 | 16.35 | — | |
| 5 | + CumSum prefix/suffix-OR (drop 4 triangular inits, 580 params) | A | 5089 | 58 | **16.45** | 199-200/200 | ADOPT-CANDIDATE |

## Best achieved
16.45 @ mem 5089 params 58 — beats prior 16.14 by +0.31 (≥+0.3 ✓). Fresh: 5/6 runs 200/200,
one 199/200; 500/500 clean on a separate run; aggregate ~1/1200 fail rate, ALL the rare
single-interior-blue-cell orientation ambiguity (structurally undecidable, see below).

## Irreducible-floor analysis
Dominant intermediates: the two 30×30 planes for the final `Where` cond — red30 (uint8, 900B
from Pad) + cond (bool, 900B from Cast). ORT `Pad` rejects bool, `Where` requires bool, so the
u8-pad→bool-cast pair (1800B) is the irreducible carrier for routing the W×W mask into the
30×30 output. Next: blue_f32 (572B, the one fp32 channel-1 slice; fp32 because CumSum/ReduceMax
reject fp16) and 4 fp16 13×11 planes (blue, fig_a, fig_b, fig_s = 1144B). Everything else is
sub-150B 1-D vectors.

## OPEN ANGLES (re-attack backlog)
- Resolve the single-interior-cell ambiguity (0.08% fail): a lone interior blue cell gives NO
  input signal for H vs V cutline (generator prior is ~50/50). Verified truly ambiguous — no
  positional/geometric tell. Likely unrecoverable; would need a guess that beats 50%.
- Drop the fp16 `blue` plane (286B) by running the cutline MatMuls + notblue in fp32 off
  blue_f32 (needs col_inner_T/row_inner_T in fp32). ~286B → ~16.50.
- Tighter canvas via col-offset slice (cols 2:11, WC=9) with shifted final Pad — saves ~26%
  on each W×W plane but complicates the pad placement; modest (~few hundred B).

## INSIGHT (transferable)
⭐ "Restore the erased (off-color) cells of a separable figure" = recover the figure geometry
from the SURVIVING color's bbox + a count-based cutline detector, then route ONLY the
restored cells into the FREE `Where` (red_oh, input) — you never rebuild blue/bg, you just
flip the empty figure-cells. ⭐ CumSum(reverse=1)>0 is a drop-in prefix/suffix-OR that REPLACES
the two triangular-matrix initializers of the task070 bbox-as-mask idiom (saved 580 params,
~0.1 pt) — but CumSum rejects fp16, so feed it the fp32 occupancy vector. ⭐ Per-row/col
interior counts come from MatMul(plane, vec) (contract one axis) — no interior WxW product
plane needed; orientation of a unique full line = compare max-row-count vs max-col-count.
