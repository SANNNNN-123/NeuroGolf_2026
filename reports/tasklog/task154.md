# task154 — fold-outside-across-two-red-edges

**Rule:** Fold the outside gray pixels across the two red "gripper" box edges into the
box interior. Reflection across an edge row `e` is `out = 2*e - in` (midpoint == edge).
Orientation (transpose) recoverable FREE from box aspect ratio: generator guarantees
tall(8-9) > wide(5-7), so box_width > box_height ⇔ xpose.

**Target tier:** B reached. pts 14.67 → **15.85** (+1.18). mem 9311, params 71.
fresh 200/200 + 500/500. Fully closed-form, no banned ops, no data-dependent shape.

**Dominant intermediate:** [1,1,15,15] reflection-matrix/matmul planes (bool 225B each).
Irreducible: data-dependent reflection across two red edges needs two boolean permutation
MatMuls (row + col axis) plus an orientation select. Colour-index label Pad to 30×30 uint8.

**OPEN ANGLES:** shrink working canvas below 15×15 (low payoff, reflection uses full 0..14);
collapse the two Equal-Or reflection-matrix builds into one banded matrix.

**INSIGHT:** task112 4-fold-reflection idiom generalized to TWO integer edge-axes + an
orientation branch. (1) edge-row reflection = `Equal(2*e - src_arange, out_arange)` OR'd
over both edges; (2) orientation free from box aspect ratio via one Where.
CRITICAL GOTCHA: off-grid cells (rows/cols ≥ actual grid extent) must be ALL-channel-False,
NOT ch0-True — pad the colour-index label with sentinel >9 (99) before final Equal(L,arange)
or every off-grid cell wrongly reports background and the [1,10,30,30] compare fails 0/N.
