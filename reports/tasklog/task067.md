# task067 — 2dee498d

**Rule:** Generator builds an input grid of (size) rows x (3*size) cols, size in 2..5,
as three side-by-side size x size blocks: block0 = colors[r][c], block1 = colors
optionally vertically flipped, block2 = colors[r][c]. The OUTPUT is the size x size
grid colors[r][c], i.e. exactly block0 (== block2) — the LEFT size columns of the
input cropped to size rows. Pure copy/crop. Off-grid cells are all-zero; in-grid
cells (incl. colour 0 -> channel 0 = 1) always occupy their column.
**Current:** 25.000 pts, custom self-Einsum gate, mem 0, params 0
**Target tier:** S (pure spatial copy/crop, output is a column-subset of input)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: ReduceMax col_occ + cumsum-Conv + Greater + Where | S | 282 | 33 | 19.25 | — | baseline |
| 1 | scalar size^2 from ReduceSum(input,[1,2,3])/3; keep c iff c^2<size^2 via const squared col-ramp; Where | S | 38 | 32 | 20.75 | 200/200 | ADOPT-worthy |
| 2 | one-node self-Einsum: `bkrc,blcd->bkrc` gates each output column c by active row c | S+ | 0 | 0 | 25.00 | 1500/1500 | ADOPTED |

## Best achieved
25.00 @ mem 0 params 0 — beats prior local 21.15 by +3.85 and the old 20.75 scalar-gate by +4.25.

## 2026-07-05 — 25pt self-Einsum breakthrough

The old mask-builder was not a true floor.  It materialized `size` as a scalar and
then built a column mask.  The zero-cost route is to use the input itself as the
column gate inside one `Einsum` node:

`output[b,k,r,c] = input[b,k,r,c] * sum_{l,d} input[b,l,c,d]`

ONNX equation: `bkrc,blcd->bkrc`, with both operands wired to the graph input.

Why it works:

- row `c` is nonzero iff `c < size`, because the active input has exactly `size`
  rows;
- for `c < size`, `sum_{l,d} input[b,l,c,d] = 3*size > 0`;
- for `c >= size`, that sum is zero;
- multiplying by `input[b,k,r,c]` preserves the left block and zeroes every
  column from `size` onward, including the middle/third copies;
- rows `r >= size` were already zero in `input`.

Cost:

- one node: `Einsum(input, input) -> output`;
- no initializers;
- no counted intermediates because the only node output is graph output;
- measured `mem=0`, `params=0`, `points=25.0`.

Verification:

- stored: `reports/scripts/measure_task.py 067` => 266/266, mem 0, params 0, pts 25.0;
- fresh: `reports/scripts/fresh_verify.py 067` => 1500/1500, incumbent fail 0;
- artifact: `python -m src.harness networks/task067.onnx 067` => 266/266, mem 0, params 0, pts 25.0.

## Irreducible-floor analysis
Superseded.  The 30B bool mask was only a floor for explicit mask-building.  A
self-Einsum can create the data-dependent gate inside the final output op, avoiding
all counted carriers.

## OPEN ANGLES (re-attack backlog)
- Mine other repeated/cropped tasks for one-node self-Einsum gates where a spatial
  index can be reused as a row/column activity probe.

## 2026-07-05 — transfer sweep across all tasks

Ran two bounded sweeps for direct full-output replacement, without adopting any
other task:

- fixed identity-left self-gates `bkrc,b...->bkrc` over axis-activity variants;
- broader two-input self-Einsum axis remaps `b... , b... -> bkrc` where output
  `k,r,c` labels can come from either input operand.

Exact 25-point full replacements found only:

- task067: true axis-activity gate, current 25.0;
- task179/task241: transpose-like axis relabeling (`bkcr,...->bkrc`), both already
  25.0 via one `Transpose` node.

No non-25 task had a positive direct-replacement hit in this equation family.
Transfer conclusion: this is a high-value mechanism, but the exact one-node
frontier class is narrow.  For non-25 tasks it should be used as a *subgraph
deletion pattern* when a costly row/column activity carrier is only consumed by
an output-only `Einsum`/`Where`, not as a blind whole-task replacement.

## INSIGHT (transferable)
⭐ If a crop boundary can be recovered from the same input as an axis activity
predicate, a one-node self-Einsum can gate the final output without any explicit
mask tensor or constants.  The general form is:

`output[..., i, j] = input[..., i, j] * sum(input over an axis slice keyed by j)`

For task067, using output column `c` as a row index in the second operand gives
`row_active[c]`, deleting the prior scalar-size and 30B bool-mask machinery.

⭐ For a "crop the input to its first `size` columns/rows" task where the grid width
is a known multiple of size (here 3*size) and EVERY in-grid cell occupies its row/col
(channel-0 set for colour-0), recover size as a pure SCALAR: total = ReduceSum(input,
[1,2,3]) = K*size^2; size^2 = total/K. Then keep column c iff c^2 < size^2 by comparing
a CONSTANT squared ramp const [0,1,4,...] against the scalar — NO per-column occupancy
plane and NO cumsum Conv. This collapses the public "ReduceMax-occ + cumsum-Conv +
threshold" idiom (two [1,1,1,30] fp32 planes, 240B) down to two scalars + one 30B bool
mask (282 -> 38B, 19.25 -> 20.75, +1.50). Comparing c^2<size^2 avoids a Sqrt entirely.
