# task037 - 1f876c06

## Human-readable rule

Fixed 10x10 active grid, output padded to the normal 30x30 one-hot harness shape.
Each nonzero colour appears as exactly two endpoint cells.  The output fills the
45-degree diagonal segment between those endpoints, inclusive, using that same
colour.  Valid directions are down-right and down-left; segment lengths are 3..7.
Colours are distinct per generated line and generated lines do not overlap.

Stored examples:

- train0: 3 colours, 6 input endpoint cells -> 13 filled output cells.
- train1: 4 colours, 8 endpoints -> 18 filled cells.
- train2: 4 colours, 8 endpoints -> 19 filled cells.
- test0: 5 colours, 10 endpoints -> 24 filled cells.
- stored arc-gen includes 1..6 generated segments; this contradicts the old
  "3..6 segments" wording, which describes attempts before rejected short lines
  are skipped, not the final accepted count.

Confidence: verified.  A simple Python oracle ("for each colour with two
endpoints, fill the straight 45-degree segment between them") matched all 266
stored cases and 500 fresh generated cases.

## Current source/live state

Current source: `src/custom/task037.py` is an exact source reconstruction of the
live ONNX graph, not the older Conv candidate.

Manifest / task-only measurement:

| item | value |
|---|---:|
| stored pass/fail | 266 / 0 |
| memory | 4614 |
| params | 132 |
| points | 16.534942563004293 |
| method | custom:task037 |

Live `networks/task037.onnx` matches the same metrics: filesize 2876, memory
4614, params 132, pass 266, fail 0.

Fresh checks:

- `reports/scripts/fresh_verify.py 037 src/custom/task037.py 300`: incumbent
  fail 0, candidate fail 0, candidate != incumbent 0.
- `src.genverify.fresh_pass(37, 200)`: 200/200.

## Current mechanism

The incumbent no longer scans/fills diagonals directly.  It reconstructs a small
set of segment descriptors from the input one-hot planes:

1. Reduce over width/height to find row/column occupancy per colour.
2. ArgMax gives first row, last row, and minimum column per colour.
3. Gather the colour at `(first_row, min_col)` to infer diagonal orientation:
   if that top/min cell is an endpoint, use the min column; otherwise use max
   column.  This distinguishes down-right from down-left.
4. TopK selects the up-to-six present non-background colours by smallest row
   span; spans greater than 6 are rejected as absent/background.
5. For offsets 0..6, compute linear cell indices along each selected segment.
6. Scatter those colour ids into a 10x10 uint8 label plane.
7. Pad to 30x30 with sentinel 255 and emit `Equal(label, channel_ids)`.

This is a scalar endpoint/scatter solution.  It exploits generator bounds
(max six accepted lines, length <=7, fixed 10x10 active grid, distinct colours)
instead of building per-colour diagonal reachability planes.

## Cost anatomy

Scorer tensor memory from current source:

| component | tensors | bytes | semantic job |
|---|---:|---:|---|
| full-grid row/column scans | `row_has`, `col_has` | 2400 | detect which rows/cols contain each colour over the 30x30 harness input |
| final label carrier | `scalar_big` | 900 | 30x30 uint8 label map used by free final `Equal` output |
| scatter index/update path | `linear_idx`, `linear_cols`, `linear_idx_u8`, `updates*`, `valid_offsets_raw`, `offset_delta` | ~420 | write at most 6 x 7 segment cells into a flat 10x10 label map |
| endpoint coordinate extraction | `top_min_idx`, `first_row_raw`, `last_row_raw`, `min_col_raw`, casts/gathers | ~520 | find endpoints and orientation per colour |
| selected segment descriptors | `slot_idx`, `first_row`, `last_offset`, `min_col`, `main_orient`, colour ids | ~130 | keep only reachable colours/segments |
| 10x10 label intermediates | `scalar_flat`, `scalar_small` | 200 | compact carrier before padding |
| small scalars/rest | misc bool/u8/f16 tensors | ~44 | thresholds, constants, shape glue |

Dominant cost: the two full 30-wide occupancy planes (`row_has`, `col_has`) at
2400 bytes.  The next irreducible-looking cost is the 900-byte padded label map.
The old Conv-output floor analysis is no longer valid for the current graph.

Params: 132 initializer elements, dominated by `base_grid` (100), shape/pad
constants, `channel_ids`, and the 0..6 offset vector.  There are no Conv kernels
in the current graph.

## Prior notes challenged

Still valid:

- The semantic rule is diagonal endpoint fill, not connectivity/flood fill.
- The bounded diagonal Conv mechanism is a real source-owned way to solve the
  task and did beat the older 14.12 baseline.
- Length bound <=7 and distinct non-overlapping colours are useful generator
  facts.

Contradicted or stale:

- "Current: 14.12 pts, mem 45000, params 8326" is stale.  Current is 16.5349,
  mem 4614, params 132.
- "Best achieved 14.84 @ mem 25700 params 146" is stale.  That Conv graph is now
  worse than incumbent by 21086 memory and 14 params.
- "3..6 diagonal segments" is not true for final generated examples.  Generator
  samples 3..6 attempts but may reject short/overlapping attempts; in 2000 fresh
  examples the accepted segment count ranged 1..6.
- "Irreducible floor bound by four fp16 7x7 Conv outputs" applies only to the old
  Conv graph, not to the current scalar endpoint/scatter graph.

## Mechanism probes

### 1. Direction-separable diagonal Conv

Hypothesis: bounded diagonal prefix/suffix Conv remains a good replacement for
the incumbent.

Expected payoff: old note claimed +0.72 versus a pre-existing 14.12 graph.  If
still competitive, it would need to approach or beat 4614 memory / 132 params.

Proof test: measured the historical source-owned Conv implementation from commit
`024b949` in memory, without editing current source.

Result:

- stored: ok, pass 266, fail 0, memory 25700, params 146, points 14.840088870914864.
- fresh: 200/200.

Kill condition met for adoption now: correct but much more expensive than the
current scalar endpoint/scatter graph.  Keep the Conv lesson as transferable for
tasks where endpoints cannot be reduced to compact descriptors, but do not
adopt it for task037.

### 2. Bounded crop before scan / direct active-area scans

Hypothesis: the current 2400-byte row/column occupancy cost could be reduced by
scanning only the fixed 10x10 active grid.

Expected payoff if free: shrink `row_has` and `col_has` from 1200+1200 to
400+400, saving 1600 memory and reaching roughly memory 3014 / params 132
(about 16.95 points).

Proof test: scorer-side anatomy plus ONNX operator accounting.  The input shape
is fixed [1,10,30,30].  To make ReduceMax produce 10-wide occupancy, a source
graph must first materialize an active crop.  Cropping all channels to 10x10
would add a [1,10,10,10] float32 Slice tensor (4000 bytes), which is larger than
the 1600-byte reduction saving.  Cropping after ReduceMax cannot remove the
already-materialized full `row_has`/`col_has` tensors and adds extra tensors.

Kill condition met for a naive crop candidate: no net scorer win unless a
non-materializing crop/range-limited reduction trick is found.

## Next exact experiment

Look for a way to avoid one of `row_has` or `col_has`, not merely crop it.  The
current graph needs both because it extracts first/last rows and a minimum
column/orientation per colour.  A possible next probe is a descriptor route based
on sparse endpoint linear indices per colour: if min/max linear index plus row
span can infer orientation and start column without `col_has`, expected payoff
is up to 1200 bytes; kill it if it requires a dense [colour,100] index carrier or
more than one extra 30-wide plane.

Reusable insight: for bounded, distinct-colour endpoint lines, scalar endpoint
descriptor reconstruction can dominate direction-separable Conv by an order of
magnitude in memory.  Conv remains useful for reachability-style tasks, but when
the generator guarantees exactly two same-colour endpoints, first/last endpoint
coordinates plus a short offset scatter are the stronger family.

## S8 (2026-07-02) — pow2-log extremes (+0.280) ADOPTED, div 0
row_has/col_has occupancy bands + 3 ArgMax → 3 free-input pow2-weight einsums (≤1 px/row per colour ⇒ exact). TRAP: ArgMax(select_last_index=1) on all-zero returns 29, and ch0 overflows pow2 — gated by ReduceSum(input)==2 (2-endpoint diagonals only). 3384+201 vs 4614+132 → 16.535→16.815.
LESSON: REDUCE_ONLY converts ONLY when ≤1 pixel per reduced line (pow2 exact) AND consumers are index/extreme extraction; occupancy (max) over multi-pixel colours = floor.
