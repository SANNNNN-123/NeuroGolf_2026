# task013 — 0a938d79 (alternating periodic stripes from two seeds)

**Rule:** Grid W×H (W∈20..30, H∈6..12). Two seed pixels at columns `start` and
`start+sep+1` (period p=sep+1∈2..6), each in row 0 or H-1, colours c0,c1∈1..9.
Output paints FULL vertical stripes at cols start, start+p, start+2p,… (<W),
alternating c0,c1,c0,…  If `xpose`, the whole grid (in & out) is transposed →
stripes become full rows. Orientation recoverable: xpose=1 IFF both seed columns
lie in {0,W-1}. Closed-form, fully separable: recover period-axis colour vector
pvec[30] from two no-pad colour-weighted profile Convs, alternate by (t-first)%2p,
gate to the in-grid rect, route the 10-ch expansion into the FREE bool output.

**Current (prior):** 15.77 pts, closed-form separable net, mem 10101, params 65
(ledger 'memory' was STALE — real scored mem was 10101, not the ~1800 in docstring).
**Target tier:** A (separable row⊗col into free output; the orientation swap forces
one fp16 combine plane, see floor analysis).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior net (3 full fp16/bool planes + 2×1200B ReduceSum) | A | 10101 | 65 | 15.77 | — | baseline |
| 1 | ROW/COL-SUM-PROFILE-AS-ONE-CONV (kill perch_row/col 1200B planes) | A | 8991 | 656 | 15.83 | — | partial |
| 2 | uint8 (255-sentinel) for the 3 full planes | A | 6021 | 656 | 16.19 | — | partial |
| 3 | single fp16-Max combine plane (orientation folded into vectors) | A | 5841 | 655 | 16.22 | — | partial |
| 4 | fp16 the whole position/colour recovery vector chain | A | 5115 | 657 | 16.34 | — | partial |
| 5 | drop fp32-cast-for-reduce (ReduceSum/Max/Min accept fp16 now) | A | 4379 | 655 | 16.48 | 200/200, 300/300 | ADOPT |

## Best achieved
16.48 @ mem 4379 params 655 — beats prior 15.77 by **+0.70** (≥+0.3 ✓).
Stored 267/267, isolated fresh 500/500 (200+300 batches).

## Irreducible-floor analysis
Dominant intermediate: **L16, the single fp16 [1,1,30,30] combine plane (1800B)**.
The output is `Equal(L16, arange_ch)`; L16 must be a full-grid colour-index plane.
It cannot drop to uint8 (900B): the combine of "colour on the period axis" with
"in-grid gate on the cross axis" needs a uint8 `Max`/`Add`, both ORT-unsupported,
and the orientation (xpose) SWAPS the row/col roles so a single uint8 `Where`
(whose 3 args have fixed axis roles) cannot serve both orientations — a `Where`
per orientation + a select = THREE uint8 planes (2700B) > one fp16 Max (1800B).
Remaining 4×120B fp32 are the two profile Convs + two ReduceMax occupancy outputs
(born fp32, immediately Cast→fp16). ~28×60B fp16 [30] recovery vectors + 13×30B
bool make up the rest; each is already minimal-dtype. Params 655 = the two
[1,10,30,1]/[1,10,1,30] profile kernels (600 elems, must span 30 rows for arbitrary
grid heights; replacing with ReduceSum reintroduces 2×1200B planes = net worse).

## OPEN ANGLES (re-attack backlog)
- uint8 single combine plane (900B): would need a uint8 Max OR an orientation-free
  formulation. A Transpose-of-one-canonical-plane route still costs 3 planes. If a
  future ORT build adds uint8 Max, the combine drops 1800→900 (+~0.13).
- Smaller profile kernel: xpose=1 grids are up to 30 tall so the 30-row kernel is
  required; no saving available.

## INSIGHT (transferable)
⭐ When orientation (xpose) swaps the row/col roles of a separable row⊗col output,
the cross-axis gate + period-axis colour can be combined into ONE plane with a
broadcast `Max` of two perpendicular [30] vectors (off-grid→200 sentinel on EITHER
axis, in-grid→max(colour,0)=colour), selecting each vector's CONTENT (not axis) by
`Where(xpose,…)`. This collapses the usual 3-plane "build L0, build L1, select"
into a SINGLE fp16 combine plane. The combine is pinned at fp16 (1800B) because ORT
has no uint8 Max/Add and the role-swap blocks a single uint8 `Where`.
⭐ ReduceSum/ReduceMax/ReduceMin ACCEPT fp16 under ORT_DISABLE_ALL on the current
build (the "reduce ops reject fp16" gotcha is STALE) — so the whole integer
position/colour recovery chain runs in fp16 with NO fp32 bridge casts, halving every
[30] working vector (120→60B). Only Conv and the input-ReduceMax outputs are forced
fp32 (born from the fp32 input); Cast them to fp16 immediately.

## 2026-07-01 sequential deep pass

Current source has advanced beyond the older 4379B fp16 combine-plane solution:

- **memory 1869, params 61, points 17.43472471810107**
- fresh recheck: **1000/1000 pass**
- mechanism: recover horizontal/vertical seed profiles, build one 30-long
  alternating `line_pattern`, then emit the final one-hot output directly by
  comparing `row_side [1,10,30,1]` with `col_side [1,10,1,30]`.

Dominant costs:

- `row_side`: **300B**.
- `col_side`: **300B**.
- profile lines and base-valid slices: four 30-long fp32 vectors = **480B**.
- all remaining row/col pattern and validity vectors are 30B or scalar.

Rechecked direct-output alternatives:

- Building per-orientation full bool outputs and selecting between them would
  introduce full 10x30x30 planes, much worse.
- Replacing `row_side/col_side` with separate `Equal(channel_ids, row_code)` and
  `Equal(channel_ids, col_code)` still materializes the same 300B+300B side
  tensors, just as bool instead of uint8.
- The current representation is the compact broadcast form; it avoids every
  counted full 30x30 plane.

Conclusion: no adoptable improvement found.  This task is already using the
high-score-style direct output comparison pattern.

## 2026-07-01 parallel task013 deep dive

Scope: task013 only.  I did not edit `src/custom/task013.py`, `networks/`, the
manifest, or any global registry/script output.

Human-readable rule, verified:

- Stored examples have 4 train, 1 test, and 262 arc-gen cases.
- Input shape is either the generator's `height x width` with `height in 6..12`
  and `width in 20..30`, or its transpose.  Fresh generator sampling confirms
  reachable scored shapes span both axes: H 6..30, W 6..30.
- There are exactly two nonzero seed pixels with colors in 1..9.  In the
  untransposed form they are at columns `start` and `start + sep + 1`, each on
  either the top or bottom row.  The output paints full vertical stripes starting
  at `start`, stepping by `sep + 1`, alternating the two seed colors.
- If transposed, the same pattern becomes full horizontal stripes.  Edge cases
  include same-top/same-bottom seeds, same-left/same-right seeds after transpose,
  and max axis length 30; all are represented in fresh samples.

Current source/live state:

| item | result |
|---|---:|
| manifest / `measure_task.py 013` | 17.43472471810107 pts, mem 1869, params 61 |
| source stored eval | 267/267 pass |
| live `networks/task013.onnx` eval | 267/267 pass, mem 1869, params 61 |
| fresh verify | 1500/1500 incumbent pass |
| Python semantic oracle | stored 267/267, fresh 2000/2000 |

Cost anatomy from the live ONNX profile:

| component | bytes | semantic job |
|---|---:|---|
| `row_side [1,10,30,1]` | 300 | channel/row-side code for final broadcast equality |
| `col_side [1,10,1,30]` | 300 | channel/col-side code for final broadcast equality |
| `row_color_line`, `col_color_line` | 240 | color-weighted row/col seed profiles |
| `row_valid_base_f`, `col_valid_base_f` | 240 | recover in-grid row/col validity from padded input |
| 21 pattern/presence/gate vectors | 630 | 30-long bool/uint8 line, phase, validity, and code vectors |
| scalar/index recovery values | 159 | endpoints, colors, period, mode, and modulo scalars |
| output | 0 counted | final `[1,10,30,30]` bool tensor is scorer-exempt |

Dominant cost is the two 300B side tensors.  They exist because the final output
must be a 10-channel one-hot grid, but the scorer does not count the final
output tensor itself; the current graph pays for two skinny broadcasts and lets
`Equal(row_side, col_side)` materialize the free output directly.

Prior-note challenge:

- The older 4379B/fp16-combine-plane floor is contradicted by current source and
  live ONNX: the current graph has no full color-index combine plane and scores
  mem 1869.
- The 2026-07-01 sequential note claiming 1869B/61 params and direct output
  comparison is still valid.  I strengthened its fresh evidence from 1000/1000
  to 1500/1500 and added a separate Python oracle proof.
- The old "smaller profile kernel unavailable" claim is still valid: generator
  sampling found H=30 and W=30, so a fixed sub-30 axis would miss reachable
  cases.

Mechanism hypotheses tested:

1. Direct output threshold algebra / side-tensor deletion.
   - Expected payoff: remove one 300B side tensor, possibly +0.15 points if the
     output can be produced from one channel-axis comparison plus 30-long gates.
   - Proof test: inspect scorer accounting and broadcast shapes for alternative
     forms such as `Equal(channel_ids, row_code)`/`Equal(channel_ids, col_code)`
     plus orientation selection.
   - Kill condition met: every exact one-hot formulation still needs a
     channel-by-axis 10x30 broadcast for the stripe axis in each possible
     orientation, or else creates one or more full 10x30x30 bool intermediates.
     Current `row_side`/`col_side` is the compact two-orientation broadcast form.

2. Bounded crop before scan / shorter line vectors.
   - Expected payoff: if one axis were strictly below 30, the 30-long vectors
     and 300B side tensors could shrink proportionally.
   - Proof test: inspect `/tmp/arc-gen/tasks/task_0a938d79.py`, run a Python
     oracle, and sample 10000 fresh generator cases for reachable dimensions.
   - Kill condition met: fresh cases reached H=30 and W=30, with both row and
     column stripe orientations.  The 30-long line vectors and 10x30 side
     tensors are semantically required under the fixed scorer shape.

No source-owned candidate was created.  Recommended next experiment: only revisit
if a new primitive can produce orientation-swapped channel equality as a single
counted 10x30 side tensor, or if official/runtime accounting changes to exempt
one broadcast input to the final output op.  Otherwise this is a verified
no-adopt but useful failure.
