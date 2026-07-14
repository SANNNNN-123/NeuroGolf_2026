# task086 - 3befdf3e

Deep dive date: 2026-07-01

## Human-readable rule

Confidence: verified.

The input grid is size 10, 11, or 12, with one or two non-overlapping square
"flowers". Each flower has length `L in {1,2}`:

- input block is `(L+2) x (L+2)`;
- the 1-pixel border is color `c1`;
- the `L x L` center is color `c0`;
- `c1` is the more frequent non-background input color.

The output keeps the same two colors but redraws each flower as a concentric
stamp at the same location:

- the original `(L+2) x (L+2)` block border becomes `c0`;
- the original `L x L` center becomes `c1`;
- orthogonal arms of length `L` extend outward above, below, left, and right
  along the original block span, also in `c1`;
- corners of the larger bounding square remain background.

Stored examples:

- train0: one `3x3` flower, colors `6/4`, output is a `5x5` plus-like stamp.
- train1/train2: one `4x4` flower, output is an `8x8` cross-with-ring stamp.
- test0: two flowers in a `12x12` grid, one `3x3` and one `4x4`; stamps do not
  overlap.

The generator confirms `size in {10,11,12}`, attempts two placements, and drops
the second if it overlaps the first. Placement bounds keep the output stamp
inside the grid.

Python oracle status:

- bundled `train + test + arc-gen`: pass;
- fresh generator: 2000/2000 pass.

## Current source/live state

Current `src/custom/task086.py` is an exact live-source reconstruction, not the
older source-owned L-parametric morphology candidate described below.

Measured current source:

```text
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py 086
{'ok': True, 'pass': 266, 'fail': 0, 'memory': 2946, 'params': 190,
 'points': 16.9492966185297, 'error': None}
```

Measured live ONNX:

```text
PYTHONPATH=. .venv/bin/python src/harness.py networks/task086.onnx 86
ok=true, pass=266, fail=0, memory=2946, params=190, points=16.9492966185297
```

Fresh incumbent verification:

```text
PYTHONPATH=. .venv/bin/python reports/scripts/fresh_verify.py 086
task86 arc=3befdf3e fresh_instances=1500/1500
  incumbent fail = 0
```

Manifest entry: `custom:task086`, `memory=2946`, `params=190`,
`points=16.9492966185297`.

## Current ONNX mechanism

The live graph:

1. Slices the top-left `12x12` area.
2. Infers actual grid size as `10 + input[10,0] + input[11,0]` using the padded
   one-hot background channel.
3. Builds a valid-size mask and occupancy mask.
4. Uses one `QLinearConv` with an `11x11` int8 kernel and bias `-28` to classify
   the output `c1` spatial mask over the reachable generator state space.
5. Uses channel counts and `TopK` to recover the two non-background colors:
   most frequent input color is `c1`, second is `c0`.
6. Builds a uint8 label plane at `12x12`: background sentinel, `c0` on occupied
   input block, `c1` wherever the learned outer mask is true.
7. Pads that label plane to `30x30` and emits one-hot output via final `Equal`.

## Cost anatomy

| component | tensor(s) | bytes | params | semantic job |
|---|---:|---:|---:|---|
| final label carrier | `labels30` uint8 `[1,1,30,30]` | 900 | 0 | per-cell color index feeding final free `Equal` |
| input crop | `background12` fp32 `[1,1,12,12]` | 576 | 0 | top-left background channel used for size and occupancy |
| 12x12 boolean masks | `background_bool`, `valid12`, `occ_bool`, `outer_mask` | 576 | 0 | validity, occupancy, and c1-spatial mask |
| 12x12 int8 conv path | `occ`, `outer_score` | 288 | 0 | compact local classifier input/output |
| 12x12 uint8 label path | `base_labels`, `occ_labels`, `labels12` | 432 | 0 | staged label construction before 30x30 pad |
| color recovery | `all_color_counts`, `color_counts`, `TopK` outputs, scalar labels | 174 | 0 | global c0/c1 recovery |
| dense local classifier | `outer_weight` int8 `[1,1,11,11]`, bias, constants | 0 | 190 total | one-shot reachable-state morphology classifier |

Dominant memory is the output-side label carrier plus the fp32 crop and several
12x12 masks. The current graph already uses the key scoring trick: final
`Equal` makes the large `[1,10,30,30]` one-hot output free, while only a
single-channel uint8 label plane is counted.

The 18-point threshold requires total `memory + params <= ~1096`. Current total
is `3136`. If all 12x12 intermediates vanished and only the uint8 30x30 label
plane plus current params remained, the score would be about `18.006`; therefore
the remaining path to 18 is not parameter shaving alone, but removing or making
free most crop/mask intermediates while keeping the final label plane.

## Challenge to prior notes

Old note: "15.67 @ mem 11168 params 98 ... Beats prior 14.89? Y (+0.78)."

Status: stale and not adoptable now. It may have beaten an older baseline, but
current source/live is `16.9493 @ mem 2946 params 190`. The claimed
L-parametric color-index/free-Equal candidate would score only
`25 - log(11168 + 98) = 15.6705`, below current by about `1.28` points.

Old note: "The L/fp16 color-index plane is the floor."

Status: contradicted by current live graph. Current already uses a uint8
`30x30` label plane (`900B`), not an fp16/fp32 `1800B/3600B` plane. The current
floor candidate is the uint8 label carrier plus whatever spatial masks are
needed to form it.

Old note: "Semantic rule: L-parametric concentric stamp."

Status: still valid. The rule is verified by generator inspection, bundled
oracle pass, and 2000 fresh oracle pass.

## Mechanism hypotheses tested

### 1. L-parametric morphology + color-index -> final Equal

Expected payoff:

- Old claimed route: `mem=11168`, `params=98`, `points=15.6705`; no adoption
  payoff versus current `16.9493`.
- A hypothetical perfect label-only route with `900B + 98 params` would score
  `18.094`, but the old L-parametric graph did not get near that because its
  intermediate morphology planes dominated memory.

Proof test:

- verify the semantic oracle on bundled and fresh data;
- compare claimed cost against current source/live.

Result:

- Semantic rule validated.
- Specific old adoption claim killed for current mainline: it is stale and
  lower-scoring than live.

Kill condition:

- Any morphology implementation that materializes multiple full 30x30 or
  fp16/fp32 planes cannot beat the current uint8-label/QLinear route.

### 2. Shrink the learned local QLinear classifier kernel

Expected payoff:

- Replacing the `11x11` int8 kernel with `7x7` saves 72 params, worth only about
  `+0.023` points if memory is unchanged.
- Replacing it with `5x5` saves 96 params, worth about `+0.031` points.

Proof test:

- Enumerated all reachable generator states for sizes 10-12, one or two
  non-overlapping flowers.
- For every 12x12 cell, classified whether it should be in the output `c1` mask.
- Tested local patch sufficiency and linear separability for `k x k` binary
  patches using `scipy.optimize.linprog`.

Result:

```text
states 4099
k 3 KILL same local patch opposite labels
k 5 KILL same local patch opposite labels
k 7 unique 2682 pos 955 neg 1727
   linprog infeasible
k 9 unique 5894 pos 1771 neg 4123
   linprog infeasible
k 11 unique 10178 pos 2683 neg 7495
   linprog feasible
```

Kill condition met for single-conv shrink: below `11x11`, either local evidence
is insufficient (`3x3`, `5x5`) or a single linear threshold cannot separate the
reachable labels (`7x7`, `9x9`). The current `11x11` `QLinearConv` is justified
for the one-shot local classifier family.

## Recommendation / next experiment

No adoption candidate from this session. Keep current `src/custom/task086.py`
and `networks/task086.onnx` as-is.

Next exact experiment, if task086 is revisited: try to make the final uint8
`labels30` become the direct final output of a graph whose last op is a
channel-wise comparison/combination, while avoiding materializing either
`labels12` or multiple `12x12` mask copies. The target is to reduce counted
memory by about 2000B; parameter-only changes are too small to matter.

Reusable insight:

For small ARC-GEN morphology tasks, a dense local classifier can beat explicit
semantic morphology when the semantic rewrite materializes many planes. Before
rewriting a compact learned `QLinearConv`, prove whether smaller local windows
are sufficient and linearly separable over the full reachable generator state
space. For task086, `11x11` is the minimum tested single-threshold local window.
