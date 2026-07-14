# task012 — 0962bcdd

**Rule:** size=12 grid, two distinct colours c0,c1 (1..9). Two centres placed; a
gravity reflect/transpose is applied EQUALLY to input and output, so the
input→output map is gravity-INVARIANT (gravity is just a symmetry of the whole
example, not a per-cell motion). INPUT: each centre is a 5-cell plus — centre=c0,
the 4 orthogonal neighbours=c1. OUTPUT: each centre grows a 5x5 stamp — c0 at the
centre and the 8 diagonal cells (dist 1 & 2: (0,0),(±1,±1),(±2,±2)); c1 at the 8
orthogonal cells (dist 1 & 2: (±1,0),(0,±1),(±2,0),(0,±2)). The two stamps never
overlap (centres 6 rows apart, stamps reach ±2). Output colour per cell is a
deterministic LOCAL function of input.

**Current:** 16.64 pts, label-map + structural centre Conv, mem 4170, params 103
**Target tier:** B (label map + final Equal). Tier S blocked: output colours
c0,c1 are random per instance, a fixed Conv cannot route them to the correct
output channel. Tier A blocked: the 5x5 X/plus stamp is not a row⊗col separable
rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colour-image Conv→A30, centre=Equal(A,c0), stamp Convs, label-map | B | 9066 | 102 | 15.88 | 200/200 | works, A30 [1,1,30,30] f32 = 3600 dominates |
| 2 | structural centre detector (slice ch0 12x12, count nonbg orth nbrs ≥4), drop A30 | B | 5610 | 102 | 16.35 | 200/200 | A30 eliminated |
| 3 | fp16 for all Conv-path 12x12 planes | B | 4170 | 103 | 16.64 | 200/200 | BEST |

## Best achieved
16.64 @ mem 4170 params 103 — adopted? N (build-only per task brief). Beats prior
15.56? Y (+1.08).

## Irreducible-floor analysis
Two intermediates dominate: **L [1,1,30,30] uint8 = 900** (the single-channel
output label map that drives the final Equal — irreducible for a label-map at the
30x30 output footprint; uint8 is already the smallest dtype) and **bg [1,1,12,12]
f32 = 576** (channel-0 slice; Slice preserves the f32 input dtype). Everything
else is fp16 (288) / bool (144) / uint8 (144) on the 12x12 active canvas. To go
lower you must leave the label-map family entirely, which the data-dependent
output colours forbid (S/A blocked). 4170 is at/near the label-map floor for this
rule.

## OPEN ANGLES (re-attack backlog)
- bg slice as fp16: Slice keeps f32; a Cast adds a node but bg stays 576 either
  way (the Slice output is the f32 plane). Could try Conv ch0-selector directly
  to fp16 — marginal (~288 saved → ~16.7).
- Fold L0/L12 into one Where via arithmetic label (c0st*c0 + c1st*c1) to drop one
  144B uint8 plane — sub-0.1pt.
- The 900B L is the hard floor; only escapable by a non-label representation,
  which the random per-instance colours block.

## INSIGHT (transferable)
⭐ **Gravity in ARC-GEN apply_gravity is a SYMMETRY of the whole example, not a
per-cell force.** It reflects/transposes input AND output identically, so any
input→output rule is gravity-invariant — build it in the canonical frame and it
holds in all 4 gravities for free, IF the detector is reflection/transpose
symmetric (a plus/X kernel is). **Structural, colour-independent center detection
(count non-background orthogonal neighbours via a 4-orth plus Conv, threshold ≥4)
beats colour-matching: it removes the [1,1,30,30] colour-value image (3600B) — the
single biggest win here (15.88→16.35), because the only 30x30 plane you truly need
is a single-channel SLICE of input ch0 (576B at 12x12), not a full colour Conv.**
fp16 on every small Conv-path 12x12 plane (values 0..4, exact) then halves them
(16.35→16.64).

## 2026-07-01 sequential deep pass

Current source has advanced beyond the older 4170B label-map solution:

- **memory 2076, params 188, points 17.275111560676926**
- fresh recheck: **1000/1000 pass**
- mechanism: identify the two colours by global counts, slice only the dynamic
  center-colour mask over the 8x8 possible-center region, use first/last ArgMax
  to locate the two centres, then scatter a 17-cell sparse stamp per centre into
  a 12x12 uint8 label plane before final 30x30 `Equal`.

Current dominant costs:

- `scalar_canvas [30,30]` uint8: **900B**.
- `center_mask [1,1,8,8]` fp32: **256B**.
- `color_flat`/`color_plane_u8`: **144B + 144B**.
- scatter indices: **136B + 136B**.
- largest param item is `zero_flat [144]`.

Rechecked easy substitutions:

- Replacing `zero_flat` with `ConstantOfShape` or `Expand` would save 144 params
  but introduce a counted 144B zero data tensor, net neutral/slightly worse.
- Sparse initializer for zero/scatter data is not a good route: prior probes show
  scorer shape inference rejects sparse tensors as ordinary op inputs.
- The final 30x30 scalar label carrier remains required because output colours
  are random per instance.

Conclusion: no adoptable improvement found in this pass.  The current graph is
already the sparse-stamp form that escaped the older full label-map floor.

## 2026-07-01 parallel deep-dive verification

Scope: task012 only.  Prior notes were treated as hypotheses and rechecked
against stored examples, the current source-owned graph, and fresh generator
instances.

### Human-readable rule

Confidence: **verified**.

- Stored/generator shape: input and output grids are 12x12; ONNX uses the
  repository-standard 1x10x30x30 one-hot carrier with only the 12x12 active
  region populated.
- Colours: two non-background colours per instance.  The colour appearing 8
  times in the input is the arm colour; the colour appearing 2 times is the
  center colour.  Background is 0.
- Input object: two 5-cell plus signs.  Each plus has center colour at the
  center and arm colour at its four orthogonal neighbours.
- Output object: each plus grows into a 5x5 sparse stamp.  Center colour appears
  at the center and both diagonal rings:
  `(0,0),(+/-1,+/-1),(+/-2,+/-2)`.  Arm colour appears at the orthogonal cells
  at distances 1 and 2: `(+/-1,0),(0,+/-1),(+/-2,0),(0,+/-2)`.
- Edge cases: generator applies `apply_gravity` equally to input and output, so
  gravity is a whole-example symmetry.  Fresh oracle check observed center rows
  and cols in `[2,9]`, validating the current `2:10,2:10` center crop.  The two
  5x5 output stamps can touch visually but do not overwrite different colours
  in generated ground truth.

Python oracle result:

```text
stored failures 0 [] of 265
fresh failures 0 of 1000
fresh center row range 2 9 col range 2 9
```

### Current source/live state

- Source: `src/custom/task012.py` is a live-exact, source-owned reconstruction.
- Live graph: `networks/task012.onnx`, opset 12, 29 nodes, 16 initializers.
- Manifest: `points=17.275111560676926`, `memory=2076`, `params=188`,
  `method=custom:task012`.
- Task-only measure:

```text
{'ok': True, 'pass': 265, 'fail': 0, 'memory': 2076, 'params': 188, 'points': 17.275111560676926, 'error': None}
```

Fresh gate:

```text
task12 arc=0962bcdd fresh_instances=1000/1000
  incumbent fail = 0
  candidate fail = 0
  candidate != incumbent = 0
```

### Cost anatomy

The scorer counts intermediate tensor bytes after shape inference and excludes
graph input/output.  Static inferred tensor bytes sum exactly to 2076.

| component | tensors | bytes | semantic job |
|---|---:|---:|---|
| Final scalar carrier | `scalar_canvas [30,30] uint8` | 900 | Holds colour ids plus outside sentinel 10 before broadcast `Equal` to one-hot output. |
| Center-colour crop | `center_mask [1,1,8,8] f32`, `center_mask_u8`, `center_flat` | 384 | Dynamic slice of the center-colour channel over the reachable center region; ArgMax finds first/last center. |
| Sparse stamp plane | `color_flat`, `color_plane_u8` | 288 | 12x12 scalar colour plane created by scatter before padding. |
| Scatter index path | `scatter_idx_2x17`, `scatter_idx`, `updates`, plus small row/base tensors | 342 | Converts two center indices into 34 stamp write locations and update colours. |
| Colour discovery | `counts10`, `top_counts`, `top_colors`, `arm_color`, `center_color`, casts | 98 | Finds background, arm colour, and center colour from global counts. |
| Dynamic slice bookkeeping | `center_starts`, `center_ends`, small constants/results | 64 | Builds the dynamic channel/range slice. |

Dominant cost remains the 900B scalar canvas.  It exists because output colours
are random per instance, so the graph first builds a single-channel colour-id
image and then uses `Equal(color_id, color_bank)` to produce the required
10-channel one-hot output without materializing a counted 10-channel final
carrier.

Largest param item is still `zero_flat [144] uint8`.  Total params are 188.

### Prior tasklog challenge

- Still valid: older 4170B label-map/fp16 notes describe a superseded mechanism,
  not the current graph.  They remain useful history but are not the current
  floor.
- Still valid: gravity is a symmetry of the whole example, not per-cell motion;
  confirmed from `/tmp/arc-gen/tasks/task_0962bcdd.py` and oracle/fresh checks.
- Still valid: the current graph escapes the older full-canvas structural Conv
  cost by using global colour counts plus an 8x8 center crop and sparse scatter.
- Rechecked: replacing `zero_flat` with `ConstantOfShape` is semantically valid
  but not a win.  Fixed valid ONNX form gives `memory=2220`, `params=45`,
  `points=17.274669962082864`: params drop by 143, memory rises by 144, net
  `memory+params` worsens by 1-2 scorer units.
- Refined: "final 30x30 scalar carrier remains required" should be read as
  required within the current efficient `Equal`-broadcast family.  A direct
  one-hot family is possible semantically but worse because any counted
  10-channel 12x12/30x30 intermediate dominates the 900B scalar carrier.

### Mechanism hypotheses tested

1. **Bounded crop before scan**
   - Expected payoff: avoid scanning 30x30 or 12x12 full planes for center
     detection; keep center detection to the 8x8 reachable region.
   - Proof test: Python oracle over stored examples plus 1000 generator examples;
     generator inspection shows canonical centers at rows `2,8`, columns
     `3..9`, with gravity mapping both axes into `2..9`.
   - Kill condition: any generated center outside `[2,9]` on either axis, or
     oracle mismatch.
   - Result: proof passed.  This is already exploited by the incumbent; no new
     adoption candidate.

2. **Final Equal / overlay-only reordering**
   - Expected payoff: remove the 900B `scalar_canvas` by doing `Equal` on the
     12x12 stamp plane, then padding to 30x30 as graph output.
   - Proof test: in-memory source-owned ONNX candidate, task012 only.
   - Kill condition: ONNX invalid, stored/fresh mismatch, or
     `memory+params >= 2264` incumbent total.
   - Result: raw bool `Pad` is invalid in this opset/runtime.  Valid uint8-cast
     version passes stored and 500 fresh examples but scores much worse:

```text
equal_before_pad_cast_u8 {'ok': True, 'pass': 265, 'fail': 0, 'memory': 4056, 'params': 192, 'points': 16.645796437078225, 'error': None}
equal_before_pad_cast_u8 fresh_instances=500 fail= 0
```

### Outcome and next experiment

No adoption recommended from this pass.  The incumbent is verified and remains
better than tested alternatives.

Next exact experiment, if revisiting task012: test whether `color_flat` and
`color_plane_u8` can be collapsed by padding a reshaped scatter output directly
or by using a 2D scatter target shape that feeds `Pad` without the extra reshape
materialization.  Expected maximum payoff is only 144B unless it also removes
one scatter-index materialization; kill if shape inference still counts both
scatter output and pad input, or if params rise by >= saved bytes.

Reusable insight: for random per-instance output colours, a 30x30 scalar
colour-id carrier plus broadcast `Equal` can beat any direct one-hot output
candidate, because the graph output is free but intermediate 10-channel carriers
are not.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.002)
**Mechanism (op-census diff):** Op census and initializer set are **identical** (no structural change); the −5B is a grader-profiler-side delta only.
**Old→new:** mem 2076→2071, params 188→188.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task012_pre_s10.onnx`; source `public_candidates/bobmyers7186/task012.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
