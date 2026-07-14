# task015 — 0ca9ddb6 (twinkling stars)

**Rule:** 9x9 grid placed top-left on the 30x30 canvas (cells outside the 9x9 grid
are all-zero in input AND output). Every coloured pixel copies input->output. A
blue(1) "rook" star stamps colour 7 at its 4 orthogonal neighbours; a red(2)
"bishop" star stamps colour 4 at its 4 diagonal neighbours; colours 6 and 8 just
copy. The generator scooches twinklers into [1, size-2] and keeps every twinkler
at Chebyshev distance >=1 from all other kept pixels, so the 7/4 halos always stay
inside the grid and land ONLY on background cells (never on each other or on a
pixel).
**Current:** 18.197605 pts, ext:kojimar6275, mem 0, params 900.
**Target tier:** S (pure-param single Conv, mem 0) — already there; the question is
purely whether 900 params can be shrunk.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 2x chained Where(halo->onehot, input) | B | 52200 | 59 | 14.14 | 265/265 | 10-ch stage1 intermediate (36000B) dominates |
| 2 | 9x9 block: slice 10 ch fp32 + dilation conv + bool overlay + Concat10 + Pad | B | 6318 | 91 | 16.23 | 265/265 | 10 fp32 9x9 slices (3240B) + cat10 uint8 (810B) floor it |
| 3 | runtime-assembled rank-structured conv weight (3 distinct 10x10 planes -> Concat -> Reshape [10,10,3,3]) | S* | 10800 | 408 | 15.68 | 265/265 | assembled fp32 weight (3600B) + Concat (3600B) intermediates > 900-param saving |
| 4 | single dense Conv [10,10,3,3] (= clean reimpl of public closed form) | S | 0 | 900 | 18.20 | 200/200 | ties prior; mem 0 |
| 5 | (2026-06-19 re-attack) PERMUTED group=2 Conv [10,5,3,3] via 2 axis-1 Gathers | S* | 72000 | 470 | 13.81 | 265/265 | params drop to 470 but the two channel-reorder Gathers trace at 9000B each (fp32) = 72000B; reorder is fatal |
| 6 | (2026-06-19) 9x9 active-region Slice + 2x Where(uint8,onehot) + Pad | B | 6480 | 221 | 16.19 | 265/265 | the copy "else" branch forces an fp32 [1,10,9,9] slice (>=3240B traced) > dense mem-0 floor |

## Best achieved
18.197605 @ mem 0 params 900 — adopted? matches prior (no regression). Beats prior
18.197605? **N (exact tie / at floor).**

## Irreducible-floor analysis
The closed-form rule IS a single linear conv of the one-hot: out_k(centre)=in_k for
all 10 channels; out_7 += in_1 at the 4 rook offsets; out_4 += in_2 at the 4 bishop
offsets; out_0 = in_0 - in_1(rook) - in_2(bishop) (the two -1 terms cancel the bg bit
exactly where a halo lands so the halo cell is a clean single-colour one-hot). A
single ungrouped Conv on a 10-channel one-hot I/O FORCES weight shape
[out=10, in=10, kH=3, kW=3] = 900 elements; params count ELEMENTS not nonzeros, and
the 3x3 kernel is forced by the +-1 halo offsets. So 900 params / mem 0 is the hard
floor for this architecture. Every escape pays a full-canvas fp32 intermediate that
exceeds the saving:
  - runtime-assembled weight: Conv forces fp32 weight dtype, so the [10,10,3,3]
    intermediate is 3600B (+ the Concat that builds it) -> 10800B mem (attempt 3).
  - per-channel bool-overlay block: the [1,10,9,9] Concat is 810B uint8 AND the
    passthrough needs ~3240B of fp32 9x9 slices -> 16.2 ceiling (attempt 2).
  - grouped Conv: the cross-channel edges (out7<-in1, out0<-in1/in2, out4<-in2) cross
    every contiguous group split, so no valid group < 10 exists.
Budget check: beating 18.20 by +0.3 needs mem+params <= exp(25-18.5) = 665; no route
gets the copy+halo logic under 665 without a >665B intermediate.

## OPEN ANGLES (re-attack backlog)
- None with payoff. The only sub-900 single-op would need a Conv weight whose first
  two dims are < 10, which is impossible without slicing input channels (an
  intermediate) since I/O are fixed 10-channel one-hots. Confirmed dead.

## 2026-06-19 RE-ATTACK with grouped-Conv sub-floor lever (task352 idiom) — CONFIRMED BAIL
The cross-channel coupling component is {0,1,2,4,7}: ch0<-in1(rook)/in2(bishop),
ch4<-in2(bishop), ch7<-in1(rook). For a group=2 Conv ([10,5,3,3]=450, the win) these
5 channels must live in ONE contiguous 5-block. In the NATURAL one-hot order the
forced split is 0-4 | 5-9, and out7<-in1 crosses it (ch7 in block2, ch1 in block1) —
no valid natural group<10 exists (matches the prior verdict). PERMUTING channels so
{0,1,2,4,7} are contiguous DOES make the grouped Conv exact (measured 265/265, 470
params) but the two required axis-1 Gathers each trace a full [1,10,30,30] fp32 plane
= 72000B -> 13.81 (attempt 5). There is no free 10-channel-reorder op. The decomposition
route (attempt 6) is dominated by the fp32 [1,10,9,9] copy slice (>=3240B). Three
distinct angles measured this session; all worse than the mem-0 dense floor 18.198.
Budget to reach 18.5 is mem+params<=665 and nothing keeps the 10-ch copy at mem 0
below 665 params. STAYS AT FLOOR 18.198.

## INSIGHT (transferable)
⭐ "Pure-param single-Conv" tasks (current net = one Conv, weight [10,10,3,3]=900,
mem 0) are AT FLOOR when the rule is an exact linear map of the one-hot (copy +
local-stamp halos). The dense weight's 900 element count is irreducible: the two
channel dims are pinned to 10 by the 10-channel I/O and the kernel by the stamp
radius. Runtime-assembling the weight from smaller inits LOSES because ORT forces
the assembled Conv weight to fp32 (>=3600B intermediate) — params drop but mem
balloons past the saving. Any block/Concat/Where reformulation introduces a
full-canvas intermediate (>=810B uint8 cat10 or >=3240B fp32 slices) that caps the
score ~16-18.3 < 18.5. So: a memory-0 single-Conv pure-param net with the channel
dims already at the one-hot count is a confirmed BAIL — do not chase rank-k weight
tricks (they help PARAMS for an INITIALIZER weight, never a runtime-built one).

## 2026-07-01 sequential deep pass

Fresh recheck: **1000/1000 pass**.

Current source remains the mem0 single dense `Conv(input, W[10,10,3,3])`:

- **memory 0, params 900, points 18.197605236675688**

No new candidate was tested because all plausible reductions are the same closed
families already measured: grouped Conv needs a counted channel reorder,
runtime-built sparse/rank weights create fp32 weight intermediates, and
Where/Concat decompositions introduce full-canvas or 9x9 10-channel state.  The
task remains at the pure-param Conv floor.

## 2026-07-01 parallel deep dive (task015 only)

Scope guard: this pass measured and probed only task015; no whole-project scripts
or rebuild/adopt commands were run.

### Human rule and confidence

Stored examples are 9x9 grids embedded by the harness into the top-left of the
30x30 one-hot canvas; all cells outside the 9x9 example are zero/background in
both input and output.  Colours observed in stored train/test/arc-gen examples
are 0, 1, 2, 4, 6, 7, and 8.

Verified rule: every source pixel copies through.  A colour-1 pixel stamps colour
7 into its four orthogonal neighbours.  A colour-2 pixel stamps colour 4 into its
four diagonal neighbours.  Colours 6 and 8 are inert copy-through pixels.  The
generator moves twinklers off the border and rejects overlapping radius-1
twinkler neighbourhoods, so fresh generated halos stay in-bounds and land on
background cells.  Confidence: **verified** by Python oracle over all stored
examples (`train 3/3`, `test 1/1`, `arc-gen 261/261`) and fresh incumbent
verification (`1500/1500`, incumbent fail 0).

### Current source/live state

`src/custom/task015.py` and `networks/task015.onnx` both contain one op:
`Conv(input, W) -> output`, `kernel_shape=[3,3]`, `pads=[1,1,1,1]`, opset 10.
The live initializer is `W` with shape `[10,10,3,3]`, dtype float32, 900 counted
elements, 3600 raw bytes, and 26 nonzero weights.  Stored eval:

| source | pass/fail | mem | params | points |
|---|---:|---:|---:|---:|
| `reports/scripts/measure_task.py 015` | 265/0 | 0 | 900 | 18.197605236675688 |
| `reports/manifest.json` | n/a | 0 | 900 | 18.197605236675688 |

### Cost anatomy

| component | cost counted | semantic job | note |
|---|---:|---|---|
| Conv output carrier | 0 mem | writes final 10-channel one-hot score tensor directly to `output` | scorer excludes graph output memory |
| Copy identity weights | 10 params inside dense W | copy all input colours through at the center tap | identity is sparse, but dense Conv counts the whole initializer shape |
| Colour-1 rook halo | 4 positive params, plus 4 background-cancel params | set output ch7 at N/S/E/W and suppress output ch0 at those cells | needed because scorer thresholds every channel; background cannot stay positive |
| Colour-2 bishop halo | 4 positive params, plus 4 background-cancel params | set output ch4 at diagonals and suppress output ch0 at those cells | same thresholded one-hot requirement |
| Dense Conv shape slack | 874 counted zero params | unavoidable under a normal ungrouped Conv initializer with fixed 10 input channels, 10 output channels, 3x3 footprint | dominant cost |

Dominant cost is not arithmetic or memory; it is the counted dense initializer
footprint `10*10*3*3 = 900`.  Points are exactly the mem0/params900 score
`25 - ln(900)`.

### Prior tasklog challenge

Still valid:

- The visible rule matches the stored examples and generator.
- The current graph is an S-tier mem0 direct-output Conv.
- Removing background cancellation is invalid under threshold scoring because
  halo cells would retain positive output ch0 while also activating ch4/ch7.
- Natural grouped Conv cannot represent the rule: required cross-channel edges
  cross every legal contiguous group split smaller than 1 group.
- Runtime assembly / Where / Concat families are still dominated by counted
  intermediates and cannot beat the mem0/900 baseline without a new scorer-safe
  primitive.

Newly sharpened:

- Sparse Conv initializer looks superficially like a 26-param win, but is killed
  by the official harness path.  Direct `onnx.checker.check_model(...,
  full_check=True)` fails with `ShapeInferenceError (op_type:Conv): W typestr: T,
  has unsupported type: sparse_tensor(float)`.  The repo sanitizer also does not
  rename sparse initializers with node inputs, producing a load error in
  `evaluate`.  Raw ORT can run a small sample, but the official scorer route
  cannot measure it, so this is not adoptable.

### Mechanism probes

1. **Direct output threshold algebra: grouped Conv sub-floor.**
   Expected payoff: params 900 -> 450 for `groups=2`, 180 for `groups=5`, or 90
   for depthwise `groups=10`, all mem0 if feasible.  Proof test: enumerate live
   nonzero `W[o,i,kh,kw]` and check whether every required edge stays inside the
   legal contiguous input/output block for each group count.  Kill condition:
   any required nonzero crosses a block boundary.  Result: killed.  Violations:
   `groups=2` has 4 crossings (`out7 <- in1` rook halo); `groups=5` has 12
   crossings; `groups=10` has 16 crossings.  A channel permutation would fix
   grouping algebraically but requires a counted full 10-channel reorder.

2. **Sparse initializer shortcut for the same direct Conv.**
   Expected payoff: params 900 -> 26, mem0, if official scoring counted only
   sparse values and accepted sparse Conv weights.  Proof test: build an
   in-memory task015 Conv with `graph.sparse_initializer` for `W` using the live
   26 nonzeros.  Kill condition: official checker/scorer cannot load, infer, or
   measure it.  Result: killed.  `calculate_params` would count 26 values, and
   raw ORT loaded and verified a small sample, but official full shape inference
   rejects sparse Conv weights and repo `evaluate` reports load failure after
   sanitization.  This confirms the prior "sparse initializer shortcuts are not
   valid unless official shape inference accepts them" warning for task015.

### Verification

- Stored incumbent eval: `{'ok': True, 'pass': 265, 'fail': 0, 'memory': 0,
  'params': 900, 'points': 18.197605236675688, 'error': None}`.
- Python oracle on stored examples: `train pass 3 fail 0`, `test pass 1 fail 0`,
  `arc-gen pass 261 fail 0`.
- Fresh incumbent eval: `task15 arc=0ca9ddb6 fresh_instances=1500/1500`,
  `incumbent fail = 0`.

### Recommendation and next experiment

Recommendation: **no main-session adoption** from this pass.  The incumbent is
already verified and unchanged.

Next exact experiment only if a new primitive is allowed: test whether an
official-scorer-valid quantized or compressed Conv form can provide mem0 with a
counted weight footprint below 900.  Kill it immediately unless
`onnx.checker.check_model(..., full_check=True)`, repo `evaluate`, and fresh
task015 verification all pass; raw ORT success alone is insufficient.
