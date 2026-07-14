# task368 - e76a88a6

## 2026-07-01 deep dive - current live-exact anchor scatter verified

**Rule confidence: verified.** A 10x10 input contains 3 or 4 non-overlapping copies
of the same solid rectangle footprint. The rectangle size is one of 3x3, 3x4, or
4x3; 4x4 did not occur in 1000 fresh samples and is not supported by the current
12-offset graph. One rectangle is shown in its true two-colour pattern, using two
non-gray colours from channels 1,2,3,4,6,7,8,9. Every other rectangle is all gray
5. The output keeps the same rectangle placements, but recolours every gray
rectangle with the pattern from the coloured rectangle.

Visible examples:

- train 0: a 3x3 coloured `2/4` block at rows 1..3, cols 1..3; two gray 3x3
  blocks at rows 4..6, cols 6..8 and rows 7..9, cols 2..4. Output copies the
  exact 3x3 `222/244/444` pattern to all three placements.
- train 1: a 3x4 coloured `6/8` block and two gray 3x4 blocks. Output copies the
  `6666/8868/6888` pattern to each placement.
- test 0: a 4x3 coloured `4/1` block and three gray 4x3 blocks. Output copies
  the four-row pattern to all four placements.

Python oracle:

- Find connected nonzero components.
- The unique component containing a nonzero non-gray cell is the source pattern.
- Copy its bounding-box pattern to every component's top-left corner.
- Verification: stored oracle `0/265` failures; fresh oracle `0/1000` failures.
- Fresh rectangle/component counts observed: `(3,3,3)`, `(3,3,4)`, `(3,4,3)`,
  `(3,4,4)`, `(4,3,3)`, `(4,3,4)`.

## Current source/live state

The previous notes below were stale. Current `src/custom/task368.py` is an exact
source reconstruction of `networks/task368.onnx`, and manifest has:

| item | points | memory | params | stored | fresh/adopt |
|---|---:|---:|---:|---|---|
| current source | 16.26367127866709 | 6022 | 203 | 265/265 | fresh 500/500 |
| live ONNX | 16.26367127866709 | 6022 | 203 | 265/265 | fresh_pass 500/500 |
| adopt gate | 16.26 | 6022 | 203 | pass | rejected: candidate <= current real |

Current mechanism:

1. Slice color-5 and color-0 planes from the 10x10 input.
2. Locate the coloured source rectangle as the first cell where input is neither
   zero nor gray.
3. Locate gray rectangle anchors by taking gray cells with no gray above or left,
   then `TopK(k=3)`. If there are only two gray rectangles, duplicate the first
   gray anchor for the unused third placement.
4. Infer rectangle height/width from the first gray anchor by probing `+30` and
   `+3`, so each dimension is 3 or 4.
5. Generate the 12 possible offsets for the source and all placements. Invalid
   offsets are masked when area is 9.
6. Gather the source pattern's first colour at the source anchor, probe all
   source offsets to build a 12-cell colour mask, discover the second colour from
   the first non-first-colour offset, then form the 12 colour updates.
7. Scatter 4 copies of the 12 updates into a 10x10 uint8 label grid, pad to
   30x30 with sentinel 10, and finish with `Equal(label, channels_u8)`.

This is strictly better than the old run-length/histogram route because it uses
the task-specific fact that every rectangle is a solid 3x3/3x4/4x3 footprint and
only the anchor plus the source 12-cell pattern is needed. It avoids full-canvas
run-length planes, a 4x4 histogram, and a 30x30 colour-index plane.

## Cost anatomy

Actual scorer memory from sanitized runtime trace:

| component | dominant tensors | bytes | semantic job | keep/attack |
|---|---|---:|---|---|
| final 30x30 label | `placed_color_30_u8 [1,1,30,30]` | 900 | feed free final `Equal` and suppress off-grid with sentinel 10 | required unless output can be written directly |
| entry planes | `five_f`, `zero_f` | 800 | read gray anchors and distinguish coloured source from background/gray | likely floor for this sparse-anchor graph |
| source-offset gather indices | `a_indices_i64 [1,12,3]`, row/col/color expands | about 576 | gather the 12 source cells of the source colour | possible small int64-index target |
| anchor/source masks | gray/source 10x10 bool/uint8 planes, padded gray, `ph_flat_f16` | about 1621 | find source cell and gray top-left anchors | dominant non-final working set |
| placement/update indices | `all_pos_i32 [48]`, per-placement 12-cell position vectors | about 576 | scatter source pattern to up to four rectangles | small, structurally useful |
| colour discovery | `a0_indices_i64`, `b_indices_i64`, one-hot gathers | about 448 | identify the two non-gray palette colours | possible small target, not a big win |
| initializers | 34 tensors | 203 params | slice constants, offsets, channel vectors, blank grid | already small |

Largest individual non-output tensors:

| tensor | bytes | dtype/shape | why it exists |
|---|---:|---|---|
| `placed_color_30_u8` | 900 | uint8 `[1,1,30,30]` | final label plane for `Equal` |
| `five_f` | 400 | fp32 `[1,1,10,10]` | gray rectangle occupancy |
| `zero_f` | 400 | fp32 `[1,1,10,10]` | background exclusion for coloured-source mask |
| `a_indices_i64` | 288 | int64 `[1,12,3]` | source-pattern `GatherND` indices |
| `ph_flat_f16` | 200 | fp16 `[1,100]` | `TopK` anchor selector input |

Tier assessment: high-B / anchor-scatter. It has one final full label plane and
two 10x10 entry planes, but no full 30x30 colour-index plane and no full 10x30x30
input carrier. S/A remain blocked by arbitrary 2-colour source pattern plus
multiple placements, but the current graph is much closer to the practical floor
than the old tasklog claimed.

## Challenged prior claims

| prior claim | status | evidence |
|---|---|---|
| "Current stored 14.69, mem 29948, params 84" | contradicted | manifest and `measure_task.py 368` report 16.26367127866709, mem 6022, params 203 |
| "Best achieved 15.37, mem 14936, params 314, not adopted" | superseded | current source/live ONNX is already better and task-scoped adopt rejects only because candidate equals current |
| "Dominant floor is 3600B fp32 colour-index plane" | contradicted for current graph | no `colf30`; largest current tensor is 900B `placed_color_30_u8`; input reads are two 400B slices |
| "Run-length product-chain is the key structure" | semantically valid but no longer cost-competitive | oracle confirms solid-rectangle offset idea, but current anchor-scatter avoids computing offsets for every occupied cell |
| "Tier A blocked because pattern is arbitrary 2-D" | still valid | source pattern is arbitrary over the 3x3/3x4/4x3 footprint; row/col separability is not guaranteed |
| "4x4 table needed" | overgeneralized | reachable fresh sizes are area <= 12; current `idx12_i32` and height/width probes are enough for stored + 1000 fresh oracle + 500 fresh ONNX |

## Mechanism hypotheses tested

### 1. Solid-rectangle/run-length product-chain

- Expected payoff at time of old note: +0.68 over a 14.69/29948 baseline.
- Proof test: stored eval and fresh verify; old note reports 265/265 and 500/500.
- Current kill condition: if measured current source/live is already better in
  points and memory, do not resurrect this mechanism.
- Result: killed for adoption. The mechanism was real against the older baseline
  but loses badly to current anchor-scatter: 15.37/14936/314 versus
  16.263671/6022/203.

### 2. Current sparse anchor-scatter instead of per-cell run-length

- Expected payoff: delete the old full-canvas run-length/histogram stack and
  reduce work to anchors plus 12 source-pattern cells.
- Proof test: Python oracle on stored and fresh; source/live stored eval; fresh
  verifier; task-scoped adopt gate.
- Kill condition: any stored/fresh failure, or measured cost not beating the old
  15.37 candidate.
- Result: verified current incumbent. Stored source and live both pass 265/265 at
  mem 6022/params 203; fresh verifier and `fresh_pass` both pass 500/500; adopt
  rejects only because source is equal to current real score.

## Next experiment

No main-session adoption is recommended from this pass; task368 is already on the
better current graph. The next exact experiment, if this task is revisited, is a
small-index golf of the current graph:

- Try replacing some int64 `GatherND` index construction (`a_indices_i64`,
  `a0_indices_i64`, `b_indices_i64` and row/col expands) with cheaper int32 paths
  only if ORT accepts the relevant op signatures.
- Expected payoff: at most a few hundred bytes, roughly +0.03 to +0.08 points.
- Kill condition: ONNX checker/ORT rejects int32 indices, or the replacement
  materializes casts whose memory exceeds the saved int64 tensors.

Reusable insight: for "copy one coloured solid rectangle pattern to gray copies,"
first check whether generator bounds reduce the problem to a small fixed list of
source offsets and rectangle anchors. If yes, sparse anchor-scatter can dominate
the more general run-length/histogram formulation.

## S11 (2026-07-03) — mech-15/pointer scout: KILL — copied source is an arbitrary 2-colour 2-D footprint stamped at N gray anchors (sprite-stamp class); row/col separability not guaranteed (per own tasklog). No monochrome separable fills.

## 2026-07-05 sparse final-output scatter probe — REJECTED

Tested the proposed outer-loop/final-output-only direction on this task with
`reports/candidates/task368/sparse_output_scatter.py`.  The candidate removes
the incumbent `placed_color_30_u8` label plane and final `Equal`, then writes
directly into the free input tensor with `ScatterND`: for each gray target cell,
set channel 5 to 0 and the dynamic target colour channel to 1.

Stored result: 265/265 correct, but cost regressed from `memory=5990,
params=203, points=16.268825` to `memory=12894, params=281,
points=15.513924`.

Kill reason: deleting one 900B final label plane is not enough when the direct
output form needs dynamic int64 `[72,4]` ScatterND indices plus row/column/channel
preparation tensors.  Sparse direct-output scatter is only promising when the
dynamic index count is much smaller, the index rank is lower, or an existing
index tensor can be reused.  For this task the current 10x10 label carrier is
cheaper than direct one-hot sparse updates.


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 6193 -> 5147 (+0.185)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].