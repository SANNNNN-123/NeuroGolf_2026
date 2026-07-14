# task101 — 447fd412

## 2026-06-29 semantic rewrite screen

Rule: the generator draws 2..4 copies of one small connected sprite.  The first
copy has `bmag=1` and shows both blue(1) and red(2) pixels.  Later copies show
red pixels in the input but hide blue pixels; the output restores the hidden
blue pixels at the copy's magnification (`bmag` in 1..3).  Dimensions are
bounded: ordinary fresh grids are 12x14 or 14x17, while stored/test stress can
reach 21 columns.

Current source/live graph is a source-owned exact replay with:

- score `15.210521364075738`
- memory `16940`
- params `905`
- fresh check `1000/1000`

Dominant tensors from a traced mem profile:

| tensor | bytes | note |
|---|---:|---|
| `c1cf`, `c2cf` | 1428 each | fp32 channel slices over the 17x21 work grid |
| `crop3` | 1071 | final 3-channel bool crop before Pad to output |
| `rawpos_score` | 714 | fp16 score vector for TopK over red candidate pixels |
| 17x21 bool masks | 357 each | copy-slot coverage, blue mask, output channel masks |

Mechanism tested: transfer the task351/task201 copy compiler pattern
(`1x1 colour-index plane -> small gathers -> final one-hot`).  Reject for this
task: input colours are only 1/2, so the current direct ch1/ch2 `Slice` pair
costs `2 * 1428 + bool casts`, while a full colour-index entry plane would cost
`3600` before the same work-grid logic.  The usual colour-index copy lever is
not cheaper when the useful palette is already two fixed channels.

TopK-width probe: current graph keeps `k_raw=29` red candidate positions.  Stored
max red count is 22, but 20k fresh samples reached 30 red cells:

| source | max red count |
|---|---:|
| stored train/test/arc-gen | 22 |
| fresh 20k | 30 |

Adopt decision: **no rewrite**.  Reducing `k_raw` is unsafe under the generator,
and replacing channel slices with a colour-index plane is a net loss.  Current
fresh generalization is good (`1000/1000`), so leave the source graph unchanged.

Transferable negative insight: bounded multi-copy reconstruction tasks can look
like pure spatial-copy wins, but if the palette is fixed to a few channels,
direct channel slices can beat the standard colour-index entry plane.  Also,
candidate-list widths must be bounded from generator extremes, not only stored
examples.

## 2026-06-30 component/scale anchor research

User hypothesis: the hard part across copy tasks is not the visual rule, but
cheaply recovering component count, scale, and anchor.  Task101 is a good probe
because the generator has a small connected reference object, then places
upscaled copies where only the indicator/red cells are visible.

Generator source check (`generate_447fd412` from RE-ARC): reference object is a
connected object with bbox at most 4x4.  Later occurrences use
`outobj = upscale(obj, fac)` and input only `indic` cells; output restores the
hidden `main` cells.  This confirms that a component-level mechanism should be
possible in principle, but scale and anchor must be exact.

Stored/bundled analysis:

| probe | result |
|---|---:|
| bundled examples inspected | 266 |
| 4-connected red component count | max 8, unstable |
| 8-connected red component count | max 8, unstable |
| red cells | max 22 in bundled examples |
| naive red-to-blue offset spray | rejects: massive false positives |
| top-left template scan over reference red cells | 225/266 |
| red-anchored template scan, all red template cells visible | 212/266, miss 0 but 141 extra blue cells |
| + 4-neighbor separation filter | 262/266, miss 0 but 5 extra blue cells |
| + maximal-scale/coverage suppression | **266/266**, extra 0, miss 0 |

Mechanism candidate: replace raw red-cell enumeration with a **maximal
reference-template anchor map**:

1. Extract the reference component containing both colours.
2. Build dynamic red-template cells and blue-template cells for scale 1/2/3.
3. Candidate anchor is any placement where all in-template red cells are present.
4. Reject anchors whose 4-neighborhood touches unrelated nonzero cells, matching
   the generator's spacing invariant.
5. Process larger-scale matches first; suppress smaller matches whose red
   coverage is already covered by a larger accepted match.  This removes false
   positives caused by scale-1 submatches inside scale-2/3 red blocks.
6. Scatter only the blue-template cells for accepted anchors.

Why this matters: the current graph carries `TopK(k_raw=29)` over all red cells
and then walks copy slots with many `Gather`/coverage ops.  The Python oracle's
valid template candidates averaged 3.29 per example, max 6 on bundled data.  If
the dynamic template anchor map can be compiled cheaply, it may remove much of
the 29-wide raw candidate path.

Risk: this is not yet an ONNX improvement.  Dynamic template matching may still
be expensive because the reference red/blue masks are input-dependent, and the
maximal-coverage suppression may recreate a smaller version of the current
coverage chain.  Fresh generator is not available locally in this workspace, so
the 266/266 result is stored/bundled only; earlier tasklog fresh evidence still
applies only to the incumbent graph.

Next attack: prototype a source-owned candidate that keeps the existing 4x4
reference extraction but replaces `rawpos_score`/`TopK(k_raw=29)` with scale
template anchor candidates.  Reject unless stored passes and memory meaningfully
drops below the current `16940`.

## 2026-06-30 adoption attempts after anchor research

Goal: turn the anchor research into an actual score improvement.  No candidate
was adopted.

| candidate | stored | memory | params | points | outcome |
|---|---:|---:|---:|---:|---|
| incumbent | 266/266 | 16940 | 905 | 15.210521 | keep |
| output tail `uint8 class_idx -> Equal(output)` | 0/266 | 16769 | 917 | 0 | failed: padded/outside-grid cells became colour 0 instead of all-zero |
| output tail with invalid sentinel outside grid | 266/266 | 17126 | 918 | 15.199432 | correct but worse |
| `p1/p2` TopK score vectors as uint8 | 23/266 | 16898 | 906 | 0 | failed: ordering changed; original uses `0,-1,-2...` fp16 scores |
| `p1/p2` TopK score vectors as int8 | load fail | - | - | 0 | ORT has no int8 `Where` kernel here |
| omit TopK value output | load fail | - | - | 0 | ONNX TopK value output cannot be empty |
| `onnxsim` compression sweep | 266/266 | 16940 | 905 | 15.210521 | no gain |
| single Conv fit (`k=1,3,5`) | fail | - | - | 0 | channel 0 not separable |
| local public candidates | 266/266 | 16940 | 905 | 15.210521 | same as incumbent; URAD was slightly worse |

Conclusion: the cheap local edits around the existing graph do not improve the
score.  A real improvement still requires replacing the `rawpos_score` /
`TopK(k_raw=29)` copy-finding path with the maximal reference-template anchor
map, not just dtype or output-tail surgery.  That rewrite is nontrivial because
the ONNX graph must compile dynamic red/blue reference offsets and scale
suppression without recreating an equally large candidate/coverage chain.

## S3 re-fit pass (2026-06-30) — drop 10 no-op nodes, bit-identical
Removed 10 provably-true/no-op nodes from the copy1/copy2/copy3 scale-detection bounds-checks:
5× `Equal(x, x)` (tautological True on int tensors: copy{1,2,3}_scale_idx_rok/cok) and the
dependent `And(seen, ib)` chains where `ib`≡True, so `seenib≡seen` (copy1/2) and copy3's chain
collapses to its real row-clamp `rok`. Codegen placeholder bounds-checks that never clamp here.
- Before: mem 16940 / params 905 / 15.2105 pts.  After: mem **16850** / params 905 / **15.2156 pts (+0.0051)**.
- Gates: evaluate fail=0 (266/266); ORT_ENABLE_ALL fail=0; fresh_verify 2500 instances = candidate
  bit-identical to incumbent (0 divergence, including identical pre-existing OOB behavior), fail-vs-GT=0.
- Rest of graph is FLOOR: no other dead/dup nodes; the two 1428B fp32 channel slices are the proven
  per-cell detection floor; the two 357-elem arrays are mem↔params fungible (zero net change). LANDED.

## S8 (2026-07-02) — counting-model rebuild + CRASH FIX (+0.206) ADOPTED
Free-input einsum profiles for blue bbox; 4×4 reference patch via 'bchw,c,uh,vw->uv' with
OneHot selectors (OOB rows read zeros); nzc plane dropped; epilogue = single
Where(blue_mask30, e1_vals, input). 13573+874 vs 16880+875 → 15.216→15.422.
REAL BUG FIXED: incumbent HARD-ERRORS ORT on ~0.1% fresh (red target at rows 15-16 →
scale-probe Gather idx 357+ OOB; S3 refit removed non-tautological bounds checks) — silent
private-LB risk removed. Candidate pads c2_flat 357→401 + column-overflow guards (+476B).
Fresh (crash-tolerant gate): cached 2500 inc 2, cand 0; uncached 10000 inc 11, cand 0 —
all divergences = incumbent crash instances. TRAPS: ORT OneHot needs i32 depth (i64 depth
kernel unregistered); stock fresh_verify aborts on incumbent inference errors — scratchpad
fresh_gate.py counts them as fails (consider upstreaming).

## S11 (2026-07-03) — mech-15/pointer scout: KILL — output = data-dependent blue sprite-template scatter at searched anchors (scale 1-3, per-instance shape); cost = detection slices + TopK anchor search, no carrier. Same bucket as 233/285 (assignment/detection).

## 2026-07-10 scatter-guard elision probe — REJECTED

Tested a local tail golf on the current live/source graph: delete the
`Where(valid_idx, idx, sentinel)` guard before the final local
`ScatterElements`, feed the raw cast indices directly, and switch the scatter
to `reduction='max'` so inactive `False` writes cannot erase active `True`
writes.

Result: reject immediately.

- ORT kernel wall: `ScatterElements` with `reduction='max'` does not support
  `bool` data in this environment.
- Semantic wall: the guard is not just overwrite protection.  Raw inactive
  indices can exceed the 17x20 local flat range (`idx=346/347/350` observed
  against valid `[0,339]`), so removing the guard also introduces out-of-bounds
  failures.
- Counterfactual score if it had loaded: shape inference dropped memory from
  `12908` to `12636` by deleting the `[136]` guarded index tensor, so the idea
  was worth probing, but it is not admissible under ORT + real index bounds.

Transferable negative insight: for sparse local-stamp tails, a sentinel-guarded
index vector may be carrying both overwrite safety and bounds safety.  Do not
assume `ScatterElements(reduction='max')` can replace it, especially on `bool`
data.

## 2026-07-10 padded-4x4 semantic oracle — EXACT on local 266, compile target

Re-opened task101 from first principles instead of trusting the old floor
language.  Built a Python oracle asset at
`reports/candidates/task101/oracle_padded4.py` and verified:

- `train + test + arc-gen = 266/266` exact locally.
- The tight reference component containing blue can be padded into a top-left
  `4x4` canvas **without changing correctness** on local data.
- Reference red count is always tiny on local data: `1` in 22 examples, `2` in
  244 examples.  Hidden-copy red area (excluding the visible reference object)
  peaks at `20` red cells locally.
- Accepted placements are sparse: average `3.23` total placements per example
  (including the visible reference object), max `4`, so hidden copies average
  about `2.23`, max `3`.

Exact oracle mechanism:

1. Find the connected reference component containing blue.
2. Put its tight bbox into the top-left of a zero-padded `4x4`.
3. For scales `3,2,1`, enumerate candidate anchors from visible red cells.
4. Require every scaled red cell to be present in the input.
5. Reject candidates whose full occupied footprint (scaled red + hidden blue)
   touches unrelated nonzero cells by 4-neighborhood.
6. Suppress smaller-scale candidates whose visible red support is already
   covered by larger accepted matches.
7. Paint only the blue cells for accepted matches.

Why this matters: this is the first exact local proof that the task can be
expressed as a **fixed padded template** rather than the current imported
`TopK(k=21)+Gather` exact graph.  The remaining question is compile cost, not
semantic uncertainty.

Next attack: source-owned ONNX candidate that preserves the padded-`4x4` rule
while avoiding dense full-canvas carriers.  Likely compiler shapes to test:
dynamic small-kernel anchor detection vs. a leaner red-pair enumeration path.

## 2026-07-10 dynamic Conv/ConvTranspose semantic compiler — EXACT but KILL

Built a source-owned semantic candidate at
`reports/candidates/task101/cand_dynconv.py` using the verified padded-`4x4`
oracle:

1. extract the reference component in a local `7x7` window via blue-seeded
   flood fill;
2. crop a dynamic padded `4x4` red/blue template;
3. `Resize` it to scales `1/2/3`;
4. detect anchors with dynamic `Conv(red_pad, red_kernel)`;
5. reject adjacency with a bordered ring kernel on `nz_pad_ring`;
6. suppress smaller scales by red-coverage `ConvTranspose -> Conv`;
7. stamp blue with `ConvTranspose` and overlay with `Where(mask, blue, input)`.

Result after fixing the crop/ring bugs:

| candidate | stored | memory | params | points | outcome |
|---|---:|---:|---:|---:|---|
| incumbent | 266/266 | 12908 | 817 | 15.473026 | keep |
| `cand_dynconv.py` | 266/266 | 90745 | 113 | 13.582947 | **reject** |

Why it loses: the dynamic full-canvas Conv/ConvTranspose route recreates large
working planes (`35x39`, `32x36`, `28x32`, `24x28`) for each scale and for both
detection and suppression.  Semantics are clean, but ONNX cost is dominated by
those activation maps, not by params.  This is a direct falsification of the
"exact semantic compiler should beat the imported exact graph" hope for this
particular formulation.

New structural fact kept from the probe: the reference object has only `1` or
`2` red cells on all 266 local examples.  Any future win should exploit that
small red support directly, rather than compiling the whole padded template as
dynamic Conv kernels.

## S12 (2026-07-10) — clamp-validity tautology cleanup (+0.0010) ADOPTED, 266/266

Applied a tiny exact-graph golf in the live/source baseline: three tail checks
were recomputing "did the clamp change this coordinate?" by comparing a value
to its clamped copy:

- `c2 = Equal(cY, c1)` after `c1 = Where(c0, h, cY)`
- `d2 = Equal(dY, d1)` after `d1 = Where(d0, i, dY)`
- `el = Equal(eh, ek)` after `ek = Where(ej, i, eh)`

All three are tautologically just the existing clamp-validity masks:
`Not(c0)`, `Not(d0)`, and `Not(ej)`.  Replaced the `Equal` nodes with those
direct booleans and kept the rest of the graph unchanged.

- Before: `pass 266`, `memory 12908`, `params 817`, `points 15.4730257`
- After: `pass 266`, `memory 12897`, `params 814`, `points 15.4740463`

Transferable: when a graph clamps with `Where(too_high, limit, x)` and later
tests `Equal(x, clamped_x)`, the equality is usually just `Not(too_high)`.
This can delete small but real tail tensors without touching semantics.

## 2026-07-11 +0.01 golf session — no adopt (266/266 floor holds)

Goal: squeeze another `+0.01` from the live/source graph, then rebuild and pack
submission.

Baseline after S12 sync in source: `pass 266`, `memory 12897`, `params 814`,
`points 15.4740463`.  Target needs about `136` bytes off `params+memory`.

Probes (all exact or load-fail; none adopted):

| probe | pass | memory | params | points | outcome |
|---|---:|---:|---:|---:|---|
| incumbent | 266/266 | 12897 | 814 | 15.474046 | keep |
| onnxsim | 266/266 | 12897 | 814 | 15.474046 | 0 delta |
| single-node removal sweep (~278 nodes) | — | — | — | — | 0 wins |
| S12 `Equal`→`Not` re-applied in source | 266/266 | 12897 | 814 | 15.474046 | already at S12 score |
| fp16 channel slices (Slice still counts fp32) | 266/266 | 13949 | 814 | 15.400121 | worse |
| replace `u` init with `Range` | 266/266 | 16977 | 477 | 15.232676 | worse |
| global 900-tail scatter + offset LUT | 266/266 | 13661 | 1718 | 15.359242 | worse |
| eliminate bool `3` via 2D Gather | load fail | — | — | — | rank/shape break on `[9]` gathers |
| ScatterND `[1,1,17,20]` tail | — | — | — | — | index tensor `[136,4]` int64 costs ≫ 340B saved from dropping `ev`/`ew` |

Dominant counted activations (shape inference): two fp32 `[1,1,17,20]` channel
slices (`1360B` each), tail bool/fp16 planes (`ex` 900B, `bc` 680B, duplicate
flat/spatial bool pair `2`/`3` at 340B each, scatter flat+spatial pair
`ev`/`ew` at 340B each).  The duplicate bool/scatter reshapes look removable in
the graph drawing but each named ONNX node output is counted independently, so
eliminating one view without eliminating the op chain does not reduce memory.

Conclusion: this imported exact graph is at a verified local floor for graph
surgery.  The next real win needs a **new source-owned semantic compile** from
`reports/candidates/task101/oracle_padded4.py` that avoids both (a) the dynconv
full-canvas route (`90745` mem, killed) and (b) re-materializing the same
17x21 planes.  Until that lands, keep incumbent and still rebuild/pack for
submission parity.
