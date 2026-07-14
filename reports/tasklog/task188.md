# task188 — 7b7f7511

**Rule:** The input is a `height x width` colour tile DUPLICATED once along one
axis — vertically (grid = 2h x w, output = top half) or horizontally
(grid = h x 2w, output = left half). `width, height` are each `randint(2,4)`, so
the *duplicated* dimension is even and in {4,6,8} and the *non-duplicated* one is
in {2,3,4}. The output is exactly the unique tile = the top-left `height x width`
block; outside it is background (all-channels-off). NOT a detection task — it is a
crop/un-duplicate (mask the top-left tile out of the one-hot input).
**Current (public):** 15.73 pts, gen:thbdh6332.
**Target tier:** A/B-ish separable mask (NOT S — needs a data-dependent axis
decision + extent). Achieved a near-Tier-A separable rectangle mask: output =
input · (row<keepR) · (col<keepC) with the final op a FREE `Where`.

## Axis decision (R=occupied rows, C=occupied cols; max(R,C)>=4 always)
- R>4 -> VERTICAL (R is dup dim); C>4 -> HORIZONTAL.
- R==4 & C<4 -> VERTICAL; C==4 & R<4 -> HORIZONTAL.
- R==4 & C==4 -> ambiguous: top-half==bottom-half => VERTICAL else HORIZONTAL.
- vert = (R>4) OR (R==4 & C<4) OR (R==4 & C==4 & top==bottom).
- keepR = vert?R/2:R ; keepC = vert?C:C/2.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | extent R,C via ReduceMax/Sum; axis bool from R/C thresholds + 4x4 vtile slice-compare; separable row/col masks; final Where(mask,input,0) | A-ish | 2531 | 85 | 17.13 | 200/200 (20000: 19993) | WIN, beats 15.73 by +1.40 |

## Best achieved
**17.13 @ mem 2531 params 85 — fresh 200/200; 20000-stress 19993/20000.**
Beats public 15.73 by **+1.40**. Adopted? **N** (main adopts via `python -m src.adopt 188`).

## Irreducible-floor analysis
Dominant intermediates:
- **900 B bool keepmask [1,1,30,30]** — the separable AND of the two 1-D masks,
  consumed by the free final `Where`. Irreducible: the Where condition must span
  the 30x30 output; bool is already the cheapest dtype (Mul would force fp16
  1800 B).
- **~1280 B fp32 in the 4x4 tile-equality test** — four [1,10,2,4] (=80-elem,
  320 B) tensors: top/bottom slices, their Sub, and Abs. Only resolves the
  R==4&C==4 ambiguous case (~0.8% of 4x4). Slice preserves fp32; comparing the
  one-hot directly is exact. Could be trimmed but the gain is sub-point.
- Everything else: fp32 scalars (R, C, factors) and 1-D [1,1,30,1]/[1,1,1,30]
  profiles, all tiny.
The genuine accuracy wall is the **R==4 & C==4 doubly-tileable** grid (~0.09% of
all instances): the generator's `vert` flag is then an independent coin flip, so
the output is non-deterministic from the input — same wall the public net hits
(measured 3/5000 fails). Cannot be beaten by any net.

## OPEN ANGLES (re-attack backlog)
- Trim the ~1280 B 4x4 comparison: collapse the one-hot to a colour index before
  comparing, or compare a single representative channel. Each is sub-0.3 pts —
  not worth the complexity, but the path to ~16k-bit-cheaper exists.
- Tier S is blocked: the output is a data-dependent crop (extent + axis depend on
  the input content), so no single fixed Conv/permute produces it.

## INSIGHT (transferable)
⭐⭐ An apparent "duplicate/symmetry detection" task is really an **un-duplicate
crop**: output = input masked to the top-left tile. Compute the grid extent
(R,C) from one-hot ReduceMax profiles, derive the dup axis from the **generator's
range constraints** (dup dim is 2*tile in {4,6,8}, non-dup tile-size in {2,3,4}),
and emit `output = Where(rowmask AND colmask, input, 0)` — the copy/mask collapses
to a near-Tier-A separable rectangle, NOT a detection net. The only irreducible
loss is the small fraction of grids that tile BOTH ways at the minimal size,
where the generator's choice is a free coin flip (non-deterministic — universal
wall, not a thinking gap).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.154)
**Mechanism (real diff):** the incumbent decoded the tile via a **QLinearConv**
(quant `code_w`/`code_scale`/`code_zero` + `Split` into 11 code constants `c_0..c_10`)
producing an fp32 `first9` [1,9,4,4] plane + `first9_u8`/`cropped` tile crop, then
broadcast to a **fp16** [1,10,30,30] output. The teacher net throws all of that out
and emits the output as the **separable rectangle** this tasklog's own INSIGHT
predicted: per-cell [4]-slice constants (`cell_R_C_s/e`, 12 `Slice`), a
`row_mask` [1,1,30,1] ∧ `col_mask` [1,1,1,30] and **one Einsum** to place the colour.
Even though the nominal output is now fp32 (36000B) not fp16, the **grader mem falls
259B** because the QLinearConv quant working-buffers + the fp32 `first9`/`cropped`
tile planes are gone. Ops: QLinearConv/Split/Where/Pad/Equal→0; +Einsum, +Mul×3,
+Slice(5→12), +ReduceMax(4→11).
**Old→new:** mem 1128→869 (−259), params 59→149 (+90). LB 1187→1018.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=3 cand_fail=3 (EQUAL — the
~0.09% doubly-tileable coin-flip wall is inherent to both nets, gate PASS). No TopK.
Backup `reports/retired_networks/task188_pre_s10.onnx`; source `public_candidates/bobmyers7186/task188.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
⭐ **TRANSFERABLE:** empirical confirmation of this task's standing INSIGHT — a
"duplicate/symmetry" output is a **separable row∧col rectangle emitted by one
Einsum**, and that beats a QLinearConv tile-decode on GRADER mem even when the
Einsum path uses a larger output dtype, because QLinearConv drags quant buffers.
Prefer mask+Einsum emission over quantized-conv decode whenever the output is a
solid rect crop.
