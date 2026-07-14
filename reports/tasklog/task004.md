# task004 — 025d127b

**Rule:** Grid (size 8..16, variable H,W) holds 0..4 axis-stacked slanted parallelogram
outlines, one DISTINCT random colour each, separated by a full blank row (shapes are
ROW-SEPARABLE, never share a row). The output un-slants each shape's outline: every shape
row shifts RIGHT by +1, EXCEPT the shape's BOTTOM row (shift 0) and the rightmost pixel of
the SECOND-TO-LAST row (also shift 0). Colours simply copy the input colours. Reformulated
to a fully-local per-cell partition (verified exact 3000/3000, zero collisions):
rowany[r]=ReduceMax(occ,cols); below[r]=rowany[r+1]; below2[r]=rowany[r+2];
is_bottom=rowany∧¬below; is_2ndlast=rowany∧below∧¬below2;
special[r,c]=occ[r,c]∧is_2ndlast[r]∧occ[r+1,c] (occ pixel directly below);
copy_cell=occ∧(is_bottom∨special); shift_cell=occ∧¬copy_cell;
L_out=shiftR1(colf·shift_cell)+colf·copy_cell.
**Current:** 14.08 pts, gen:thbdh6332, mem 54000, params 1020
**Target tier:** A — output colours COPY arbitrary input colours (Tier S route blocked: a
fixed Conv can't route random per-instance colours), but the whole map is a separable
per-row shift collapsed into ONE colour-index value plane, no [1,10,H,W] product.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30 fp16 planes, single L plane + ingrid mask | A | 43926 | 32 | 14.31 | n/a | works, MARGINAL (+0.22) |
| 2 | crop all working planes to 17×17 (active-region escape) | A | 25152 | 38 | 14.87 | 200/200 + 500/500 | ADOPT (+0.78) |
| 3 | fold in-grid into the colour Conv (ch0 weight=0.5 sentinel) — kills the redundant 30×30 fp32 ReduceMax `ingrid30` (3600B) + `ingrid_f` (1156B) | A | 20396 | 38 | 15.075 | 200/200 | +0.21 vs #2 |
| 4 | crop W=17→16 (measured input/output coloured extent ≤col15 over 30k fresh; col-16 shift overflow is a clamped no-op) | A | 19118 | 38 | 15.140 | 500/500 | +0.27 vs #2 |
| 5 | opset-11 int32 `Pad` (ORT accepts it!) — Cast Lmask→int32 at 16×16 then int32-Pad, dropping the 1800B fp16 30×30 bridge plane `L30` | A | 18342 | 72 | 15.179 | 200/200 | +0.31 vs #2 |
| 6 | `shift_cell = occ − copy_cell` (copy_cell⊆occ) drops the Sub(1)+Mul `notcopy` plane | A | 17830 | 72 | 15.207 | 500/500 | **ADOPT (+0.34)** |

## Best achieved
15.207 @ mem 17830 params 72 — recommend adopt: Y. Beats prior 14.87 by +0.34 (≥ +0.3).
Remaining dominant intermediates: `colf30` (3600B fp32 Conv entry — the one mandatory 10→1
reduction, fp32 forced by fp32 input) and `L30_i32` (3600B int32 — the mandatory Equal input,
opset-10/11 Equal accepts only int32/int64/bool). Everything else is a 16×16 fp16/bool working
plane (≤1024B). The two 3600B planes are the irreducible floor of this encoding.

## Irreducible-floor analysis
Dominant intermediates: colf30 (3600B fp32 Conv entry — the irreducible 10→1 colour-index
plane), ingrid30 (3600B fp32 ReduceMax-over-channels, needed to distinguish in-grid-bg from
off-grid since both have colf=0), L30 (1800B fp16 padded value plane) and its int32 cast
L30_i32 (3600B, forced because opset-10 Equal rejects fp16/uint8 — only int32/int64/bool).
Everything else is a 17×17 fp16/bool working plane (≈300–600B each). The 17×17 active-region
crop (generator bounds H,W ≤ 16, +1 col of shift headroom) is the lever that took it from
44k → 25k.

## OPEN ANGLES (re-attack backlog)
- DONE in #3: folded in-grid into the colour Conv (ch0 weight=0.5 sentinel) — killed `ingrid30`
  (3600B) + `ingrid_f` (1156B). off-grid=0 / in-grid-bg=0.5 / coloured=k from ONE plane.
- DONE in #5: int32 `Pad` IS legal under opset-11 ORT (the tasklog claim "Pad rejects int32" is
  opset-10-only) — Cast Lmask→int32 at WxW then int32-Pad, killing the 1800B fp16 30×30 bridge.
- The two remaining 3600B planes (`colf30` fp32 entry, `L30_i32` int32 Equal-input) are the hard
  floor: the entry must be fp32 (fp32 input → fp32 Conv), and opset-10/11 Equal accepts only
  int32/int64/bool, so the final colour-index 30×30 plane must be int32. Routing the one-hot
  expansion as a Concat-padded bool [1,10,WxW] measured WORSE (the 10-ch partial-pad bool plane
  ≥5100B). No cheaper path to the 30×30 int32 Equal input than fp16-mask→cast-at-WxW→int32-Pad.
- The fp32 `colf` (1024B WxW Slice) and `Lmask_i32` (1024B) are the next tier; both are
  structurally needed (Slice preserves fp32; Where→Cast for the int32 Pad). Marginal.

## INSIGHT (transferable)
⭐ A per-shape geometric SHEAR/"un-slant" that looks like it needs shape segmentation is
fully LOCAL when shapes are row-separable: classify each row by 1-cell vertical occupancy
neighbours (rowany[r], rowany[r±1,2]) and characterise edge-case pixels by their immediate
vertical neighbour (here the special "rightmost of 2nd-to-last row" = an occupied cell with
an occupied cell directly below in a 2nd-to-last row) — NO flood-fill, NO rightmost/argmax
scan. Then a colour-COPY remap collapses to ONE colour-index value plane
L=shiftR1(colf·shiftmask)+colf·copymask (masks partition the occupied cells with zero
collisions, so Add == Or). Pairs with the active-region crop (generator size bound) for the
big byte win — the 30×30 floor only really bites the two fp32 entry planes and the final
int32 Equal plane.

## 2026-06-30 source-owned live rewrite

Current source/live before this pass was already a compact public-teacher reconstruction,
not the older 15.207 graph above:

- incumbent: `custom`/teacher exact, `points=16.406957496300326`, `mem=5300`, `params=94`;
- adopted: `custom:task004`, `points=16.45421935173185`, `mem=5044`, `params=101`,
  stored `265/265`.

Mechanism: keep the 16x16 scalar colour workspace, but replace
`below_any = Gather(any_color, axis=2)`, `right_any = Gather(any_color, axis=3)`,
`endpoint = Greater(below_any, right_any)` with a single uint8
`QLinearConv(any_color, [[0,1],[2,0]], pads=[0,0,1,1]) -> end_code`, then
`endpoint = Equal(end_code, 2)`.  This packs `2*below + right`; endpoint is exactly
`below && !right`.  Net result saves one 16x16 uint8/bool plane (256B) after paying
small scale/zp/kernel params.

Important negatives:

- 15x15 crop fails 64 stored `arc-gen` cases, so the 16x16 active box is real.
- Replacing colour-derived occupancy with `valid` is wrong because in-grid black cells are
  valid but not occupied.
- Shifting the raw code directly is wrong because the move/stay condition belongs to the
  source cell, not the destination cell.
- Bool occupancy rewrite with sentinel-preserving value path measured worse/failed.

Answer to the user hypothesis: size and exact position are not the expensive part in the
current graph.  They are already encoded cheaply by the dilated 2x2 `Conv` that emits a
16x16 colour/sentinel plane, plus fixed gather/pad indices.  The remaining cost is mostly
the mandatory fp32 `code_f` entry plane, the padded 30x30 scalar output plane before final
`Equal`, and repeated 16x16 condition/value planes.

Follow-up mechanical cleanup: reuse existing uint8 scalar `u0` as the QLinearConv
zero-point instead of carrying a separate identical `zp` initializer.  Stored eval
unchanged `265/265`; params `101 -> 100`; points
`16.45421935173185 -> 16.454413734082543`.

## 2026-06-30 Session S1 verdict — FLOOR (compact encoding at optimum)

Incumbent: `custom:task004`, points=16.45441, mem=5044, params=100. Re-confirmed
1500/1500 fresh-generated instances + 265/265 bundled (0 divergence).

**Per-tensor mem breakdown (measured):**
- `code_f` 1024B fp32 — the mandatory 10→1 channel-collapse entry (16×16 dilated Conv).
  fp32 forced by fp32 input; per structural-ceiling this is the detection floor.
- `scalar_full` 900B uint8 — the 30×30 carrier for the arbitrary per-cell colour
  output (un-slanted shapes are NOT separable rects → a real carrier is required;
  uint8 is already the min byte-width; Equal(palette) routes to the FREE output last).
- 12× ~256B 16×16 uint8/bool working planes (code, valid, vals, any_color, end_code,
  endpoint, move_cond, move_vals, stay_vals, shifted, scalar_out, scalar_valid) +
  3× 16B strips (row_has). Each is load-bearing for the local row-classification +
  right-shift partition (copy vs shift masks are per-cell because the special
  "rightmost-of-2nd-last-row" pixel is per-cell).

**Empirically falsified golf attempt:** dropping the off-grid re-mask
`Where(valid,scalar_out,255)` and padding `scalar_out` directly → 263/265 bundled
FAIL, 1980/2000 fresh divergent (a shape's rightmost coloured col DOES shift into the
off-grid col, leaking colour). The plane is essential — confirms the tasklog claim.

**No sub-4-plane formulation found** for the move/stay/shift/merge block; the 12
per-cell masks are structurally required. The +1.75 min_stat headroom is an artifact
of its floor=900 assumption (which presumes a separable/count output); this task's
output is a genuine per-cell colour map → floor ≈ 1024 (entry) + 900 (carrier) +
~12×256 (masks) ≈ current 5044. **VERDICT: FLOOR at the compact encoding. No land.**

**Tooling finding (broadly relevant):** all 400 arc-gen generators ARE present under
/tmp/arc-gen/tasks/ (load with sys.path.append('/tmp/arc-gen') — note its own `src/`
shadows ours if inserted at path[0]). The incumbent passes 1500/1500 fresh, so REAL
generalization fresh-gating is available — contradicts the playbook/memory "generator
wiped → equivalence-to-incumbent only" claim. Re-fits can now be locally fresh-gated.

## 2026-07-01 task001-insight pass

Rechecked with the task001 strategy: remove intermediate carriers where direct
`output` routing is cheaper, and exploit the exact colour domain.

Current source/live remains:

- **memory 5044, params 100, pass 265/265, points 16.454413734082543**.

Measured memory breakdown:

- `code_f [1,1,16,16] fp32`: **1024**.  This is the 10-channel one-hot input
  collapsed to a scalar colour/sentinel plane.  The dilated `Conv` is already the
  cheap way to crop and collapse without materializing a 10x16x16 slice.
- `scalar_full [1,1,30,30] uint8`: **900**.  This is the final arbitrary-colour
  scalar plane before `Equal(palette) -> output`.
- twelve 16x16 uint8/bool planes at **256** each for validity, occupancy,
  endpoint, move/stay/shift, and scalar merge.
- three 16-byte row strips.

Checked implications:

- Replacing `scalar_full -> Equal(palette)` with `Equal(scalar_valid,palette) ->
  Pad(output)` would materialize a `[1,10,16,16]` bool block (**2560 bytes**),
  worse than the 900-byte scalar carrier.
- Direct final-output routing would need either 16x16-to-30x30 spatial selectors
  or an equivalent full-canvas one-hot carrier; both are larger than the current
  scalar `Pad`.
- The move/stay/shift block can be algebraically rearranged
  (`shifted values + shifted condition` instead of `move_vals -> Gather`), but it
  still needs the same number of 16x16 working planes.  No task001-style factor
  sharing removes a plane here because the move condition belongs to the source
  cell while the shifted value is consumed at the destination cell.
- Colour handling is already optimal for arbitrary copied colours: keep a scalar
  colour plane until the final free `output` expansion.  Expanding colour earlier
  loses.

Conclusion: no new adoptable improvement.  For task004, the task001 lesson
confirms the incumbent design rather than replacing it: defer one-hot expansion
to the final `Equal`, and keep all geometry in one-byte scalar/mask planes.

## 2026-07-01 deep generator/bounds recheck

Generator `/tmp/arc-gen/tasks/task_025d127b.py` is input-deterministic.  The
visible slanted outlines fully determine the output shear; no hidden generator
state affects the target.  Incumbent verification: **1000/1000 fresh, 0 fail**.

Generator bounds:

- `width,height` are independently sampled in `[8,16]`.
- shapes start at row `1`, are row-separated by one blank row, and satisfy
  `row + tall <= height`.
- `wide in [4,width-1]`, `tall in [3,wide-1]`, `col <= width-wide`.
- output never needs a coloured pixel at column 0, but column 0 and row 0 are
  still in-grid background for many examples.

Deep recheck of possible reductions:

- Cropping the workspace below 16x16 is not a free win.  Row 0 / col 0 may be
  semantically all background, but they are still inside the variable grid.  The
  final full-canvas scalar plane must distinguish in-grid background value `0`
  from off-grid sentinel `255`; a single `Pad` constant cannot both create
  top/left in-grid background and right/bottom off-grid sentinel.  Removing
  row0/col0 would require an additional validity carrier, which is larger than
  the saved 16x16 strip.
- The 2x2 dilated `Conv` is doing two jobs at once: channel collapse and active
  16x16 crop.  A 1x1 colour conv would cut kernel params but would emit a
  30x30 fp32 plane before cropping, which is much more expensive than the current
  1024-byte `code_f`.
- `sidx`/`bidx` gather vectors are parameter-heavy but memory-cheap.  Replacing
  them with `Pad+Slice` style shifts removes some params but adds 16x16 or
  16x17 intermediate planes, losing on total cost.
- Sparse Conv weights would help the 2x2 `Wc` params in theory, but sparse Conv
  was already rejected by official shape inference in the task001 sparse probe.

Conclusion: task004 is a real compact optimum under the current operator family.
The apparent 15x15/top-left crop opportunity is blocked by the in-grid-background
vs off-grid-sentinel distinction, the same reason `scalar_valid` and
`scalar_full` remain load-bearing.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.001)
**Mechanism (op-census diff):** Replaced the **QLinearConv** label lookup with a plain `Gather` lookup (Gather 2→4, Greater+Gather replace Min/Equal), dropping the quant constants `scale`/`wend`/`u1`/`u2`. Mem unchanged; −7 params.
**Old→new:** mem 5044→5044, params 100→93.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task004_pre_s10.onnx`; source `public_candidates/bobmyers7186/task004.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k3(cost 5137): 0 contradictions이지만 18 stubborn hinge viols, LP로 infeasible 증명(ch0/ch2) — 그 18 viol 패치가 ~모든 인스턴스에 등장(val 97% fail). k5: 75k 패치 ~2.4k viols 고착. SGD로도 wall 유지. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
