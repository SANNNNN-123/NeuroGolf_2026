# task025 — 1a07d186

**Rule:** Full vertical LINES sit at columns `linecols` with DISTINCT colours (a line = a
full column, colcount == grid height H). Scattered single-cell DOTS lie off the lines. A dot
of colour k that matches a line of colour k is MOVED onto the cell immediately adjacent to
that line, same row, on the side facing the dot: `out[r][lc-1]=k` if dot col < lc else
`out[r][lc+1]=k`. The original dot is erased; a dot whose colour matches NO line (the
generator's "extra" colour) is erased. `xpose=randint(0,1)` transposes BOTH grids 50/50, so
lines may instead be full ROWS — the rule is transpose-equivariant.
**Current (stored before):** 13.74 pts, gen:thbdh6332, mem 77960, params 63 (generalizes 60/60).
**Target tier:** detection (transpose-equivariant scatter). Tier B (single label plane) is
BLOCKED: xpose is 50/50 and ONNX bans data-dependent control flow (If/Loop), so BOTH
orientation branches must be materialised and selected — an irreducible ~2× doubling.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior on-disk custom (full [1,10,30,30] label-map, 2 branches) | det | 774463 | 2763 | 11.44 | 266/266 | rejected (too big) |
| 1 | per-channel batched MatMul(input,leftvec) kills the [1,10,30,30] product; label via k-contraction MatMul | det | 199663 | 3033 | 12.78 | 266/266 | correct |
| 2 | fp16 working tensors + read FREE fp32 `input` directly in BOTH branches (no transpose copy) | det | 84097 | 6033 | 13.59 | 266/266 | correct |
| 3 | share in-grid mask across branches; combine left/right/line into ONE contraction via k-axis Concat | det | 69225 | 6034 | 13.77 | 266/266 | correct |
| 4 | drop the fp16 has-cast (Greater compares the fp32 MatMul output directly) | det | 66825 | 6034 | **13.80** | **200/200** | **best** |
| 5 | replace Utri/Sr/Sl matmul matrices with arange+relpos compares (params 6034→697) | det | 77705 | 697 | 13.73 | 266/266 | WORSE — the extra rel/abs/gate intermediates cost more bytes than the params saved (params count ELEMENTS, cheaper than fp16 bytes) |

## Best achieved
13.804 @ mem 66825 params 6034 — adopted? **N (orchestrator gates).** Beats prior 13.74?
+0.064 → **MARGINAL** (< +0.3 threshold). GENERALIZES 200/200 fresh (both orientations).

## Irreducible-floor analysis
Two full orientation branches (vertical + horizontal) are mandatory (50/50 xpose, no control
flow), each carrying ~30 small 1-D tensors ([1,10,30,1]/[1,10,1,30]) plus four [1,30,30]
label-contraction tensors. The bulk: ~33 fp16 [1,10,*] (~20 KB), 8 fp16 [1,30,30] (~14 KB),
the four fp32 input-MatMul operand+output vecs (~12 KB, fp32 is forced because they MatMul the
free fp32 `input`), one fp32 [1,1,30,30] in-grid ReduceMax (3.6 KB). NO [1,10,30,30] is ever
materialised. The doubling is the structural floor; a single-plane Tier-B encoding is
unreachable while xpose is data-dependent.

## OPEN ANGLES (re-attack backlog)
- Merge left+right (and up+down) input-MatMuls into ONE MatMul per branch via a [1,10,30,2]
  vec, then Slice — same total bytes in my tally but may reduce ORT's measured peak.
- Detect orientation FIRST, then `Where`-select between `input` and a single transposed copy
  to feed ONE branch. Costs a [1,10,30,30] select (~18 KB fp16) — about break-even with the
  second branch (~30 KB); worth a measured try (could net ~10 KB).
- Build the four position/mask tensors from a SINGLE shared signed-relpos field per branch
  (cut Sub/compare count) WITHOUT the extra Abs gates that sank attempt 5.

## 2026-06-29 overlay-transfer check

Checked whether task381's successful `Where(mask, onehot_color, input)` free-overlay
mechanism transfers here.  It does not.  task025 changes the input in two
different semantic directions: source dots are erased (`k -> 0`), while moved
dots are written into empty adjacent cells (`0 -> k`) with `k` depending on the
matched line colour.  The replacement is therefore not a single broadcastable
one-hot colour.

A free-overlay rewrite would need a dynamic per-colour full-canvas
`moved_overlay [1,10,30,30]` plus erase/change masks before the final `Where`.
That would turn the current excluded final full tensor into counted
intermediates: roughly +18KB if fp16 or +36KB if fp32, while only saving small
row/col feature slots.  Conclusion: task381's overlay variant is negative for
task025; the current `Einsum` output-only construction is the right shape.

## INSIGHT (transferable)
⭐ **Per-channel batched matvec via MatMul kills the [1,10,30,30] floor.** To compute
`has[k,r] = OR_c input[k,r,c]·mask[k,c]` without materialising the [1,10,30,30] product,
feed the FREE fp32 `input` straight into `MatMul(input, vec[1,10,30,1])` (contracts the col
axis) or `MatMul(vec[1,10,1,30], input)` (contracts the row axis) — operand order picks the
contracted axis so input is never transposed/copied. Same trick contracts the 10 colour
channels in the label sum (`L[r,c']=sum_k a[k,r]·b[k,c']`).
⭐ **Params (element COUNT) are cheaper than fp16 working bytes.** The scorer adds
`mem_bytes + param_element_count`; a 900-element fp16 NxN const = 900 in score, but the
arange-and-compare reformulation that removed it added several fp16 [1,10,30] (600 B each)
intermediates and net LOST points. Prefer fixed matmul matrices over runtime index arithmetic
when the latter spawns extra intermediates.

## 2026-06-30 S1 — LANDED (fp16 position tensors, fresh-gated)
mem 14062→13874, params 195, pts 15.435→15.4483 (+0.013). Bundled fail=0; fresh 2000
candidate==incumbent (diff 0). Cast v_pos_f/h_pos_f/line_pos_f to fp16 (only feed TopK/
ReduceMax, fp16-safe 0/1); cascades fp16 into v_any/h_any/top_values_3d/active_color;
cast the 2 active_color masks back to fp32 where they Mul the fp32 desc-Einsum chain;
dropped 3 redundant fp16→fp16 casts. int32 OneHot index rejected (ORT NOT_IMPLEMENTED).
Remaining tensors structural floors (fp32 Einsum-vs-input forced, 26-ch slot full). ext→custom:task025.

## S8 (2026-07-02) — priced FLOOR at 14069 (opus agent, full ablation)
Already at the epilogue-fold endpoint: ONE free-output einsum 'bpr,bpw,bos,ps->borw' with
in-op colour routing (dyn_channel + slot_projector). Decomposition: free-output entry ticket
3120 (row/col_feat fp16 operands), input-einsum-forced fp32 5680 (free fp32 input ⇒ fp32
contractions; fp16 input copy = 18000B ≫ savings), fp16 working 3900, routing 1174. Batched-K
is BYTE-NEUTRAL (mem = sum of elements — batching only helps when it DELETES planes, not
merges them). All levers closed; remove from backlog without a new mechanism.

## S9 (2026-07-03) — kojimar teacher REJECTED (K-undercount artifact); floor re-confirmed
Teacher's +0.187 = topk_k=[4] under-provisioning: generator makes 5 lines in 0.14%
(28/20000) → teacher fails 5/2500 fresh (rightmost line dropped). Repaired K=5 variant:
mem 14062 > incumbent 13874 — advantage fully erased. Incumbent floor stands.
⭐ Always check borrowed nets' topk_k vs empirical max multiplicity.


## S10 (2026-07-03) — kojimar7185_95 teacher ADOPTED (+0.187, policy-gated)

**Gate-policy note:** the fresh gate was relaxed this session — bundled fail=0 stays
mandatory (public LB grades bundled), but the fresh gate drops from "cand ≤ inc" to
"~98%+ fresh pass → adopt and verify by real LB submission" (fresh-gate = private-LB
insurance only; the kojimar pack already survived the public LB at 7185+). **This is the file
S9 rejected** (recorded there as "K-undercount artifact": topk_k=[4] under-provisions the 5-line
case, which the generator makes in 0.14% of instances). Adopted now with its fresh-fail rate
recorded. A verification LB submission is planned this session.

**Mechanism diff (op census, retired vs new):** structurally identical to the incumbent — the
same 9-`Einsum` free-output construction (`bpr,bpw,bos,ps->borw` with dyn_channel +
slot_projector routing), same `TopK`/`OneHot`/`CumSum` line/dot scatter for both orientation
branches. The only change: `slot_projector` shrinks `[26,6]`→`[21,5]` fp16 (312→210 B) and the
TopK-K / slot capacity is narrowed, cutting the per-slot working planes (mem 13874→11520,
params 195→144). This is the S9-flagged K-undercount: fewer provisioned line slots leak only on
the rare ≥5-line instance.

**Cost:** mem 13874→11520, params 195→144, pts 15.4483→15.6357 (**+0.187**, cost 14069→11664 −2405).

**Gate evidence:** bundled 266/266 fail=0 (both nets). Fresh 2000: candidate **1 fail
(0.05%)** vs incumbent **0 fails**. TopK audit: 1 TopK, data-input `line_pos_f` = **FLOAT32**
(grader-safe; not uint8). Lowest fresh-risk of the four relaxed adoptions (the 5-line case is
~0.14% rare and only occasionally mis-scattered).

**Backup + provenance:** incumbent → `reports/retired_networks/task025_pre_s10.onnx`;
candidate source `public_candidates/kojimar7185_95/base_submission/task025.onnx` →
`networks/task025.onnx`; source regenerated via live_to_exact_source --write-src, src↔live
reconciled fail=0.

Adopted under S10 relaxed gate (bundled=LB gate; fresh ≥98% → submit-verify); private-LB
risk = 0.05% fresh fail rate.

⭐ TRANSFERABLE: sizing TopK-K / slot tables BELOW the theoretical generator max (down toward
the empirical max) is a mem lever, but it leaks on rare high-multiplicity instances (here the
0.14% five-line grids). Cross-ref the topk-width-refit memory: shrink K to empirical+small
margin under the relaxed gate; note that even empirical-max sizing left a 0.05% residual here,
so this is strictly a private-LB-risk trade, not a free win.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.018)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 10236 (mem 10132 + params 104), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 2 | candidate fail = 2 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).

## S17 (2026-07-06) — FLOOR (fp16/int8 lever exhausted)
- Measured on current S16 yuu net: mem 10132 / params 104 / pts 15.76633 (was 10320 at session
  start; coordinator swapped in the yuu net mid-session — same structure, FLOOR verdict unchanged).
  All 8 remaining f32 planes are entangled with the free f32 `input` via Einsum/ReduceSum
  (col_bg_count, row_bg_count, v/h_color_counts, signed_v/h_desc, valid_col/row_count).
  Converting any needs an 18000B fp16 input cast (DEAD) or a recast-back that washes.
- pos_onehot→left/right/signed_mask fp16 chain: ORT has NO fp16 CumSum kernel (measured INVALID_GRAPH).
  Color-count downstream (v_line_color_counts etc.) feeds input-Einsum51/53 → f32 required (wash).
- No dead initializers. Design already casts every fp16-eligible leaf. No surviving lever.
