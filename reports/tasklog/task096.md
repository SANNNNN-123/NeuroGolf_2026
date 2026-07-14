# task096 — 4290ef0e

**Rule:** Input H×W grid (13..19), bg = most-frequent colour. K (=4..6) non-bg colours each
own a UNIQUE ring index idx∈0..K-1 (a permutation). Ring idx draws its colour at 4 corners
(±idx,±idx) about a per-shape RANDOM centre, each corner with two inward arms of length L_idx;
shapes are scattered and CLIPPED at edges (gen guarantees ≥2 quadrants drawn). Output is a
(2K-1)² concentric reassembly: offset (a,b) from centre (K-1,K-1), m=max(|a|,|b|), n=min(|a|,|b|)
→ colours[m] iff (m<K and n ≥ m−L_m+1) else bg.

## SESSION 2026-06-19 (P=15.88, deployed = ext:kojimar7113 profile net) — INFEASIBLE
**Current champion** is the external kojimar net `networks/task096.onnx` (124 nodes, **15.884 pts,
mem 8278, params 817**) — a fully closed-form PROFILE recovery (no matched-filter conv). It vastly
beats the prior exact-conv build in `src/custom/task096.py` (13.196, mem 132013, which is now stale).
**+0.3 bar = 16.18 (mem+par ≤ 6770).**

**Verdict: INFEASIBLE (+0.3). Best achievable here = ZERO net gain over 15.88.**

### Where the kojimar mem lives (exact, shape_inference × declared-dtype)
| plane | bytes | role |
|---|---|---|
| `row_sum_full` f32 [10,30] | **1200** | ReduceSum(input,[0,3]) per-(channel,row) sum |
| `col_sum_full` f32 [10,30] | **1200** | ReduceSum(input,[0,2]) per-(channel,col) sum |
| `padded_color` u8 [1,1,30,30] | 900 | 30×30 colour-index plane fed to output Equal (fixed) |
| `radius_i32` int32 [11,11] | 484 | Gather index for synthesis canvas (int-only, fixed) |
| `row/col_present_full` bool [10,30] | 300+300 | Greater(sum>0) presence |
| ~14× [11,11] synthesis + [5,19] profile planes | ≤121 each | all load-bearing |

The **two f32 profile planes (2400B = 29% of mem)** are the only ≥1kB targets. The math: removing
BOTH (−2400) → 16.19 (+0.31, would just clear the bar); removing/halving ONE (−1200) → 16.03 (+0.14).
**Neither is achievable** — every shrink route was tested and blocked:
- **fp16-declared reduce output** → ORT type-checker REJECTS (ReduceSum/ReduceMax on f32 input MUST
  emit f32; declaring the value_info fp16 = "Type ... does not match expected type" load failure).
  ⇒ the stale-dtype re-probe lever does NOT apply to reduces (only to Conv, which keeps fp16).
- **width-19 crop** (grid ≤19) → needs `Slice(input,…0:19)` = ≥5700B new f32 plane, net loss; the
  reduce output shape [10,30] is fixed by its input, can't crop post-hoc.
- **Gather-select-then-reduce** (only K≤5 channels used) → `Gather(input, top_colors, axis=1)` =
  [1,5,30,30] f32 4500B before reduce, net loss; selection is data-dependent (TopK) so no static
  channel pre-slice.
- **grouped/no-pad Conv profile** → depthwise `Conv(input, W[10,1,1,30])` keeps f32, same 1200B.
- **fp16-input cast** → [1,10,30,30] fp16 = 18000B + cascading Where/Equal type errors.
- **channel_count redirect** to direct `ReduceSum(input,[0,2,3])` [10] (40B): valid + still 266/266
  stored, but ZERO mem change (the sums are still needed for presence) ⇒ not worth writing.

Outside the sums there is NO removable 2325B mass: `padded_color` (900) is the mandatory 30×30 Equal
input; `radius_i32` (484) is an int-only Gather index on the fixed 11×11 max canvas; the rest are
≤121B and all consumed. So the only path to +0.3 is the two f32 reduce planes, and they are a HARD
floor of any profile-based recovery on the full 10-ch 30-wide input.

**Prior-session content below is HISTORICAL (the 13.196 exact-conv build, now superseded by kojimar).**

---

**Prior bar (session 1):** 12.97 (weak gen-import, 248 nodes). +0.3 bar = 13.27.

## Result of THIS session
**pts 13.196, mem 132013, params 1825, fresh 200/200, stored 4/4 → BEATS 12.97 by +0.226 (MARGINAL, <+0.3).**
The prior agent's "INFEASIBLE — exact net floors at 12.54" verdict was based on a STALE fp32-conv
claim. Re-tested: **fp16 Conv keeps fp16 output under ORT_DISABLE_ALL on the current ORT** (the
"ORT upcasts fp16 conv → 0 pass" claim is false now). That + crop-to-WORK19 + sig-as-1×1-conv +
type-0 drop + uint8 output dropped the exact net from 12.54 → 13.196.

## The net (exact, 0-err over 2100+ fresh + 4 stored)
1. Crop input to [0:19,0:19], cast fp16. `sig_k = ingrid − 2·mask_k` for all k in ONE 1×1 conv
   (W = ones(10,10) − 2·I). Reshape channels→batch [10,1,19,19].
2. Matched filter: `cdo = Conv(sig, K_t)` = og − 2cm, 11 type kernels (idx≥1), fp16, pads
   [5,6,6,5] → [10,11,20,20] so centres reach 1 cell off the left/bottom edge (hand-authored ARC
   examples place a centre at col −1 / row 19). `mind = ReduceMin(cdo)`; `dist = tot + mind`;
   `match = (dist==0)`. **min-idx tiebreak**: ArgMax over the idx-ordered type axis returns the
   first True (a clipped large stamp also fits smaller types; the true type is the MIN-idx match).
3. idx0 = single-pixel colour (tot==1; the ≥2-quadrant rule gives idx≥1 ⇒ tot≥2), handled by a
   Where override (its kernel dropped from the conv to save one type).
4. Scatter (k, L_k) by recovered idx into length-6 ring vectors; K = max(visible idx)+1 (robust to
   bg-coloured invisible inner rings in the hand-authored ARC examples — bg excluded by gen for
   fresh). Invisible (bg-coloured) rings default to bg colour & L=6 so their cells render bg.
5. Closed-form synthesis on an 11×11 centred canvas (gather ring colour/L by m=max(|a|,|b|), gate
   by n>m−L), Where bg fill, crop/shift to (2K-1)² at top-left (Gather by srci = i+cen+1−K, clamp),
   valid-mask off-grid → sentinel 99, uint8 Pad → Equal → BOOL one-hot.

## Memory floor analysis (irreducible for the EXACT net)
| plane | bytes | why irreducible |
|---|---|---|
| matched-filter conv [10,11,20,20] fp16 | **88000** | 10 ch (per-colour masks needed, bg/absent can't be dropped statically), 11 types (every (idx,L) needs its own kernel — corner-only=75% idx err, all 11 occur in fresh), 20×20 = centre range rows[0..19]×cols[−1..18] forced by the off-grid hand-authored centres; uint8/int8 Conv unsupported by ORT ⇒ fp16 is the dtype floor |
| fp32 input Slice [1,10,19,19] | 14440 | the one fp32 entry plane (Slice preserves fp32; must feed the fp16 cast) |
| sig chain (inwf cast + sig 1×1 conv + batch reshape), each [.,.,19,19] fp16 | 3×7220 | front-end; the reshape is layout-only but counts; a grouped conv would drop it but costs +11.9k params (net worse) |
Total ≈ 132k. **+0.3 needs mem+params ≤ 124244** — the gap (~10.8k) is below the conv plane; no
dtype/crop/type-drop trick closes it without breaking exactness.

## OPEN ANGLES (could break the floor, all currently blocked)
- Cheap exact idx recovery (to cut the 11-type conv): bbox/2 = 10% err (clip underestimate); rank
  by extent = 18% instance err; corner-only matched filter = 75% err. The full per-(idx,L) 2-D
  match is the only verified-exact recovery → conv channels×types is load-bearing.
- Off-grid stored centres force conv 20×20 (vs 19×19 fresh-only). Special-casing the 3 boundary
  centres could shrink to 19×19 (~−8.8k) but is fragile / not generalizing-clean.
- Reducing 10 channels (only ≤7 present) needs a data-dependent gather-compact → symbolic-dim trap.

## INSIGHT (transferable)
⭐ **fp16 Conv now keeps fp16 output under ORT_DISABLE_ALL** — re-test before trusting any
"fp32-conv-at-floor" verdict; halved the matched-filter plane (173k→88k). ⭐ **sig = ingrid−2·mask
for all channels = ONE 1×1 conv** (W = ones − 2·I) — folds the cross-channel ingrid sum + the −2·mask
into a single op, no per-channel Mul/Sub. ⭐ **min-idx tiebreak via ArgMax over an idx-ordered type
axis** cleanly resolves the nested-stamp ambiguity (a clipped large stamp matches smaller types; the
true one is the smallest-idx match). ⭐ For a permutation-indexed concentric figure, **K = max(visible
idx)+1**, robust to an invisible (bg-coloured) inner ring — count-of-present is WRONG when bg ∈ colours
(the hand-authored ARC examples) even though the random generator excludes bg.

## S9 (2026-07-03) — drop color_grid_4d Unsqueeze plane (+0.014) ADOPTED
Pad 2D color_grid [11,11]→[30,30] directly; final Equal broadcasts [1,10,1,1]×[30,30].
mem 8108→7987, params 805→801. Bit-identical: uncached fresh 800 (agent) + 600
(orchestrator): inc 0 / cand 0 / div 0. Latency 0.048ms. NOTE random-input div check
inapplicable (data-dependent Gather OOB on invalid instances — expected).
FLOORS re-priced: row/col_sum_full 2400 (fp16 reduce type-rejected by ORT), padded_color
900 optimal (index-first < one-hot-first 1210), radius_i32 484 (Gather needs int32+),
int16 arithmetic dead (Add/Sub/Mul/Div reject int16; Where/Min/Max NOT_IMPLEMENTED).
Backup reports/retired_networks/task096_pre_s9.onnx.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.086)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 7682 (mem 7261 + params 421), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
