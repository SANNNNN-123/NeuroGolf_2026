# task366 — e6721834

## Current live

`memory=35927`, `params=490`, `points=14.497209022434086`.
The deployed graph is a high-quality heuristic; prior fresh sanity was `39/40`, so any rewrite needs
fresh validation.

## Semantic rule

Input contains two same-sized panels, stacked or side-by-side.
One panel is the template: a background plus 2–3 solid rectangles in a single `forecolor`, each with
1/2/3 same-coloured dots punched into it.
The other panel contains only the dot stencils at new positions.
Output is the non-template panel background with the missing full rectangles reconstructed at those
dot positions.

## Bottlenecks

- `label30` one-hot-to-colour Conv: ~3600B.
- template colour-present machinery, including int32 template label lookup: ~1KB plus several
  255/510B tensors.
- repeated `k0/k1/k2` stencil matching blocks: roughly 7KB.
- repeated placement/stamping blocks: roughly 5.5KB.
- final label path: roughly 5.7KB.

## Re-attack angle

Generator fact: punched dot colours exclude both backgrounds and `forecolor`.

Therefore a cheaper template-dot mask may be:

`T_dot = T_non_background AND T != forecolor`

instead of the current “template cell colour appears in placement dots” machinery. If `forecolor`
can be derived cheaply from the rectangle mask, this may remove 2–4KB. If deriving `forecolor`
requires full per-colour counting, it probably gives the savings back.

Larger idea: use the generator guarantee that rectangle `idx` has `idx+1` dots to map 1/2/3-dot
templates to placement clusters and delete much of the `k0/k1/k2` matching. This is much riskier
because count/color collisions exist; treat as research, not an immediate adoption path.

## 2026-06-29 forecolor/dot-mask probe

Hypothesis: replace the current placement-colour membership path

`pos_color -> T_color_present -> GatherElements(T_idx_for_present) -> T_present`

with `T_dot = T_non_background AND T != forecolor`, using the generator fact
that dot colours exclude both backgrounds and `forecolor`.

Generator probes:

- First non-background template cell is a dot colour in about `30/300` samples,
  so it is unsafe as a cheap `forecolor` proxy.
- Component first-cell/majority proxies are also unsafe (`~8-40%` failures in
  1000-sample probes depending on proxy).
- Template non-background mode is safe in tested samples (`0/300` dot-colour
  collisions), because rectangles dominate the dot pixels.

Conclusion:

The semantic fact is valid, but the safe `forecolor=mode(non-bg)` route likely
needs per-colour counting over the template panel.  That may cost as much as or
more than the current `T_present` path.  Do not patch until a cheap mode extractor
is designed.

## 2026-06-29 fresh rare-failure capture

- Public candidates for task366 (`boristown`, `lucifer`, `biohack_mix`, `urad`) are identical to live/source on stored eval: `pts=14.497209`, `mem=35927`, `params=490`, `pass=255/255`.
- Fresh sanity in this session: `40/40`, then `199/200`.
- Captured a failing eligible fresh case (`input_shape=(26,11)`, `output_shape=(13,11)`, `diff_count=20`) where the target should reconstruct a 5x4 rectangle from a single placement dot color `3`, but prediction left the placement panel unchanged.
- The visible template panel contained the same dot color `3` in another box/dot-count context. This confirms the earlier risk: dot colors are not unique per rectangle index, so color-present routing alone can bind the placement dot to the wrong template evidence.

Conclusion: a score-improving semantic rewrite must route by geometric dot stencil/count and placement cluster, not only by dot color membership. The cheap `T_dot = T_non_background AND T != forecolor` idea remains useful only if paired with a reliable stencil/count association.

## 2026-06-29 — uint8 TopK mask enumeration trim adopted

- Previous source/live: `pts=14.497209022434086`, `mem=35927`, `params=490`.
- Rewrite: five bool-mask `TopK` inputs (`A_nb8_flat_f`, `P_nb_flat_f`,
  `k0/k1/k2_anchor_flat_f`) were changed from bool→fp16 full-vector casts to
  bool→uint8 full-vector casts. The small `TopK` value outputs are then cast
  back to fp16 for existing downstream `Greater`/`ReduceSum` consumers.
- Stored eval: `255/255`, `mem=34678`, `params=490`, `pts=14.532108142796652`.
- Fresh truth nuance: both incumbent and candidate have rare failures against
  the current local generator, but the candidate matched the incumbent exactly
  over `20000/20000` eligible fresh examples.

Insight: bool-mask `TopK` cannot run directly in ORT, but bool→uint8→TopK plus
small value-output recast can remove large fp16 mask casts while preserving the
same selected indices and downstream semantics.

## 2026-06-29 dot-count semantic reference probe

Read the generator directly.  Strong constraint confirmed: template index `idx`
has exactly `idx+1` punched dots, and placement boxes reuse the same relative dot
stencil.  This suggests a colour-collision-safe rewrite should route by
1/2/3-dot geometry rather than dot colour.

A quick Python semantic reference using `forecolor=mode(non-bg)`, forecolor
components as template rectangles, and per-colour relative-stencil placement
matching was **not** exact: `1137/2000` fresh.  The simplification fails because
punched dot holes can split a forecolor rectangle into multiple 4-connected
components, and same-colour/multiple-placement cases need a stronger rectangle
reconstruction step before matching.  Do not lower this to ONNX yet.

Current conclusion: dot-count/stencil routing remains the right high-ceiling
direction for task366, but the semantic compiler needs a robust template
rectangle extractor (probably bbox/gap based, not component-only) before it can
replace the live 640-node heuristic.

## 2026-06-29 exact-cover semantic compiler probe

Upgraded the Python reference from component extraction to a more faithful
mathematical compiler:

1. infer candidate template panel/background/forecolor,
2. enumerate all 2..7 by 2..7 rectangles whose cells are non-background and are
   exactly `forecolor` plus `k in {1,2,3}` same-colour punched dots,
3. choose one rectangle for each dot count `k` by exact cover of all template
   non-background cells,
4. enumerate placement anchors whose relative dot stencil exactly matches and
   whose rectangle interior contains no extra placement dots,
5. exact-cover all placement dots and reconstruct the output rectangles.

This is a much better semantic model than the component-only probe: **987/1000**
fresh on the first run.  Adding a simple prior that the template panel has more
non-background cells than the placement panel gave **2950/3000**, so the remaining
errors are not solved by a crude panel-size prior.

Interpretation: the high-level mechanism is real, but rare ambiguity remains in
template/placement selection or in exact-cover tie-breaking.  The next useful
step is to capture failing cases and inspect them in the trace/transform viewer,
not to lower this compiler to ONNX yet.  A successful version would be a genuine
large rewrite candidate because it replaces much of the current colour-membership
and repeated k0/k1/k2 heuristic routing with bounded rectangle/stencil exact
cover.

## 2026-06-29 exact-cover bg-candidate fix

The `987/1000` failure mode was not primarily dot-count ambiguity.  The wrong
assumption was `template_background = mode(template_panel)`: large foreground
rectangles can cover more cells than the template background.

Probe update:

- keep placement background as panel mode, because the placement panel contains
  only sparse dots on its background;
- enumerate template background candidates among colours with count `> 3`
  (dot colours occur at most three times);
- choose foreground and 1/2/3-dot rectangles by bounded exact cover;
- use an uncovered-cell exact-cover recursion so false background candidates do
  not explode combinatorially.

Fresh reference results:

- `1000/1000` with seed 0;
- `5000/5000` with seed 1.

Interpretation: task366 now has a clean source-level mathematical specification:
split panels, infer the placement side by sparse dot panel, enumerate candidate
template backgrounds rather than trusting the mode, extract one rectangle per
dot count by exact cover, then stamp matching dot stencils into the placement
panel.  This is not adopted yet because lowering this exact-cover compiler to a
smaller ONNX graph is still the hard part.  The key transferable lesson is that
majority-colour/background heuristics are unsafe when generated rectangles can
dominate the panel area.

## S8 (2026-07-02) — output-preserving surgery wave (+0.143) ADOPTED
NO iterative flood here (~60×255B bool planes + 8.5KB scalar soup) — literal walk-einsum N/A.
Landed: has8-count as ONE 4-op Einsum vs free input (deletes 6 planes + fp16/TopK-8 chain);
bgT-mux recompute of T_nb0/P_nb; k-anchor TopK feeds shared as one fp16 presence Gather + Where
(deletes 3×(mask+flat+cast); TopK feeds stay fp16 = grader-safe); c-block first-cell via 4D
ReduceMax/ArgMax; mask re-association (255B→15B); Not(Less)→GreaterOrEqual fusion ×26;
row/col-any planes → Einsums vs free input. 30983+576 vs 35927+490 → 14.497→14.640.
Divergence 0 on ~9.5k fresh + stored + 500 random (arithmetic identities). TRAPS (transferable):
ORT CPU has NO bool Where kernel; np.ascontiguousarray promotes 0-d→(1,) (use np.asarray);
rank-0 initializers need value_infos rewritten to [] for strict inference.
Floors: label30 3600B, T_idx 1020B (Gather idx must be int), out_label30 900B Pad.
Adopted via ONNX materialization + live_to_exact_source.

## S9 (2026-07-03) — fold 2nd pass: FLOOR re-confirmed (no change)
13a N/A (no walk/flood einsum; output = index-shift Gather stamping). Batched-K on the
12-fold rectangle banks = byte-identical (outputs counted, nodes free; params only 576).
Byte-rank floors: label30 3600 fp32 Conv read, label30_u8/out_label30 900 each,
T_idx_for_present 1020 int32 (GatherElements needs int), TopK fp16 feeds 5×510,
B_rows_gather 450 (row-first already minimal). Remaining ~21.5KB = flat tail of needed
255B masks/int32 scalars across replicated rect machinery. Not→GEq fusion exhausted;
free-input einsum inapplicable (masks also Gathered; derive from T_nb0 not input).
Only ceiling-lifter = exact-cover semantic compiler (un-lowered, research). DO NOT re-probe.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — stamped template rects with punched dots are 2-D non-separable AND all colours are instance-dependent (no constant signed W exists); ~21.5KB = replicated rectangle-extraction/stencil-match blocks (assignment machinery). Priority already via Where chain, no [30,30] carrier.

## 2026-07-06 golf re-attack — NEGATIVE (batching is byte-neutral)

Confirmed incumbent = cheapest known: measured ALL public dumps (kojimar/urad7225/bobmyers/
lucifer, incl. 7220-7225 tier) — every one is 33.8K-35.9K mem. Our `networks/task366.onnx`
(30983) is the global minimum. No borrow available.

**Key correction (was the whole premise, and it was WRONG):** grader memory =
`sum over EVERY intermediate tensor of (num_elements * dtype_itemsize)` (harness
`calculate_memory`). Collapsing the 3x unrolled `c0/c1/c2`, `k0/k1/k2`, `r0/r1/r2` blocks
into one batched block over a size-3 axis turns three `[1,255]` planes into one `[3,255]`
plane = **identical bytes**. Batching cuts tensor COUNT, not total element-bytes. So
"vectorize the unrolled blocks" gives ZERO memory reduction. Do not re-attempt.

**What actually costs (measured):** 3600B `label30` detection Conv (FLOOR, static 30x30
colour read) + ~10.5KB of ~41 full-panel `(1,1,15,17)`=255B uint8/bool mask planes + a
~16KB tail of ~200 small int32/bool coordinate tensors. Memory is a function of the NUMBER
of distinct plane/coord tensors the algorithm materializes, i.e. algorithm structure.

**dtype golf ≈ 0:** the only >255B downcastable planes (`k*_anchor_flat_f`, `presF_flat`,
510B f16 each) all FEED TopK — uint8 TopK crashes the grader. `label30_u8` already minimal.

**Leaner rewrite: high-risk, not smaller.** From-scratch correlation solver prototype
(match template dot-pattern vs stencil, stamp rect) scored 220/2855 = 7.7% error, 38x the
0.2% bar. Failures = dual-background + large-forecolor cases that defeat most-common-colour
bg detection — exactly what the incumbent's corner-voting `A_tl_eq_tr`/`A_bg` machinery
(~40 nodes) exists to handle. A *correct* correlation solver still needs several full-size
planes per box (color mask + correlation response + rect mask) => memory floor near the
incumbent.

**16.0 (mem+params <= 8103) is INFEASIBLE.** Hard floor of the current structure alone =
3600 (Conv) + 15503 (1-byte masks) = ~19.1K => ceiling ~15.15 even with perfect dtype golf.
The whole public field at 30K+ corroborates. VERDICT: keep incumbent, de-prioritize task366.
