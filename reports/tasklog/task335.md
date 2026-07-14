# task335 — d4a91cb9

**Rule:** common.hpwl. Black canvas; red(2) dot at (r0,c0)=start, cyan(8) dot at (r1,c1)=end.
Output = both dots unchanged + a yellow(4) L-path: horizontal seg on row r0 over cols STRICTLY
between c0,c1; vertical seg on col c1 over rows from r0 (incl) to r1 (excl). Grid 10–20 sq, top-left.
**Current (prior):** 16.294 pts, shared `_hpwl.build_hpwl` per-channel double-MatMul (R10 fp16 2400B), mem 5520, params 520.
**Target tier:** B (separable closed-form path routed into the free Where output).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Where + rank-2 mask, f32 1-D, 4 single-color convs | B | 5100 | 731 | 16.33 | — | works, params heavy |
| 2 | v1 but all 1-D in fp16 | B | 4200 | 731 | 16.50 | — | +0.20, conv params still 720 |
| 3 | `_hpwl` double-MatMul, fp16-direct concat (no f32 VrT/VcT) | B | 4200 | 520 | 16.54 | — | +0.25; needs rowin/colin (grid extent) |
| 4 | Where + rank-2 mask + ONE value-conv/axis (red=1,cyan=2) + fp16 1-D | B | 4080 | 373 | **16.599** | 200/200 | ADOPTED, +0.305 |

## Best achieved
16.599 @ mem 4080 params 373 (mem+params 4453) — adopted. Beats prior 16.294 by +0.305 (Y).

## Irreducible-floor analysis
Dominant: the single 30x30 fp16 mask product `onpath_f` (1800B) + its bool `onpath` (900B) = 2700B
(66% of mem). The Where idiom routes the 10-ch expansion into the FREE output but its cond must be a
materialized bool, and the rank-2 L-mask needs one fp16 30x30 plane to build it (MatMul has no bool).
Bounding the product to 18x18 doesn't help: Where needs a [1,1,30,30] cond and Pad rejects bool (padding
the fp16 instead just re-creates the 1800B plane). The per-channel double-MatMul route has a lower 2-D
cost (R10 2400, no bool) but must reconstruct ch0 (grid extent via rowin/colin ReduceMax, ~360B) which
the Where route gets for free by preserving the input background — net the Where route wins (4453 vs 4720).

## OPEN ANGLES (re-attack backlog)
- Break 2700: find a way to feed the fp16 product directly to Where (no separate bool) — not currently
  possible in ORT (Where requires bool cond).
- Narrow the per-channel basis from width-4 to width-3: proved impossible — a single value-signal
  vr∈{0,1,2} cannot isolate the dots (a bilinear form over 9 grid-points is overdetermined), so the
  separate cyan-marker `gr` is load-bearing.

## INSIGHT (transferable)
⭐ For two-endpoint L / wire-route tasks where the path lies on BACKGROUND cells, the whole output is
ONE `Where(on_path_bool, color_onehot, input)` — the input preserves BOTH endpoints AND the background
channel (ch0) for FREE, so you never reconstruct grid extent (rowin/colin), unlike the per-channel
double-MatMul builder. The L-mask is a rank-2 outer product `rowA⊗colA + rowB⊗colB` (A@B, one fp16
30x30 plane). And ONE value-conv per axis (col0->1, col1->2) recovers BOTH dot rows/cols from a single
collapse-conv (red = clip(v) - (v-clip(v)); cyan = v-clip(v)), halving conv params vs separate red/cyan
convs. This dropped the shared `_hpwl` builder from 16.29 to 16.599 (mem 5520->4080, params 520->373).

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 20 (bundled max 13×12). Separable-einsum net 2215 total; the final Einsum 'ntc,th,tw->nchw' writes free [1,10,30,30]; factors Rf/Cf [5,30] and all builder planes are welded to 30 (task077 free-operand refutation). Cropping → counted 4000B Pad. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
