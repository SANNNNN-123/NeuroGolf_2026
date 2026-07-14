# task216 — 8efcae92 ("mostpixels")

**Rule:** 20x20 grid with 3-4 non-overlapping solid BLUE(1) rectangles ("boxes"),
each separated by a >=1-cell gap **in at least one axis** (common.overlaps spacing=1
=> two boxes may touch in one axis but never overlap in BOTH). Each box has RED(2)
pixels scattered inside; red counts are DISTINCT across boxes. The box with the MOST
red pixels (generator forces box 0 = max) is the winner. Output = exactly that
winning box (its blue rectangle + reds) placed at the top-left (0,0); cells outside
the box's own size are all-channels-zero (output grid IS box-sized).
**Current (public):** 13.56 pts, mem 92612, params 101 (CumSum-scan segmentation +
ArgMax + crop/translate). Generalizes 300/300 fresh — a real, at-floor score.
**Target tier:** detection (segmentation + global ArgMax + data-dependent crop &
translate). NOT B/A/S: output is a variable-size data-dependent crop of one of
several connected components selected by a non-local global maximum.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | corner-label (T,L) via triangular ReduceMax; per-corner red histogram via batched double-MatMul; ArgMax winner; double-MatMul shift to origin | det | 176k | 361 | 12.92 | 266/266 stored | correct but heavy (W^3 histogram + W^3 running-max) |
| 2 | running-max -> Hillis-Steele doubling scan (O(W^2)); drop histogram, count reds via 2-D prefix-sum integral image over [T,L,B,R] (needs R,B suffix-min scans) | det | 138k | 404 | 13.16 | 266/266, 300/300 fresh | correct, leaner |
| 3 | fp16 everywhere (scans, run masks, prefix-sum cast post-CumSum, gather idx int32); fp16 neighbour shifts | det | 125k | 404 | 13.26 | 266/266 | dtype squeeze |
| 4 | doubling shift via fixed {0,1} MatMul (no Pad/Slice), col-orientation + transpose for row scans (10 shared shift mats), factored integral-image index math | det | **106k** | 4219 | **13.39** | **200/200 fresh** | best; still < public 13.56 |

## Best achieved
**13.39 @ mem 105972 params 4219 — 266/266 stored, 200/200 (& 300/300) fresh.**
Adopted? **N.** Beats prior 13.56? **NO** — 0.17 BELOW. Does NOT beat by +0.3;
in fact a slight regression. Verdict: MARGINAL-LOSS, do not adopt.

## Irreducible-floor analysis
Dominant memory = ~91 fp16 [1,1,20,20] (800 B) intermediates. Breakdown of the
irreducible work:
- **4 corner maps (T,L,B,R)**: each needs a prefix/suffix MAX scan over the run-
  start/run-end index encoding. Hillis-Steele doubling = 5 steps (W=20 => need
  reach 18 => log2 => 5 steps unavoidable), 2 tensors/step (shift + Max) = 40
  fp16 [1,1,20,20] = **~32 kB. Byte-invariant**: stacking maps onto channels makes
  each tensor proportionally bigger (no win); uint8 would halve it but ORT
  Max/Min/MatMul all reject uint8; CumSum rejects fp16. fp16 is the floor.
- **integral-image red count** (PS 2-D CumSum f32 + 4 Gathers + index math) ~20 kB;
  needs R,B (=> the 2 extra scans). The rectangle [T..B]x[L..R] provably contains
  only the winning box (no two boxes overlap in both axes), so the integral image
  is exact — but it requires all four corners.
- run-start/end detection (~16 tensors) + double-MatMul shift-to-origin (~12) +
  uint8 label Pad + final Equal (output is FREE) ~25 kB.
The winner-SELECTION (which box has the most reds) is the genuine cost driver: it
is a non-local global ArgMax over per-box red totals, which forces full per-cell
segmentation (corner labels) — exactly what the public CumSum net also pays for.

## OPEN ANGLES (re-attack backlog)
- **Drop R,B (2 scans, ~16 kB)** by counting box reds with only L,T. Every cheap
  formulation tried still needs the box right/bottom edge (row-run total needs R;
  box total needs B): the integral image over [T,L,B,R] is the minimal exact form.
  A genuinely L,T-only group-by-corner count is the W^3 histogram (worse). OPEN: is
  there an O(W^2) group-sum keyed by (T,L) using two reset-CumSums that reuse L,T?
- **uint8 scan** blocked: ORT Max/Min reject uint8 (verified). A custom non-Max
  monotone combine (e.g. Add of disjoint magnitude bands) might pack the running
  max into uint8 — untried, fiddly, ~16 kB upside if it works.
- Tier S/A/B all blocked: output is a variable-size crop of a globally-selected
  connected component => no fixed Conv/permute/separable form exists.

## INSIGHT (transferable)
⭐ **2D-separated rectangles get a unique per-box label from (T,L) = (top of its
column-run, left of its row-run), recoverable by prefix/suffix MAX scans; and the
bbox [T..B]x[L..R] then contains ONLY that box** (because spacing>=1 forbids overlap
in both axes), so per-box reductions (here red count) reduce to a 2-D prefix-sum
integral image with 4 Gathers — NO connected-component flood-fill needed. Beware the
spacing=1 loophole: two boxes CAN share a column range (row-gap separated), so a
naive "reds in bounding rectangle" using a buggy suffix-MAX over-counts; the run-end
edge must be suffix-MIN (nearest end >= position), validated 800/800.
⭐ **Hillis-Steele doubling prefix-max via a fixed {0,1} MatMul shift** (encode values
so 0-fill is order-safe; col orientation + Transpose for row axis to share matrices)
replaces Pad+Slice+Max (3 tensors/step -> 2) — but the byte cost is invariant under
channel-stacking and fp16 is the dtype floor (Max rejects uint8), so this whole
"segment + global-argmax + crop" class floors near the memorizer/CumSum level (~13.4)
for everyone. This is the non-local detection wall the sweep guidance warns about:
the public net is already at it.

## S9 (2026-07-03) — free-operand einsum red-count (+0.053) ADOPTED
NOTE: this tasklog was STALE (described a 92k segmentation net); live incumbent was
already 9135 via QLinearConv-corner + integral-free design. S9 golf: red plane (400B) +
2×MatMulInteger contractions (320B i32 + 64B cross-terms) replaced by ONE 4-operand
Einsum 'nkrc,k,br,bc->b' reusing counted c12_f32 as free operand → per-box red counts
(4,) direct. mem 9048→8584, params 87→76. Bit-identical: 2500+600 uncached fresh 0/0/0.
FLOORS: c12_f32 3200 = entry floor (both channels needed at 20×20; alternatives ≥cost),
corner-finding 1200 + run-scan 1120 near-minimal u8. Backup task216_pre_s9.onnx.

## S11 (2026-07-03) — mech-15/pointer scout: KILL — output = variable-size crop of globally-ArgMax-selected most-red box; cost = winner-selection (3200B entry + corner/run-scan). No separable fills, no carrier. Floor stands.
