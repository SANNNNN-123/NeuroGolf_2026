# task021 — 1190e5a7

## 2026-06-29 output-mask screen

Current source score: 17.935241 @ mem 1072 params 98.

Rule: an input grid is divided by grid lines; output is a small rectangle whose
height/width are the counts of row/column cells, filled with the main colour.

The graph is close to the final-output frontier: it computes the main colour and
two scalar output dimensions, builds a full [1,1,30,30] bool `cell_in_out` mask
(900 B), and uses `Where(mask, main4, zero) -> output` so the 10-channel rectangle
is the free graph output.

`conv_fit.py 21` failed for k=1/3/5.  A smaller 7x7 mask route is not obviously
better: once the random main colour is applied, it either creates a counted
[1,10,7,7] one-hot plane or a 30x30 uint8 label before final `Equal`, roughly
matching or exceeding the current 900 B full mask.  No rewrite adopted.

## S9 (2026-07-03) — kojimar teacher REPAIRED then ADOPTED (+0.465)
Teacher drops the [1,1,30,30] bool mask (900B): computes the two dims via free-input
einsums, builds small H×W mask, Pads to 30. RAW teacher had height cap 6 (range_h6
[0..5]) but generator rows∈[2,7] → 8/761 fresh fails (bundled max H=6 hid it — classic
bundled-overfit). Repair: range extended to 7 (+78B mem, +1 param), pad fixed.
mem 1072→675, params 98→60. Gates: stored fail=0; fresh 2929 valid/10000 (oversize grids
rejected): inc 0 / repaired 0. Backup reports/retired_networks/task021_pre_s9.onnx.
⚠ raw base_submission/task021.onnx is private-LB-fragile — never adopt it directly.

## S10 (2026-07-03) — kojimar 7185.95 teacher ADOPTED (+0.819)
**Mechanism swap — drop the 2D mask + Pad, emit the rectangle by an outer-product Einsum → FREE output.**
OLD (S9-repaired teacher, 16 nodes): Slice/Squeeze read + 4 Einsum dims, Less/Less→And→Cast→And build a
small H×W bool cell mask, then `Pad` places the main colour into the [1,10,30,30] output (initializers
col0[30], range_h6[7], range_w7[7], pad6x7_to_30[8], slice specs, false_b). NEW (13 nodes, {Einsum:6,
Sub:2, Less:2, Cast:3}): 5 Einsums fully contract the FREE input against just two length-30 float vectors
`e0`[30] (=elementary basis e₀, taps the top-left cell) and reduce out the colour one-hot `bg` [1,10] and
the row/col extents (`bchw,h,w->bc`, `bchw,w->b`, `bchw,bc,w->b`, …); two Less+Cast turn the extents into
1-D length-30 row/col indicator vectors `rf`,`cf`; then ONE final Einsum `bc,r,s->bcrs` outer-products
(colour one-hot ⊗ rf ⊗ cf) to emit the [1,10,30,30] output DIRECTLY as the free graph node. **The 2-D H×W
bool mask and the Pad are both gone** — the only counted intermediates are the colour vector and the two
1-D indicators; the 30×30 canvas is never materialized as a working plane. Initializers collapse to
e0[30]=120B + rm1[30]=120B. mem 675→**264** (−411), params 60→60 (unchanged), pts 18.400→**19.219 (+0.819)**.
**Gates:** bundled fail=0; fresh arc-gen 2×2000 streams — generator emits inputs up to 36 wide so ~71% of
fresh instances are oversize (>30) and skipped by the one-hot harness → ~579 valid instances/run,
inc_fail=0, cand_fail=0, cand==inc on every valid instance; no TopK/uint8 offenders; NON-CACHED, orchestrator-
reverified. Backup reports/retired_networks/task021_pre_s10.onnx. Provenance public_candidates/kojimar7185_95/overrides/task021.onnx.
⭐ TRANSFERABLE: a solid-colour axis-aligned RECTANGLE (or any SEPARABLE box) output is a rank-1 outer
product colour⊗row_indicator⊗col_indicator — emit it with ONE Einsum `bc,r,s->bcrs` to the FREE graph output
and keep only the two 1-D length-30 indicator vectors; NEVER build the counted [1,1,30,30] bool mask + Pad.
Selection criterion: any net whose output is a solid-colour rect / separable box currently paying a counted
30×30 bool mask or a Pad-placed fill (grep nets for a [1,1,30,30] BOOL intermediate feeding Where/Pad).
Direct sibling of the signed-Einsum-routing lever (neurogolf-signed-einsum-routing) — this is that pattern
realised for the unsigned single-rect case; feed the separable-rect output scanner these criteria.
