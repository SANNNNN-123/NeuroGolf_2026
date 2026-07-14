# task092 — 40853293

**Rule:** A (10/20/30)x(10/20/30) grid holds up to 5 axis-aligned "sticks", each a DISTINCT
colour, drawn in the INPUT as their TWO endpoint pixels only. A stick is HORIZONTAL (both
endpoints in the SAME row, >=2 cols apart) or VERTICAL (both in the SAME column, >=2 rows
apart). The OUTPUT fills the whole segment between (and incl.) the two endpoints. Horizontal
sticks occupy distinct rows; vertical sticks distinct cols; a horizontal and vertical stick
may CROSS — at a crossing the VERTICAL stick's colour wins (generator draws all horizontal
sticks first, then all vertical, so the column over-writes). Verified 0/500 fresh:
rowfill_k = rowPrefixOR ∧ rowSuffixOR; colfill_k = colPrefixOR ∧ colSuffixOR; col wins.
**Current live/source-owned exact:** 16.145335 pts, mem 6825, params 182.
The old 14.866 custom draft below is superseded as an implementation, but its
semantic analysis remains useful.
**Target tier:** B/A — separable closed-form interval fill via 1-D row/col occupancy profiles
+ triangular prefix/suffix-OR; the per-cell colour-index plane is the floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior draft: per-ch tri-MatMul on 1-D profiles, fp32 ingrid ReduceMax + uint8 Where chain | B | 32350 | 1830 | 14.56 | — | stored baseline |
| 1 | drop fp32 [1,1,30,30] in-grid ReduceMax; recover H/W via ReduceMax(rowhas, ch-axis) | B | 29110 | 1829 | 14.66 | — | -3240B |
| 2 | fuse Sub+Mul+Add+Where sentinel into ONE Where(ingrid_bool, stickColor, 10) | B | 24670 | 1828 | 14.82 | — | -4440B |
| 3 | sentinel via 1-D offrow/offcol penalties in ONE variadic Sum (drops And bool plane) | B | 23950 | 1829 | 14.84 | — | -720B |
| 4 | **col prefix/suffix as RIGHT-multiply on [1,10,1,30] — kills the colhas Transpose** | B | 23350 | 1829 | **14.866** | 500/500 | best |
| 5 | public/live exact source-owned scatter compiler: endpoint moments + 1-channel scalar label scatter + final Equal | B+ | 6825 | 182 | **16.145** | stored 265/265 | current live/source baseline |
| 6 | remove vertical column-map roundtrip; route `v_score TopK` directly to final ScatterElements | probe | 6610 | 152 | 0 | stored 264/265 | rejected: inactive TopK slots duplicate active columns and overwrite active vertical strokes with base values |

## Best achieved
16.406031 @ mem 4940 params 459 — S11 signed-channel priority overlay (adopted 2026-07-03).

## S11 (2026-07-03) — ADOPTED: signed-channel priority overlay (NEW MECHANISM, playbook 15)
Replaced the scatter compiler (3× [30,30] canvas planes, 2700B) with ONE free final einsum
`bqr,qc,qv->bvrc` = RS[q,r]·CS[q,c]·W[q,v] writing the graph output directly.
KEY INSIGHT: grader decodes with `(out > 0.0)` per channel (src/harness.py:218), so the
"vertical wins at crossings" priority is LINEAR — signed weight matrix W[10,10]:
horizontal q → e_q − e_0; vertical q → 2·e_q − 𝟙; slot 0 = background band. No [30,30]
priority/label carrier at all. Fill is orientation-uniform: rowspan_q ⊗ colspan_q
(between min/max occupied index per colour), spans from fp16 endpoint-moment einsums.
Gates: bundled 265/265; fresh_verify 2000/2000 uncached, 0 divergence vs old incumbent
(bit-identical); +5000 fresh random; 4 hand-built crossing-heavy grids.
16.165 → 16.406 (+0.241, 6872B → 5399B). Backup: reports/retired_networks/task092_pre_s11_signedpriority.onnx.
FLOOR NOTE: remaining cost dominated by the two [10,30] band profiles + compare
transients (~3000B) — data-dependent 2-D interval fill floor; ≤1100B (18+) not reachable
with this construction. ORT einsum rejects uint8/int8 operands → bands must be fp16.
⭐TRANSFERABLE: (1) overlap/paint-order priority ⇒ signed channel matrix, deletes any
[30,30] scalar-label/priority carrier; (2) separable axis-aligned fill ⇒ one free
`sr,sc,sv->vrc` einsum. Cohort: 054/076/101/118/133/206/216/233/265/285/330/342/366/368/370
(prior FLOOR verdicts there predate this mechanism — label-plane floors are re-attackable).

The prior 14.866 custom source is obsolete for score, but still documents the
separable row/column semantics.  The current best graph instead avoids
10-channel working planes: it computes per-colour endpoints as tiny vectors,
uses 1-channel uint8 scatter planes for the scalar colour label, and routes
one-hot expansion into the final free `Equal(..., palette) -> output`.

## 2026-06-28 high-score frontier probe

This task is a useful frontier seed because the semantic rule is simple but it
still cannot currently enter the 20+ tier.

Dominant current tensors:

- `base_idx`: `[1,1,30,30]` uint8, 900B.
- `h_canvas`: `[1,1,30,30]` uint8, 900B.
- `scalar_color_u8`: `[1,1,30,30]` uint8, 900B.
- two expanded scatter index planes: 600B + 600B.

The live graph already delays 10-channel one-hot expansion until final output.
The remaining floor is the 1-channel scalar colour carrier.  Since even one
`30x30` uint8 carrier is 900B, this structure cannot reach 20+.

Rejected direct-output idea:

- Use the free `input` tensor as ScatterElements/ScatterND data so no zero
  full-canvas data tensor is needed.
- Scatter horizontal/vertical one-hot fills directly into `output`.
- Problem: crossings require removing the horizontal channel under the vertical
  stroke. A single scatter must either generate crossing negative updates or
  avoid duplicate inactive writes. ScatterND requires int64 `[N,4]` dynamic
  indices, which is large; ScatterElements direct routing creates inactive
  TopK duplicate columns. A concrete probe that skipped the `col_color_u8`
  column-map cut memory from 6825 to 6610 but failed stored example 26 because
  inactive slots overwrote active vertical colour 8 at column 0.

Current wall:

To break 20+, task092 needs a fundamentally different primitive: either a
single final-output op that can express interval-XOR/priority directly, or a
compact way to generate scatter indices without full/dynamic `[N,4]` int64
planes and without inactive duplicate writes. Simple graph surgery on the live
scatter compiler is unlikely to be enough.

## Irreducible-floor analysis
Memory now bound by FOUR [30,30] f16 planes (1800 each = 7200): the two colour MatMuls
(rowColor, colColor), the col-priority Where (stickColor), and the off-grid sentinel Sum (L2f).
Plus two fp32 ReduceMax occupancy reductions ([1,10,30,1]/[1,10,1,30] = 1200 each) — FORCED
because ORT ReduceMax rejects bool/uint8 and the input is fp32. The two colour MatMuls are
irreducible: a crossing cell is covered by a row-stick AND a col-stick, so a single linear
channel-contraction would SUM the two colours; the split (iscol/isrow weights) + a Where is
required to realise the "col wins" priority. ~14.9 is the practical floor for this two-plane
colour-priority fill.

## OPEN ANGLES (re-attack backlog)
- Collapse the col-priority Where: encode col colours at magnitude 10·c and Max() with the
  row plane (col always wins), then decode c=combined/10 for col cells — but decode needs a
  Where anyway, so net wash unless the decode folds into the final Equal arange (untried).
- Eliminate one fp32 ReduceMax (2400B total): both row- and col-occupancy come from the same
  input; no single reduction yields both, but a per-channel batched MatVec contracting the
  free fp32 input directly (guide's MatMul(input,vec) lever) might dodge one fp32 transient.
- Merge the two colour MatMuls into one batched [2,30,30] MatMul + Slice (fewer nodes, same
  bytes) — no mem win but cleaner.

## INSIGHT (transferable)
⭐ The axis-aligned analog of task037's diagonal endpoint-fill collapses to 1-D row/col
occupancy profiles (ReduceMax over the col/row axis → [1,10,30,1]/[1,10,1,30]), so the fill
is a triangular prefix∧suffix-OR on tiny per-channel VECTORS (params, not 30×30 planes), and
the colour is recovered by a [channel]×[30] MatMul contraction — never a [1,10,30,30] plane.
⭐ Do col-direction prefix/suffix-OR as a RIGHT-multiply (colhas[1,10,1,30] @ Tri[30,30])
instead of transposing to [.,30,1] and left-multiplying — saves the whole Transpose plane.
⭐ Off-grid sentinel without an in-grid 30×30 plane: add 1-D penalties offrow[r]=10·(r>=H),
offcol[c]=10·(c>=W) to the colour plane in ONE variadic Sum (broadcast) — off-grid cells go
>=10 so Equal(L, arange[0..9]) matches nothing (all-zero target), in-grid background stays 0
(ch0=1). Recover H/W as ReduceMax of per-channel occupancy over the CHANNEL axis (bg ch0=1
fills every in-grid cell), no separate frame ReduceMax over the full input.

## 2026-06-30 (S2) — int64→int32 gather-index micro-golf (LANDED)

Output-preserving safe-golf. `v_color_i64_flat = Cast(v_color_f32_flat, to=int64)`
feeds only two GatherElements as the index (top_u8 / bottom_u8). Retargeted the Cast
to int32 (one-attr change, no added plane). GatherElements accepts int32 indices;
values are colour ids 0–9 (no overflow).

- Before: `memory=6825`, `params=182`, `points=16.145335`.
- After:  `memory=6805`, `params=182`, `points=16.148193` (−20B, +0.0029).
- Gate: bundled 265/265 identical (old vs new, 0 divergence) + **fresh 1500/1500
  pass**. Bit-identical by construction (dtype-only on bounded index).
