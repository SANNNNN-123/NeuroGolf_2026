# task355 — de1cd16c

**Rule:** The grid is a 2×2 tiling of solid-colour rectangular blocks (block size 5..10
per side, fully filling a ≤20×20 canvas, anchored top-left). The 4 block colours are
DISTINCT (`sample`). A unique `pcolor` (not any block colour) is scattered as `counts[idx]`
single specks over block idx, the counts a DISTINCT sample of range(6). The 1×1 output =
`mostest`, the colour of the block that received the MOST specks (unique max count).
(The hand-written `validate()` test uses 2×3 blocks with repeated colours, but `generate()`
— what generalization is scored against — is always 2×2 with distinct colours.)

**Current best (this session):** 17.063 pts, mem 2772, params 26 — area−count identity, no speck plane.
**Target tier:** A — closed-form per-channel scalar count via the `area = cnt + specks` partition.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel bbox (first/last) + fp16 MatMul | A | 17588 | 44 | 15.22 | — | too many [1,10,*] planes |
| 2 | occupancy-IS-the-band (drop first/last/fill), fp16 | A | 11388 | 72 | 15.65 | — | +0.03, big simplification |
| 3 | same, fp32 MatMul (no speck fp16 copy) | A | 10168 | 72 | 15.77 | 200/200 | prior probe best; +0.14 |
| 4 | Gather index-shape [1] (no Reshape dup) + uint8 Pad out + consolidated inits | A | 9840 | 25 | 15.803 | 500/500 | prior stored best (+0.18) — STILL carries the 3600B speck plane |
| 5 | fp16 cast of speck+occ downstream | A | 11020 | 36 | 15.69 | — | net-NEGATIVE: fp32 occ from ReduceMax stays, fp16 adds planes |
| 6 | **area−count identity (NO speck plane)** | A | **2772** | 26 | **17.063** | **500/500** | ⭐ **NEW BEST +1.26** — speck plane eliminated entirely, written to src/custom |

## Best achieved
**17.063 @ mem 2772 params 26 — src/custom/task355.py WRITTEN. Beats prior stored 15.803 by +1.26 (500/500 fresh).**

⭐ BREAKTHROUGH — the "irreducible 3600B speck Gather" wall in attempt #4's floor analysis was FALSE.
A speck OVERWRITES the block colour, so inside block k's rectangle every cell is either
block-colour-k or pcolor. Hence the EXACT partition `area_k = cnt_k + specks_k`, giving
`specks_k = area_k − cnt_k` with NO speck plane and NO [1,10,30,*] projection at all:
  - `cnt = ReduceSum(input,[2,3])` → [1,10,1,1] (per-channel pixel counts)
  - `area_k = rowcount_k × colcount_k`, where rowcount/colcount = ReduceSum of the binary
    occupancy bands `ReduceMax(input,[3])` / `ReduceMax(input,[2])`
  - `specks_k = area_k − cnt_k`; mask pcolor (smallest nonzero cnt) & absent channels → ArgMax
  - one-hot at (0,0), Cast uint8, Pad into the FREE output.
The ONLY mid-size intermediates are the two occupancy bands ([1,10,30,1]/[1,10,1,30] f32 =
1200B each); everything else is a [1,10,1,1] scalar. mem 2772 ≈ 2×1200 + ~370 misc.

Key fixes over attempt #3: (a) Gather(input, pcolor) with a **[1]-shaped index** yields [1,1,30,30] directly — the old Squeeze-to-scalar + Reshape created a DUPLICATE 3600B speck plane (7200B). (b) uint8 Pad to route the one-hot to cell (0,0) (bool Pad fails onnx.checker full_check, so Cast→uint8, declare output uint8; harness scores out>0). (c) consolidated arange(10) inits + dropped redundant `present` Where (absent channels have boxcnt=0 so ArgMax never picks them; only the pcolor channel must be zeroed since its bbox spans all specks).

Approach: cnt=ReduceSum(input,[2,3]); pcolor=argmin nonzero count; speck=Gather(input,pcolor)
[1,1,30,30]. Key simplification: a SOLID block fills every row/col in its range, so the 1-D
per-channel occupancy `ReduceMax(input,axis)` IS the bbox band directly — no first/last/fill
needed (verified 800/800). boxcnt[k] = rowband_k @ speck @ colband_k via two batched MatMuls
(broadcast 10 vs 1, never materialises the 10×30×30 product). Zero the pcolor & absent
channels, ArgMax → mostest. Output = And(row0, And(col0, onehot(mostest))) routed into the
FREE bool output (associative broadcast, no full label plane).

## Irreducible-floor analysis (updated)
Dominant intermediates are now just the two binary occupancy bands `ReduceMax(input,[3])` =
[1,10,30,1] and `ReduceMax(input,[2])` = [1,10,1,30], 1200B each (300 elems × fp32). Born f32
(ReduceMax rejects narrow dtypes) so fp16/uint8 cast only ADDS planes. The band is inherently
10ch × 30 — slicing to present channels (≤5) or the ≤20×20 active region trips the symbolic-dim
trap / costs a 10ch input slice (6000B+). Both bands are required (area = rows × cols). So
~2400B of bands + ~370B scalars/misc ≈ 2772 is the practical floor for this identity. Pushing
lower would need rowcount/colcount WITHOUT a per-row occupancy signal, with no known cheaper op.

The OLD "3600B speck-Gather wall" was a FALSE floor — it presumed you must spatially count the
marker, but the area−count algebra needs only scalar per-channel counts.

## INSIGHT (transferable)
⭐⭐ MARKER-OVERWRITES-FILL → AREA−COUNT, NO SPATIAL COUNT: when a sparse marker colour is
stamped ON TOP of solid coloured regions (overwriting the region colour) and you need the
per-region marker COUNT, do NOT gather/project the marker plane (3600B). The marker punches
holes in its host region, so for region k: `area_k = cnt_k + markers_k` EXACTLY (every cell in
the region is either region-colour-k or marker). Thus `markers_k = area_k − cnt_k`, all scalars:
`cnt_k = ReduceSum(input,[2,3])`, `area_k = (#rows occ)×(#cols occ)` from the two 1-D occupancy
bands. This turns an apparent "count sparse pixels per region" task (which looks like it needs a
full marker plane + per-region bbox projection) into a pure [1,10,1,1] scalar pipeline — the only
mid planes are the two occupancy bands. Collapsed 9840→2772 (+1.26). Generalises to ANY
"overlay-marker per-solid-region count / argmax / rank" where regions are solid & disjoint and
the marker is a distinct colour. Mask the marker channel itself (its own bbox spans all markers
→ huge spurious area−count) before the argmax.

## OPEN ANGLES (minor)
- The two occupancy bands (2400B) are the remaining cost. No cheaper per-channel-per-row
  occupancy op is known (ReduceMax born f32; channel/region slicing trips symbolic dims). 2772
  is effectively the floor.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2688/17.09; CONST-OUTPUT false—output=colour of most-speck block; needs per-block speck count (area-cnt), two occupancy bands ~2400B floor. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-1/2 opus agents; occupancy/max-semiring reductions or sub-400B u8 banks). Do not re-attempt without a new mechanism.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.000)
**Mechanism (op-census diff):** Scalar/arange const dedup+rename (dropped `thr`/`neg`/`zero`/`ax*`/`arange10_i` for `safe_name_*`). −7 params, mem flat.
**Old→new:** mem 2688→2688, params 27→20.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task355_pre_s10.onnx`; source `public_candidates/bobmyers7186/task355.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 2708 -> 2704 (+0.001); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].