# task155 — FLOOR (S4, 2026-06-30)

**Rule:** vertical flip (`flipud`) of a top-left-aligned variable-size square grid.
Output row i = input row (G-1-i), bottom rows are background.

**Live incumbent (already optimal):** mem 128, params 30, **19.937 pts**
(`method: ext:franksunp7166_65`). Graph: `ReduceL2(input)`→ scalar `G`
(= sqrt of one-hot cell count G²) → `Cast`→int32 → `Sub(G, row_ids[1..30])`
→ src_rows `[G-1,…,G-30]` → `Gather(input, src_rows, axis=2)` into the FREE output.
Negative-index wrap fills the bottom rows with zeros. Verified ok, fail=0 on all 266 bundled.

## Why this is the practical floor
- Output must be `[1,10,30,30]` ⇒ the `Gather` needs a **30-element** index. ONNX Gather
  indices accept only int32/int64 ⇒ **int32 [30] = 120B is the hard floor** for any
  Gather-based row permutation. (Same conclusion the sibling task150 — a *column*
  permutation — reached: it landed at mem 136 / params 31 / 19.88, **worse** than 155.)
- The single `[30]` `Sub` output (120B) + two scalars (8B) = 128 mem; the `[1..30]` init
  = 30 params. Total **158**. The min_stat ceiling (20.84 ≡ mem+params≈64) is therefore
  **unreachable** — the index alone exceeds 64B. The "+0.90 headroom" is the inflated
  generic FIXED_TRANSFORM heuristic (cf. playbook §5 caveat 2), not an achievable target.

## Only param-saving idea — BLOCKED by an ORT bug
Replacing the 30-element init with a `Range(G-1, G-31, -1)` would cut params 30→2 for
+0.13 pts. **But dynamic-input `Range`→`Gather` produces 0 rows in onnxruntime 1.26.0**
(the data-dependent Range output dim is mis-inferred as 0; only *constant*-input Range
constant-folds correctly). Verified: with constant start/limit it matches; with
input-derived start/limit the Gather output is `{1,10,0,30}` → all-zero → fails. The
official grader uses the same ORT, so this rewrite would score **0**. Not landable.
The alternative (Range-generated arange as an intermediate) adds a second `[30]` tensor
(+120B) — a net loss. **Held: no robust improvement exists; incumbent stays.**

## S9 (2026-07-03) — Range(H-1,H-31,-1) swap (+0.107) ADOPTED; stale ORT bug REFUTED
Old claim "Range→Gather = 0 rows on ORT 1.26.0" is FALSE on current ORT 1.26.0:
teacher passes 266 stored + 2500 fresh bit-identical + 0/600 across every G=1..30.
30-elem int32 row_ids init (30p) → Range (2p); mem 128→140, total 158→142.
Backup task155_pre_s9.onnx. FOLLOW-UP: task150 (column-permute sibling, 136/31) is a
candidate for the same swap.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 8. Same structure as task150 (row flip, axis=2): 120B Gather index is a floor, must match the 30-wide free output. Cropping forces a 9600B counted re-embed. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
