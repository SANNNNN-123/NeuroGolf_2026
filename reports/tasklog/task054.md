# task054 — 264363fd

## Current live/source-owned exact

`memory=26877`, `params=238`, `points=14.792158`.

The graph is exact-preserve/source-owned.  It detects a reference star motif,
removes the reference motif, finds up to four seed stars inside large boxes,
draws horizontal/vertical guide lines through each seed across its containing
box, stamps the motif at each seed, and emits final one-hot with `Equal`.

## Dominant memory

- 3600B colour-index Conv output.
- Many 900B full-canvas uint8/bool edit planes:
  `label_u8`, `bg_mask`, `other/seeds`, `h_line`, `line_with_seeds`,
  `line_target`, `cleared_label`, `filled_label`, `output_label`.
- 1024B/960B int64 sparse fill/index tensors.

## 2026-06-29 sparse-edit-stream probe

Hypothesis: replace the full-canvas line cascade

`h_line_u8 -> line_with_seeds_u8 -> line_target_b -> output_label`

with sparse row/column edits applied directly to `filled_label`.

Graph-surgery candidate:

- Gather current rows from `filled_label` at `h_rows_4`.
- Use `h_updates_4_30 > h_seed_rows` to build sparse horizontal line updates.
- `ScatterND` those rows into `filled_label`.
- Gather current vertical columns and scatter vertical line updates.

Result:

- Best attempted candidate: `memory=26397`, `params=238`, but failed stored
  `244/266` (`22` arc-gen failures).  The raw saving was only `480B`.
- Variant trying to mimic inactive vertical overwrite by gathering from original
  `filled_label` failed much worse (`137/266`).

Reason:

The incumbent relies on `ScatterElements` duplicate/overwrite semantics for
inactive vertical slots.  Inactive vertical updates can intentionally erase a
horizontal line candidate before the final `line_with > seeds` comparison.
Applying sparse label edits directly loses this subtle mask-space behavior.

Conclusion:

The broad mechanism is still plausible only when sparse edits are monotonic
overwrites.  It is not safe when the intermediate mask uses inactive duplicate
scatter writes as part of the logic.  For task054, a successful rewrite must
either preserve the exact mask-space overwrite semantics or avoid generating
duplicate inactive vertical indices at the source.

## S8 (2026-07-02) — sparse-edit chain + free-input profiles (+0.234) ADOPTED
15×900B mask planes + 1920B CumSum stream → 4-plane sparse-edit chain: seeds via free-input
einsum row profiles ('bchw,c->h' + col-weighted) with motif 3×3 zone masked (seeds Chebyshev≥4
from centre); ScatterND ref-wipe → ring → row → col. KEY: line cells written as 255 with
reduction='max' → duplicate scatter writes = idempotent union (kills the 2026-06-29 overwrite
failure mode); final Equal matches 255-adjusted colour table. 20335+336 vs 25885+238 → +0.234.
Fresh 2500+5000+1500 div 0. TRAP: train[2] = fixed 3-box validation example outside the random
generator's 2-box path — box-agnostic segment spans required.

## S8 (2026-07-02, late) — select_last_index idiom ×4 (+0.014) ADOPTED, div 0

## S10 (2026-07-03) — scout re-confirm: FLOOR (label-read floor + irreducible edit chain)
Counted 20091 = labf 3600 (per-cell colour read, structural floor, measured 7 ways) +
fidx i64 1024 (ScatterND requires int64) + 4× uint8 edit boards 3600 (S8 golf result;
stages consumed sequentially, reduction='max' union semantics block fusion) + CumSum
segment planes. Seeds already free-input einsum profiles. Only sub-300B micro-levers left.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — output preserves the arbitrary input (per-cell read floor 3600B) and the line overlays are data-dependent 2-D interval fills = mechanism-15's own ~3000B band floor; incumbent's 4x900B edit boards land at the same cost. S8 FLOOR stands.
