# task264 - a8c38be5

## 2026-07-01 deep dive - current live/source frontier

**Status:** verified no-adopt but useful failure. The old `15.04 pts @ mem 20200,
params 944` matched-filter/template note was a real semantic direction against an
older baseline, but it is not the current frontier. Current source and live ONNX
are byte-identical exact-source reconstructions of a smaller hash/TopK template
graph:

- `reports/scripts/measure_task.py 264`: `265/265`, `memory=5348`,
  `params=706`, `points=16.291525510418335`.
- `networks/task264.onnx` and `src.custom.task264.build(...)` SHA256:
  `cb1d2440f85289e4a2c934fec4dd51fa21d537edc1069e676bc5395be9a4d7d6`.
- Fresh incumbent gate: `src.genverify.fresh_pass(264, 500)` -> `(500, 500)`.

### Human-readable rule

Input is a 14..16 by 14..16 grid padded to the 30x30 one-hot scorer shape. It
contains nine non-overlapping 3x3 sprites on black. Each sprite is first filled
with gray `5`; then fixed per-index glyph cells are overwritten with that
sprite's non-gray color. Colors can repeat between sprites. Index 4, the center
sprite, has an empty glyph, so its color is irrelevant and the output center
3x3 block stays gray.

Output is always a 9x9 chart in the top-left of the scorer canvas. It is a fixed
3x3 arrangement of 3x3 glyph slots: slot `idx` is stamped with glyph `idx` in the
color recovered from the matching input sprite, and every non-glyph cell is gray.
All cells outside the 9x9 output are false in every color channel.

Stored examples verify the visible rule:

- train/test/stored arc-gen count: `265`.
- strict Python oracle that scans for exactly one matching non-center glyph,
  recovers one non-gray color per glyph, and stamps the fixed 9x9 chart:
  `stored_oracle 265 / 265`.
- smaller fresh oracle sanity: `fresh_oracle 100 / 100`.

Confidence: **verified** for the generator family and stored examples.

### Current ONNX mechanism

The current graph does not use the older eight-channel matched-filter stack. It:

1. Converts the top-left 16x16 region from one-hot color to scalar labels with
   one `Einsum`.
2. Computes `is_gray = label16 == 5`, casts it to fp16, and applies one 3x3 fp16
   hash convolution over gray/not-gray windows.
3. Flattens 14x14 hash values, casts to int32, gathers a 69-entry fp16
   `rank_table`, then `TopK(8)` finds the eight ranked glyph windows.
4. Converts each selected top-left to one fixed anchor cell per glyph and
   `GatherND`s the scalar color from `label16`.
5. Builds a 9-color palette of eight recovered colors plus gray, gathers a fixed
   9x9 `slot_map`, compares it to color channels, and pads that 9x9 boolean
   tensor directly into the free 30x30 boolean output.

### Cost anatomy

| component | charged bytes / params | why it exists | current status |
|---|---:|---|---|
| `label16` fp32 `[1,1,16,16]` | 1024 B | scalar color carrier used both for gray detection and anchor color reads | dominant full-plane cost; 16 is needed because top-left can be row/col 13 |
| `eq9` bool `[1,10,9,9]` | 810 B | final one-hot 9x9 chart before Pad-to-output | cheaper than padding a uint8 30x30 label grid then Equal |
| `hash_idx` int32 `[1,196]` | 784 B | integer indices into `rank_table` after the gray hash conv | TopK/rank route cost |
| `gray_h` fp16 `[1,1,16,16]` | 512 B | ORT Conv needs numeric gray plane | required by current hash conv |
| `hash14`, `hash_flat`, `rank_flat` fp16 `[196]` each | 392 B each | one-channel window hash and rank scores over 14x14 candidate top-lefts | far smaller than prior 8-channel matched filters |
| `is_gray` bool `[1,1,16,16]` | 256 B | exact gray mask before Conv | could only disappear if color read and gray hash are both redesigned |
| `color_indices` int64 `[8,4]` | 256 B | dynamic GatherND indices for the eight color anchor reads | small but index dtype is expensive |
| small TopK/anchor/palette tensors | 405 B total | row/col arithmetic, eight colors, 9x9 uint8 label grid | mostly irreducible plumbing |
| selector initializer | 480 params | crops 30x30 directly to 16x16 inside the color-label `Einsum` | trades params for avoiding a 10-channel or 30x30 activation |
| `slot_map` initializer | 81 params | fixed 9x9 chart template | semantically required unless replaced by a larger dynamic output builder |
| `rank_table` initializer | 69 params | maps hash values to TopK ranks | tested for compaction; no smaller safe hash found in bounded search |
| remaining initializers | 76 params | color weights, hash kernel/bias, anchors, pads, channel values | small fixed constants |

Dominant cost is the 16x16 scalar label carrier plus the required gray/hash
working set. Current tier: **A/B boundary**. It avoids full 30x30 intermediates
except the free output, but it still carries one 16x16 scalar label plane because
the sprite position is unknown and the recovered color must be read after
detection.

### Prior notes challenged

- **Still valid:** The old semantic insight is right: this is recover-eight-color
  scalars plus stamp-a-fixed-template, not a full output label-map task.
- **Contradicted as current status:** The tasklog's previous "Current / Best
  achieved" value `15.04 pts @ mem 20200, params 944` is stale. Current manifest,
  source, and live ONNX are `16.2915 pts @ mem 5348, params 706`.
- **Superseded:** The old eight matched-filter detectors and color-sum readout
  are no longer an adoption candidate. The live graph compresses the same idea
  into one gray hash Conv, `rank_table`, `TopK`, and anchor reads.
- **Unverified from old note:** The old "irreducible floor" around three
  8-channel 14x14 fp16 planes is not a floor for the current graph; those planes
  have already been removed.

### Mechanism hypotheses tested

1. **Template/component matching via matched-filter glyph detect.**
   - Expected payoff: validate or kill the NEAR_18 claim of `+1.84` from
     matched-filter glyph detection plus fixed template stamping.
   - Proof test: compare against current manifest/source/live graph and run
     stored/fresh verification on the current source-owned graph.
   - Evidence: current graph passes stored `265/265` and fresh `500/500`, scores
     `16.2915`, and has `mem+params=6054`; the logged matched-filter graph scored
     `15.04` with `mem+params=21144`.
   - Kill condition met for adoption: matched-filter/template is semantically
     valid but strictly dominated by the already-current hash/TopK template graph.

2. **Hash-table compaction for the gray-window rank detector.**
   - Expected payoff: reduce the 69-param `rank_table` and possibly a few points
     of total cost; no memory payoff.
   - Proof test: decode current target gray-complement patterns, check current
     hash range over all 512 binary 3x3 windows, collect reachable fresh patterns,
     then run a bounded vectorized search for a smaller collision-safe linear hash.
   - Evidence:
     - current all-512 hash range: `0..68`, so `rank_table_len_needed=69`;
     - target hashes: `{0:31, 1:56, 2:57, 3:28, 5:54, 6:0, 7:10, 8:15}`;
     - 300 fresh instances exposed 170 reachable 3x3 gray patterns and all target
       complements appeared;
     - bounded vectorized random search over weight ranges `[-8,8]`,
       `[-16,16]`, `[-24,24]`, `[-32,32]` found no span below 68.
   - Kill condition met for this session: no smaller verified source-owned hash;
     even a best-case win is params-only and too small to justify unbounded search.

### Next exact experiment

If task264 is revisited, the only plausible remaining high-leverage direction is
to remove `label16` entirely by detecting from gray-channel input and reading
colors directly from the original one-hot input at the eight anchors. The proof
test must account for dynamic `[8,10]` anchor/channel gathers: if the dynamic
indices exceed roughly 1 KB or require a 10-channel crop, it loses immediately
against the current 1024 B `label16` carrier and 256 B `color_indices` route.

No reusable mechanism beyond the already-known "fixed recolored glyphs -> recover
scalars + stamp template" was found in this pass.


## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.113, policy-gated)

Clean adoption (candidate ≤ incumbent on every gate). Same hash/rank/TopK glyph-chart
template as the incumbent — the whole 20-node graph is structurally identical.

**Mechanism diff (op census, retired vs new):** only change is `Conv` → `QLinearConv`.
The fixed 3×3 gray-window hash convolution is int8-quantized; the downstream
`Einsum` colour-label read, 69-entry `rank_table` gather, `TopK(8)` glyph-window
ranking, `GatherND` anchor colour reads, 9×9 `slot_map` compare and `Pad`-to-free-output
are unchanged. Initializer footprint essentially identical (3122 → 3120 B). The
−646 cost comes from the int8 conv output/working plane replacing the fp16 hash plane;
the hash values still feed `TopK` through an fp16 rank score, so ranking stays exact.

**Cost:** mem 5348→4700, params 706→708, pts 16.2915→16.4044 (**+0.113**, cost 6054→5408 −646).

**Gate evidence:** bundled 265/265 fail=0 (both nets). Fresh 2×2000 uncached: candidate
0 fails, incumbent 0 fails, 0 divergence. TopK audit: 1 TopK, data-input `safe_name_23`
= **FLOAT16** (grader-safe; not uint8).

**Backup + provenance:** incumbent → `reports/retired_networks/task264_pre_s10.onnx`;
candidate source `public_candidates/bobmyers7186/task264.onnx` → `networks/task264.onnx`;
source regenerated via live_to_exact_source --write-src, src↔live reconciled fail=0.

⭐ TRANSFERABLE: int8 `QLinearConv` on a **fixed-kernel detection/hash Conv whose output
only needs to preserve ORDER (feeds TopK/ArgMax through an fp16 score), not exact values**
is bit-safe here and shaves the conv working plane. Propagate to hash/rank/TopK template
nets — selection criterion: a fixed-weight Conv whose output flows only into a TopK or
ArgMax rank (not into arithmetic that must stay float-exact). Sibling hit this session = task365.
