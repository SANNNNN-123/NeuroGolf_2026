# task056 — 27a28665

**Rule:** Input is a 3×3 grid containing one shape (some colour) drawn at one of 4 fixed
pixel patterns indexed idx∈{1,2,3,6}; OUTPUT is a 1×1 grid whose single cell value == idx.
Colour-agnostic, size always 3×3 → pure 4-way shape classification → scalar.
**Current:** 19.64 pts, ext:kojimar6275, mem 136, params 77
**Target tier:** A/B — classification → single scalar fingerprint → one-hot, routed to free output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | [10,1,3,3] signed-template Conv + Equal==5 | B | 112 | 110 | 19.60 | - | worse (params 90-elem conv) |
| 2 | single [1,1,3,3] positional Conv → scalar + bank Equal | B | 94 | 38 | 20.12 | - | +0.48 |
| 3 | run Conv on bg directly, fold Sw=11 offset into bank (drop binary plane) | B | 60 | 37 | 20.43 | 200/200 | ADOPT-candidate |

## Best achieved
20.43 @ mem 60 params 37 — adopted? N (build agent does not adopt). Beats prior 19.64? Y (+0.79).

## Irreducible-floor analysis
Dominant intermediate = the bg slice [1,1,3,3] fp32 = 36B; Slice preserves the fp32 input
dtype so it cannot narrow, and it is the Conv input. Remaining: score [1,1,1,1] fp32 4B,
onehot [1,10,1,1] bool 10B + uint8 cast 10B (Pad rejects bool, so the uint8 cast is required).
Total 60B is near floor for a fp32-input single-cell-region classification.

## OPEN ANGLES (re-attack backlog)
- Slice only the 6 weight-nonzero cells to a smaller region (non-contiguous → more Slice ops,
  likely net loss vs the 36B floor). Not worth it; gain <2pts of mem.
- Output as bool instead of uint8 to drop the cast — blocked: ORT Pad rejects bool.

## INSIGHT (transferable)
⭐ A k-way FIXED-SHAPE classifier collapses to ONE [1,1,k,k] positional-weight Conv producing
a single scalar fingerprint (pick small integer weights giving k DISTINCT sums), then a
[1,10,1,1] "bank" const holding each class's fingerprint at its own OUTPUT channel makes
`Equal(score, bank)` the 10-channel one-hot DIRECTLY — no ArgMax/Gather/output_bank, no per-class
template stack. Run the Conv on the bg (channel-0) slice directly and fold the weight-sum offset
(Sw − fp) into the bank constants to delete the 1−bg binary plane entirely (drops a Cast + Sub +
fp16 plane). Output is the value of idx at one cell, but the encoding never depends on idx being
small/contiguous.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** 3×3 grid contains one shape drawn as one of 4 fixed pixel patterns idx∈{1,2,3,6}; output = 1×1 grid whose single value == idx. Colour-agnostic 4-way shape classification → scalar.

**Cost (grader mem 34, params 0):** ops And×3/Slice×2/Greater×2/Cast×2/Equal/Not/Concat/Pad, ZERO initializers. Counted intermediates: `active_channels` [1,6,1,1] fp16 12B, `active_bool` [1,6,1,1] bool 6B, two scalar `b0_f`/`b2_f` [1,1,1,1] fp32 4B each, plus handful of [1,1,1,1] bool bits. Output [1,10,30,30] fp16 18000B is FREE.

**Blocker class:** already-at-floor. Everything counted is a channel-vector or scalar (≤12B); a 4-way fingerprint classifier has no spatial working plane. The rich log's build-agent reached mem 60; the LANDED net is even leaner at 34B.

**Lever:** no lever visible. Near the mem-0/scalar floor; no plane to narrow.
