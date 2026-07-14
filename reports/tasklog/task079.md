# task079 — 39a8645d

**Rule:** A 14x14 grid holds 2-3 monochrome 3x3 sprite TYPES (distinct shapes,
distinct colours). Each type is placed `num` non-overlapping copies, with the
per-type copy counts sampled DISTINCT from {1,2,3} and sorted descending, so
type 0 has the strictly-most copies. The 3x3 output is the shape of type 0 in
its colour. Dominant colour = argmax over colours of copies = total_pixels /
sprite_size, where sprite_size = max 3x3-window pixel count (one aligned copy).
"Most pixels" is WRONG (a 6px shape x3 < an 8px shape... must divide by size).
**Current:** 15.03 pts (Conv→ArgMax crop net), mem 21138, params 245
**Target tier:** B/detection — output is a data-dependent 3x3 crop selected by a
per-colour copy-count argmax; two fp32 per-channel spatial reductions form a
floor (~15.8 ceiling per project note).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | "most pixels" colour + block-argmax crop | B | 5824 | 54 | — | — | WRONG rule (3 ties fail) |
| 2 | copies=cnt/blkmax, depthwise 3x3 conv, 10ch | B | 16220 | 136 | 15.30 | — | correct but <+0.3 |
| 3 | drop bg ch0: slice+conv on 9 colour ch, fp32 | B | 14844 | 121 | 15.39 | 500/500 | ADOPT-CANDIDATE |

## Best achieved
15.39 @ mem 14844 params 121 — adopted? N (build-only). Beats prior 15.03 by
+0.36 (>+0.3). fresh 500/500 isolated.

## Irreducible-floor analysis
Two dominant planes: (a) Slice of input to the 14x14 active region across the 9
colour channels = [1,9,14,14] fp32 = 7056B (entry plane; Slice inherits fp32, the
conv needs all 9 colour channels at full 14x14 so it can't be cropped further);
(b) the depthwise 3x3 sum-conv [1,9,12,12] fp32 = 5184B (ORT computes Conv in
fp32 and emits a PrecisionFreeCast fp32 plane even for fp16 weights, verified —
so fp16 conv is NOT cheaper). cnt is free (ReduceSum). Everything downstream is
tiny (1-channel 12x12 / 14x14 / 3x3 gathers). The remaining ~1.9KB is the
Mblk/Reshape/Mflat argmax-position chain + the Mk crop plane.

## OPEN ANGLES (re-attack backlog)
- Replace the depthwise 3x3 conv-then-ReduceMax with a separable row-3 then
  col-3 conv to see if the intermediate is smaller (likely same, both need the
  9ch×14×14 slice).
- Find a copy-COUNT proxy that avoids per-channel sprite_size entirely (e.g. a
  scalar that monotonically tracks copies without dividing) — would drop the conv.
- Collapse Mblk argmax-position chain by gathering channel k0s from a pre-flattened
  blk [1,9,144] once (saves one ~576B reshape).

## INSIGHT (transferable)
⭐ "output the most-COMMON of K identical-shape sprite types" is NOT most-pixels:
copy-count = total_pixels / sprite_size, and sprite_size = MAX 3x3-window pixel
count (one aligned non-overlapping copy). A depthwise 3x3 sum-conv + per-channel
ReduceMax gives every colour's sprite_size in one pass; argmax(cnt/size) picks the
dominant colour. The data-dependent 3x3 crop falls out of an ArgMax over the
dominant colour's flattened block-count plane (ties between identical copies are
fine — flat-argmax picks the first, all copies share the shape). Dropping bg ch0
from the slice+conv (9 not 10 channels) is a free ~10% mem cut. Re-confirmed:
fp16 Conv is NOT cheaper — ORT always emits an fp32 PrecisionFreeCast conv plane.

## S16 (2026-07-06) — public bit-identical golf (franksunp) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
