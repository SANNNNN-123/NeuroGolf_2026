# task257 — a68b268e

**Rule:** 9x9 input split by a blue cross (row4/col4) into four 4x4 quadrants, each a FIXED colour: TL=7, TR=4, BL=8, BR=6. Output 4x4 overlays the quadrants; each cell takes the FIRST non-background quadrant in priority TL(7)>TR(4)>BL(8)>BR(6) (generator writes idx 3,2,1,0 so idx0=colour7 wins). Background cells in the 4x4 set channel-0.
**Current:** 18.02 pts, ext:kojimar6275 import, mem 1024, params 53 (fp16 Sub/Mul priority net)
**Target tier:** S/copy-mask — pure one-hot {0,1} overlay, no value arithmetic needed; harness scores (out>0) so dtype is free.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4 fp32 quadrant slices -> Cast bool -> And/Or/Not priority -> Cast uint8 -> Concat10 -> Pad | S | 720 | 52 | 18.35 | 200/200 | ADOPTED (+0.33) |

## Best achieved
18.35 @ mem 720 params 52 — beats prior 18.02 by +0.333 (clears +0.3). Fresh isolated 200/200.

## Irreducible-floor analysis
Two dominant items: (1) the four fp32 quadrant Slices [1,1,4,4] = 64B each = 256B — irreducible because Slice preserves the fp32 input dtype and the 4 quadrants live at distinct channel+location regions (channels 4,6,7,8 are non-contiguous; a single bounding-box slice over channels 4-8 = 405 elems = 1620B, far worse). Casting to bool happens AFTER the slice so the fp32 plane is still counted. (2) Concat [1,10,4,4] uint8 = 160B — the 10-channel output is mandatory and 6 channels are zero but Pad cannot interleave them into a smaller active group. Bool intermediates (Or/Not/And, ~12x16B) and 5 uint8 casts (80B) make up the rest.

## OPEN ANGLES (re-attack backlog)
- The 256B fp32-slice floor is the prize. No tried route beats 4x16-elem reads. A Gather that strides rows [0-3,5-8] x cols [0-3,5-8] could pull all four 4x4 blocks in ONE op, but each quadrant needs a DIFFERENT channel so the channel selection can't be unified — would still need 4 channel-specific gathers. Unexplored.
- Concat 160B: could a channel-axis Pad insert the 6 zero channels around a smaller active Concat? Active channels 0,4,6,7,8 are non-contiguous, so no single edge-Pad reproduces the interleave.

## INSIGHT (transferable)
⭐ Priority-overlay of K fixed-colour one-hot quadrants = pure BOOL logic (no Sub/Mul). out_hi = mask_hi; out_lo = mask_lo AND NOT(OR of all higher masks); ch0 background = NOT(OR of all masks). And/Or/Not all run on bool under ORT_DISABLE_ALL (1B), beating the public fp16 Sub/Mul net (2B) — but the LOAD-BEARING catch: ORT Pad REJECTS bool, so Cast the per-channel results to uint8 before Concat+Pad. Also: a one-hot overlay MUST emit channel-0 for background cells (= NOT-any-quadrant) or the padded-region-vs-in-grid background distinction fails silently (here the 4x4 block is fully covered, every cell sets exactly one channel).
