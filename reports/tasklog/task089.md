# task089 — 3e980e27

**Rule:** Grid always 13x13. Two 3x3 sprites: idx0 carries one RED(2) marker + body of colour A; idx1 carries one GREEN(3) marker + body of colour B. Each sprite is stamped at several non-overlapping mega-positions (bounding boxes separated by >=1 empty row/col, so sprites are never even diagonally touching). The FIRST occurrence of each sprite is drawn in FULL in the input; later occurrences show ONLY the marker pixel. idx0's later copies are COLUMN-MIRRORED (c->2-c); idx1 is never mirrored. OUTPUT = draw the full sprite at every marker.
**Current:** 14.26 pts, method gen:vyank6322, mem 46056, params 98
**Target tier:** A (closed-form stamp via runtime conv kernel on a small 13x13 canvas; no global argmax over variable components)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 5x5 marker-centred window as conv kernel | - | - | - | - | 282/300 ref | FAIL: window catches neighbour sprite at offset 2 (gap=1) |
| 2 | per-marker bounded 8-dilation isolates sprite (4 steps), full-marker = body-reach, runtime conv kernel placement | A | 26082 | 114 | 14.83 | 200/200 | ADOPT |

## Best achieved
14.83 @ mem 26082 params 114 — beats prior 14.26 by +0.57 (Y).

## Irreducible-floor analysis
Dominant intermediate is the fp16 13x13 working planes (~338B each) of the two dilation chains (8 MaxPool+Mul each) plus the 30x30 fp16 colf30 Conv output (~7200B as the only full-30x30 plane besides the free input/output). The colf30 Conv output is fp32 [1,1,30,30]=3600B — the one irreducible entry plane (10->1 reduction must be fp32). All downstream work is on the 13x13 slice in fp16, which is why mem is far under the old 46k. Not at a wall — see open angles.

## OPEN ANGLES (re-attack backlog)
- The colf30 [1,1,30,30] fp32 Conv output is sliced to 13x13 immediately; if Conv could emit only the in-grid region (it cannot without a runtime crop) ~3600B would drop. Minor.
- The two dilation chains (4 steps x2 = 8 MaxPools each colour) could possibly be shortened to 3 steps if sprite diameter <=3 always holds (diagonally_connected 3x3 -> max 2 diagonal steps from any cell, so 2-3 steps may suffice) — would shave a few fp16 planes. Verified 4 is safe; 2-3 untested for full 200/200.
- Marker-position scalars use rowramp/colramp Muls (small). Fine.

## INSIGHT (transferable)
"Stamp a data-dependent sprite at every marker" is NOT a connectivity/shape-correspondence wall when (a) the canvas is generator-bounded small (13x13) and (b) sprite instances are guaranteed non-touching (overlaps(spacing>=1) => no 8-connectivity across sprites). Then: a per-marker BOUNDED dilation gated by occupancy isolates each sprite (lone markers stay single); the FULL sprite is recovered as dilate(body) where body = dilate(marker) AND NOT marker (lone markers have no body); the full-marker position is two ReduceSum(mask*ramp) scalars; the template re-centred on its marker becomes a RUNTIME 5x5 Conv weight (Gather a fy/fx-indexed window of the clean sprite plane); and stamping at every marker is ONE Conv(markermap, rot180(kernel), pad=2) — stamps never overlap so the conv sum per cell is a single tap. A per-instance column-mirror folds into a step=-1 Slice of the runtime kernel before rot180. ⭐ Runtime-computed Conv weights (non-initializer 2nd input) are accepted by ORT and turn "place this learned-at-runtime stencil everywhere" into a single op.

## S8 (2026-07-02) — chained-scatter fold (+0.024) ADOPTED, div 0
Two chained ScatterElements (reduction=max, disjoint writes) → ONE scatter over concatenated [96] idx/update tensors; drops the 169B intermediate. LESSON (cross-task): K-batching repeated planes is BYTE-NEUTRAL (grader sums elements) — wins only come from ELIMINATING intermediates.

## S16 (2026-07-06) — public bit-identical golf (franksunp) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
