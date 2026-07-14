# task386 — f2829549

**Rule:** Input is a fixed 4x7 grid with a blue separator column at col 3. Left
half (cols 0..2) carries orange (colour 7) pixels, right half (cols 4..6) carries
gray (colour 5) pixels. The 4x3 output is green (colour 3) at cell (r,c) iff BOTH
the left cell (r,c) AND the right cell (r,c) are empty — i.e. a per-cell NOR of the
two half-grids. Non-green output cells are background (colour 0); geometry is fixed
(width=3, height=4) for every instance.
**Current:** 18.24 pts, ext:wguesdon6304 (slice/Or/Not + 2ch Concat + Conv[10,2,1,1] + Pad), mem 828, params 35
**Target tier:** A — per-cell separable logic on a tiny fixed 4x3 active region; the
10-ch expansion routes into the free output via Pad, no full-canvas plane needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice ch7/ch5 -> Or/Not -> Pad single-ch into ch3 only | A | 156 | 24 | (fail) | - | ch0 (bg) unset for non-green cells |
| 2 | label L 4x3 -> Pad to 30x30 sentinel 10 -> Equal(arange) | A | 1056 | 36 | 18.00 | - | 900B 30x30 L plane too big |
| 3 | Equal on SMALL 4x3 label -> [1,10,4,3] uint8 -> Pad spatial | A | 396 | 35 | 18.93 | - | beats, but Equal+Cast = 240B |
| 4 | drop uint8 Cast; opset-13 bool Pad on [1,10,4,3] one-hot | A | 276 | 36 | 19.26 | 200/200 | ADOPTED |

## Best achieved
19.26 @ mem 276 params 36 — adopted? Y. Beats prior 18.24? Y (+1.02).

## Irreducible-floor analysis
Dominant intermediates: the [1,10,4,3] bool one-hot block (120B) + two fp32
half-grid slices (48B each = 96B). The one-hot must be 10-channel (output is
10-channel) over the 4x3 active region and both channel 0 (bg) and channel 3
(green) must be emitted, so 120B is the floor for the channel expansion of the
active block. The fp32 slices are forced fp32 by the input dtype (Slice preserves
dtype); casting the input to fp16 first would cost 18000B. Spatially the block is
already minimal (4x3). Not worth chasing further — near floor for a two-channel
fixed-region emit.

## OPEN ANGLES (re-attack backlog)
- Collapse the two fp32 slices: both occupancy signals are input ch0 (left empty =
  ch0 over cols0:3, right empty = ch0 over cols4:7), green = AND of the two ch0
  slices. Same op/byte count (two 48B slices) so no win, but a single ch0 slice
  cols0:7 ([1,1,4,7]=112B) + a shifted AND could in principle fuse — measured
  bigger, abandoned.
- Build only channels {0,3} via Concat with zero-channel inits instead of Equal —
  same 120B [1,10,4,3], no win.

## INSIGHT (transferable)
⭐ Two-channel fixed-region emit (bg ch0 + one fg colour) where output cells are
either the fg colour or background: the per-cell NOR / mask -> uint8 label ->
`Equal(label, arange[1,10,1,1])` on the SMALL active block (NOT padded to 30x30
first) -> opset-13 bool Pad straight to the [1,10,30,30] BOOL output. Padding the
LABEL to 30x30 before Equal costs a 900B plane (task attempt 2); doing Equal on the
4x3 block first keeps it at 120B. The off-grid zero-fill of the bool one-hot is
exactly the all-channel-zero one-hot target, and channel 0 falls out for free as
the L==0 match (every non-fg output cell is background). Opset-13 bool Pad removes
the uint8 Cast entirely (saves the duplicate block). 18.24 -> 19.26 (+1.02).
