# task073 — 3618c87e

**Rule:** 5x5 grid at the canvas top-left (size fixed 5). Bottom row (row 4) is all
gray; each "tower" column c also has gray at row 3 and a single blue pixel floating
at row 2. Output keeps the gray towers + gray floor, removes the floating blue at
row 2, and paints blue onto the floor cell (row 4) of every tower column. Only
colours 0 (bg), 1 (blue), 5 (gray) ever appear. Purely VERTICAL per-column,
per-channel rearrangement.
**Current:** 20.50 pts, dwconv9x1 (depthwise 9x1 SAME conv), mem 0, params 90.
**Target tier:** S-ish (single Conv into the free output, mem 0) — minimise the
depthwise kernel HEIGHT (params = 10 * H).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | shrink depthwise kernel to the offsets actually used (-3..0) → height-4 conv | S | 0 | 40 | 21.31 | 200/200 | adopt candidate, beats 20.50 by +0.81 |

## Best achieved
**21.31 @ mem 0 params 40 — evaluate ok (pass 15/0), fresh 200/200.** Beats prior
20.50 by **+0.81**. Adopted? **N** (main adopts via `python -m src.adopt 73`).

## Irreducible-floor analysis
mem 0 (single Conv whose output IS the graph output). params = 10 channels * kernel
height. Per-channel taps:
- ch1 (blue): out[r]=in[r-2]  → offset {-2}
- ch5 (gray): out[r]=in[r]-in[r-1] → offsets {0,-1} (gray at row 3 marks the tower,
  subtracting it removes the floor gray exactly where blue lands)
- ch0 (bg):   out[r]=-in[r-3]+in[r-2]+in[r] → offsets {-3,-2,0} (restores the vacated
  row-2 tower cells; far -3 tap keeps SAME-pad edges correct)
Combined offset span across all channels = -3..0 → kernel height 4 → params 40.
Brute-forced: no ch0 kernel exists with span ≤ 2 (even with integer weights up to
±2), so height 4 is the floor for the single-depthwise-Conv form. The public net's
height-9 kernel was over-modelled (its ch0/ch5 kernels redundantly reached ±4).

## OPEN ANGLES (re-attack backlog)
- Below params 40 would need fewer than 10 conv groups (impossible — input is 10ch
  depthwise) or kernel height < 4 (ch0 brute-proven impossible). Any 2-op
  decomposition pays a ≥3600B 30×30 intermediate that drops the score well below
  21.31. Closed.

## INSIGHT (transferable)
⭐ A public dwconv "N×1" net is frequently over-modelled: its SAME-pad height is set
to the widest channel's reach, but the taps ACTUALLY used may span far less. Decode
the existing kernel (offsets with nonzero weight per channel), brute-search the
MINIMAL per-channel offset set that reproduces (out>0), then set the conv height =
combined offset span + 1 with an asymmetric pad. Here 9→4 cut params 90→40 (+0.81)
with mem still 0. Always check tap span before assuming a depthwise kernel height is
irreducible.
