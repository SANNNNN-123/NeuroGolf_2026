# task127 — 54d9e175

**Rule:** Canvas is 11 wide, height 3 or 7. Gray (5) grid-lines at cols c∈{3,7} and (when height 7) row r=3 partition the grid into a (1 or 2)×3 layout of 3×3 cells. Each cell's CENTER pixel (4R+1,4C+1) holds a color k∈{1,2,3,4}. Output: fill that cell's whole 3×3 block with k+5 (channels 6–9); gray grid-lines stay gray (5); everything else background (0).
**Current:** 18.20 pts, single `Conv(input, W[10,10,3,3])` (37 nonzero weights), mem 0, params 900.
**Target tier:** mem-0 single-conv — already at HARD floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | reproduce optimal dense single conv (dilate colors→k+5, copy gray) | conv | 0 | 900 | 18.198 | 200/200 | matches existing; no improvement possible |
| 2 | depthwise dilate (90p) + Gather channel-shift | — | 36000 (full 10ch) | 90 | 14.5 | — | rejected: 10ch intermediate floor |
| 3 | active-region slice + small conv + Pad | — | ≥3080 (slice+conv out) | 225 | ≤16.9 | — | rejected: slice+conv intermediates cap below 18.20 |

## Best achieved
18.198 @ mem 0 params 900 — adopted? N (identical to existing net). Beats prior 18.20? **N (MARGINAL — tie, 0 gain).**

## Irreducible-floor analysis
This is the prompt's ⛔ MEM-0 SINGLE-CONV-AT-FLOOR case, confirmed irreducible:
- **Output IS the graph output ⇒ mem 0.** Any decomposition materializes a multi-channel plane costing ≥1540B (fp16 active 7×11) up to 36000B (fp32 full 10ch), each of which alone drops the score below 18.20.
- **Channel shift color k → channel k+5 forbids groups.** Verified for groups 10/5/2/1: no group partition keeps input-channel k and output-channel k+5 in the same block — only group=1 (dense) works. So the weight is a DENSE cross-channel [10,10,3,3].
- **Dilation footprint forced to 3×3.** A cell's corner pixel (offset (0,0)) must tap its center at relative offset (+1,+1); k=3 is minimal.
- **params count ELEMENTS (dtype-free)** ⇒ fp16 weight does not shrink 900.
⇒ 10·10·3·3 = 900, mem 0, score 25−ln(900) = 18.198. To beat by +0.3 needs params+mem < exp(6.5) ≈ 665, unreachable: the dense channel-shift conv is exactly 900 and mem-0, and the only way to fewer params (grouped, 90–450) cannot perform the +5 shift.

## OPEN ANGLES (exhausted)
- No separable / dilated-stride / count→pattern / scalar-recovery escape exists: the op genuinely taps a neighbour (cell center) AND requires a cross-channel recolor, the two constraints that pin a dense 3×3 single conv. None of the PROVEN LEVERS reduce a dense channel-mixing conv whose output already is the graph output.

## INSIGHT (transferable)
A mem-0 single dense Conv that performs BOTH a small spatial dilation AND a cross-channel relabel (color k → channel k+5) is at hard floor: the relabel forbids grouping (forcing dense in×out), the dilation forbids 1×1 (forcing the footprint), and splitting the two steps always pays a ≥1.5KB multi-channel intermediate that beats mem-0 only BELOW the existing score. Discriminator vs golfable conv tasks: if removing the dilation leaves a pure channel permutation that a Gather could do for free, it's still blocked because the Gather needs a materialized dilated plane. ⭐ "dilate + channel-shift in one conv" = confirmed MEM-0 floor sibling of task120/task095.
