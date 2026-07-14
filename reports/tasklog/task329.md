# task329 — d23f8c26

**Rule:** Square SIZE×SIZE grid (SIZE odd ∈ {3,5,7,9}) anchored top-left; each in-grid cell is bg(0) or a random colour. Output = input with everything zeroed EXCEPT the middle column (c == SIZE//2), which is copied verbatim. Off-grid cells = background (all channels off).
**Current:** 15.76 pts (prior). **Target tier:** A — spatial copy gated by a separable row⊗col in-grid mask, routed into the FREE Where output; no per-cell colour plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior on-disk: 3600B Conv colour plane + uint8 L plane + Equal | B | ~ | ~ | 15.76 | — | superseded |
| 2 | Where(fill, bg_onehot, input); fill = rowany∧(c≠mid)∧colany; mid=floor(SIZE/2); SIZE=#in-grid cols | A | 1302 | 41 | **17.80** | 500/500 | ADOPT-candidate |

## Best achieved
17.80 @ mem 1302 params 41 — beats prior 15.76 by **+2.04**. fresh 500/500.

## Irreducible-floor analysis
Dominant intermediate = the [1,1,30,30] bool `fill` plane (900B). It is the broadcast of `rowany[1,1,30,1] ∧ fillcol[1,1,1,30]`; ORT materialises the And result before the Where. Could in principle be shaved by folding the row/col gate into the Where, but Where takes a single cond tensor, so the 900B plane is the natural floor. Everything else is 1-D (≤120B) or scalar. No 3600B colour plane and no uint8 label plane (the prior method's two costs) are needed — the output is a pure spatial copy of `input`, so the 10-channel expansion lives entirely in the FREE output.

## Key structural insight that unlocked it
In the one-hot embedding an OFF-GRID cell has ALL 10 channels = 0 (harness only sets in-grid cells), while every in-grid cell sets exactly one channel (ch0 for bg). So `ReduceMax(input, axes=[1,2])` per column is 1 iff the column is in-grid, and `SIZE = #in-grid columns` exactly (0 mismatches / 60000). This sidesteps the colour-bbox approach, which is genuinely ambiguous: edge rows/cols can be all-background, and the generator even emits rare (≈1/15000) inputs where a 5×5 grid's coloured bbox extent = 2, indistinguishable from a 3×3 grid — using coloured occupancy would silently fail generalisation on those. Channel-0 (bg) occupancy resolves SIZE unambiguously.

## OPEN ANGLES
- Tier S would need the col==mid mask without the 900B And plane (fold row+col gate directly into a single Where cond) — not expressible since the in-grid row gate is needed to keep off-grid rows zero. ~17.8 is effectively the ceiling for this rule.

## INSIGHT (transferable)
⭐ For top-left-anchored grids, OFF-GRID = all-channels-off but IN-GRID bg = ch0-on, so `ReduceMax(input, axes=[1,2])` (per-col) / `axes=[1,3]` (per-row) recover the exact grid SIZE as a 120B vector — robust where coloured-pixel bounding box is ambiguous (edge rows/cols can be all-bg). A "keep one column/row" rule is then a pure spatial copy: `Where(non-mid ∧ in-grid, bg_onehot, input)` routes the whole 10-ch expansion into the FREE output, beating the label-map floor (~15.8→~17.8).

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 9 (static + 5000 fresh + bundled all agree). The flagged `keep_input_b` [30,30] bool = 900B is the **Where cond** for the free 30×30 output; padding it back re-adds 900B. Cropping to B=9 forces a valid-conv k=22 → 4840 params, and 9×9 Where slicing adds +6480B — the crop costs far more than the 900B it saves. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
