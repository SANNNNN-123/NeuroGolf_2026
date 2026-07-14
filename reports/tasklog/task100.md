# task100 — 445eab21

**Rule:** A `size`=10 grid (on the 30×30 canvas) holds TWO hollow rectangle OUTLINES drawn in two distinct colours `colors[0]`/`colors[1]`, with widths/heights `wide`/`tall`. The 2×2 output grid is filled entirely with the colour of the box whose AREA `wide*tall` is larger. `xpose` may transpose the grid but never changes which box is larger. Harness embeds the 2×2 output at top-left of a 30×30 array: channel=winner True at cells (0..1,0..1), all else False.
**Current:** 16.21 pts (public net)
**Target tier:** A — closed-form scalar argmax + separable free-output route; no per-cell colour/value plane needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bbox via ramp min/max + area=span*span, full-30 planes | A | 6648 | 146 | 16.18 | — | below P |
| 2 | slice occ to active 10 region; assoc final And chain | A | 4848 | 110 | 16.49 | — | +0.28 marginal |
| 3 | area = ReduceSum(occ rows)*ReduceSum(occ cols) (drop ramp/min/max) | A | 3738 | 86 | 16.75 | 200/200 | ADOPTED +0.54 |

## Best achieved
16.75 @ mem 3738 params 86 — beats prior 16.21 by +0.54. fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate: the two full-30 fp32 occupancy profiles `rowocc[1,10,30,1]` and `colocc[1,10,1,30]` = 1200B each (2400B total). ReduceMax over the input emits fp32 at the full 30-row/col extent before the active-region Slice, and ReduceMax/ReduceSum cannot output a narrower dtype than the fp32 input. Everything else is tiny ([1,10,1,1] scalars, [1,10,30,1] bool routing plane 300B). Cannot remove without slicing the fp32 INPUT first (a [1,10,10,10]=4000B plane — strictly worse).

## OPEN ANGLES (re-attack backlog)
- Avoid one of the two full-30 occ planes by deriving colspan from a transposed reuse of rowocc — likely net-neutral (Transpose copy ≈ saved plane).
- Replace ReduceMax-over-input with a no-pad row/col SUM-Conv (W[1,10,1,30] kernel, ch0 weight 0) — folds "drop ch0 + collapse axis" but still emits a full [1,1,30,1] fp32; marginal.

## INSIGHT (transferable)
⭐ For a SOLID/OUTLINE bbox whose edges reach every side, bbox dimensions = pixel COUNT of the 0/1 occupancy profile: `tall = ReduceSum(rowocc, axis=row)`, `wide = ReduceSum(colocc, axis=col)` — strictly cheaper than the ramp-Where + ReduceMin/ReduceMax(min/max-coord) idiom (drops 4 fp16 ramp planes + 4 reductions). area = tall*wide; argmax over channels with ch0 forced to −1.
