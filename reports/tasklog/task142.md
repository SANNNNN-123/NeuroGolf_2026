# task142 — 62c24649

**Rule:** A size-3 grid (colours 0..3) sits at the top-left corner; the output is a
2*size=6 square with 4-fold mirror symmetry — top-left quadrant = the grid, the other
three quadrants are its horizontal / vertical / both reflections (`out[r,c]=out[r,5-c]=
out[5-r,c]=out[5-r,5-c]=grid[r][c]`). Everything off the 6x6 block is ALL-ZERO (the
scorer's expected output has NO background fill; comparison is `(output>0)` vs the
one-hot 6x6). All 262 arc-gen instances are 3x3 -> 6x6.

**Current:** 17.67 pts, public GridSample-on-30x30-input + Pad, mem ~1440, params 81
**Target tier:** S (pure spatial copy/reflection — no colour/value computation)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice 3x3 + neg-step flips + Concat + OR bg-fill | S | 59220 | 91 | 14.0 | — | WRONG (bg-fill not needed) + full fp32 planes |
| 2 | Slice[ch0:4,3x3] → Cast fp16 → GridSample 6x6 → Pad(ch+spatial) | S | 504 | 90 | 18.61 | 200/200 | ADOPT |

## Best achieved
18.613 @ mem 504 params 90 — beats prior 17.67 by +0.94. fresh isolated 200/200.

## Irreducible-floor analysis
Dominant intermediate is now the GridSample output `[1,4,6,6]` fp16 = 288B (the 6x6
mirror block on the 4 active colour channels, half precision). The slice `[1,4,3,3]`
fp32 (144B) + its fp16 cast (72B) round it out. The public net's 1440B floor came from
GridSampling the FULL 30x30 fp32 input → `[1,10,6,6]` fp32; we cut it ~3x by (a) slicing
to channels 0..3 AND the 3x3 active corner in ONE Slice BEFORE the gather (the input is
free, so a tiny 36-elem slice is far cheaper than sampling all 10 ch), and (b) casting
that 36-elem block to fp16 (the Cast is cheap because the slice is already tiny — the
"don't cast the full input" rule only bites on the 30x30 plane). Pad then restores both
the channel axis (4→10) and spatial (6→30) with zeros straight into the FREE output.

## OPEN ANGLES (re-attack backlog)
- Could drop the fp32 slice intermediate (144B) if a single op could both channel/spatial-
  crop AND cast to fp16 — none exists (Slice preserves dtype). Marginal (~30B) anyway.
- Grid is 72 params (6x6x2 fp16); a coarser/shared grid won't shrink it (every output cell
  needs its own (x,y)). Not worth chasing — params are already dwarfed by nothing.

## INSIGHT (transferable)
⭐ When a public net GridSamples/gathers the FULL 30x30 fp32 input to build a small output
block, you can usually shrink that single dominant intermediate ~3-4x by (1) Slicing the
input to the ACTIVE colour-channel subset AND the active spatial corner in ONE Slice
*before* the gather (input is free, so a 36-elem slice costs nothing vs sampling all
10ch×full grid), then (2) Casting that tiny block to fp16 and gathering in fp16. Pad
restores both the channel axis and the spatial extent with zeros into the FREE output.
The "fp16 doesn't help / don't cast the input" rule is about the 30x30 ENTRY plane — once
the working block is tiny, fp16 halves it for free. ⭐ Also: re-check the scorer's expected
output for off-grid fill — here off-grid is ALL-ZERO (no ch0=1 background), which removed
an entire OR/mask branch the "obvious" Tier-S build wrongly added (and tanked it to 14.0).

## S9 (2026-07-03) — mechanism-14 separable-remap einsum (+0.449) ADOPTED
Single 5-operand Einsum 'ra,ai,zcij,bj,sb->zcrs', mem=0: mirror-tile 3x3->6x6, U(30,3)/S(3,30) shared both axes, 282->180.
Gates: stored fail=0; uncached fresh 2000+600: 0/0/0 (bit-identical). No TopK.
NOTE: scan projection was ~8x optimistic — output axis must span the FULL 30 (grading
tensor [1,10,30,30]), so U tables are [30,K] not [out,K]. Backup task142_pre_s9.onnx.
