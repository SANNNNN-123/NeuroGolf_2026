# task083 — 3af2c5a8

**Rule:** Fixed-geometry 4-fold reflection (D2) un-fold. Input is always a 3-row x 4-col
grid (height=3, width=4) of ONE colour on background-0 at the top-left of the 30x30
canvas; output is always the 6x8 grid that is the input quadrant mirrored 4 ways:
out[r][c]=out[2H-1-r][c]=out[r][2W-1-c]=out[2H-1-r][2W-1-c]=in[r][c] (H=3,W=4). Off-grid
target is all-zero. Geometry is constant every instance (generate() always uses 4x3);
only pixel positions and the single colour vary.

**Current:** 17.50 pts, ext:kojimar6275, mem 0, params 1800
  -> a SINGLE `GridSample(input, grid[1,30,30,2])` op: one op remaps each output cell
     from input at reflected coords with zero-padding off-grid. mem 0 (only output
     tensor), params = 30*30*2 = 1800.

**Target tier:** S — already a memory-0 single-op remap; the only lever is shrinking the
1800-element grid, which is impossible for a full 30x30 output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice+revSlice+Concat+Pad copy (fp32) | S(copy) | 4800 | 23 | 16.52 | 200/200 | below current |
| - | same w/ fp16 working chain + 1 fp32 cast | S | ~4320 | 23 | ~16.6 | - | still below |
| - | Pad-input floor only [1,10,6,8] fp32 | S | 1920 | ~25 | ~17.43 | - | still below 17.50 |
| - | Where(mask30, color_oh, bgfill) routing | - | +18000 | small | <12 | - | needs 2nd full plane |

## Best achieved
16.52 @ mem 4800 params 23 (copy approach, exact, fresh 200/200). Beats prior 17.50? NO.
Pad-input-floor variant tops out ~17.43, still below the GridSample 17.50.

## Irreducible-floor analysis
The output is a full [1,10,30,30] tensor, so the output-producing op's "shape source"
must cover 30x30. Three structural routes, each with a hard floor:
 (1) GridSample: grid MUST be [1,30,30,2] = 1800 elems (params count = element count,
     dtype-free), mem 0. THIS IS THE STORED NET. Cannot shrink the grid for a 30x30 out.
 (2) Slice/Concat/Pad copy: the Pad input must be [1,10,6,8] fp32 (Pad only pads spatial
     dims, channel count fixed at 10; output dtype fp32 forces fp32 input) = 1920B floor.
     Plus the reversal/concat working chain. Best ~17.43 < 17.50.
 (3) Where/double-MatMul routing of the 10-ch one-hot: forces a second [1,10,30,30]
     plane (18000B) because off-grid must be zero while in-grid bg must be ch0 (3
     distinct fill regions cannot collapse into one [1,10,1,1] Where else-branch, and
     `input` as fallback only covers the TL quadrant). Disqualified.
To beat 17.50 by +0.3 needs mem+params < e^7.2 = 1339; the smallest single-op
full-canvas remap initializer is 1800 elems. No cheaper structure exists.

## OPEN ANGLES (exhausted)
- None viable. GridSample 1800-param grid is the structural minimum for a memory-0
  full-30x30 spatial remap. fp16 grid does not help (params = element count, not bytes).
- A smaller GridSample grid would require a smaller output canvas, which is fixed at
  30x30 by the I/O contract.

## INSIGHT (transferable)
A fixed-geometry full-canvas spatial REMAP (reflection/rotation/translation that fills
the whole 30x30 output) is already optimal as a single `GridSample(input, grid[1,30,30,2])`:
mem 0, params 1800. This is a HARD FLOOR (~17.50) — the grid cannot be shrunk below
30*30*2 for a 30x30 output, and any Slice/Concat/Pad rebuild needs the [1,10,6,8] fp32
Pad-input (1920B -> ~17.43) which is strictly worse. Do NOT spend time trying to beat a
GridSample-based ext import on a full-canvas geometric remap; it is at floor.

## S9 (2026-07-03) — mechanism-14 separable-remap einsum (+0.063) ADOPTED
Single 5-operand Einsum 'ra,ai,zcij,bj,sb->zcrs', mem=0: mirror-tile 3x4->6x8, Ur/Sr(3) + Tc/Vc(4), 447->420 (thin).
Gates: stored fail=0; uncached fresh 2000+600: 0/0/0 (bit-identical). No TopK.
NOTE: scan projection was ~8x optimistic — output axis must span the FULL 30 (grading
tensor [1,10,30,30]), so U tables are [30,K] not [out,K]. Backup task083_pre_s9.onnx.
