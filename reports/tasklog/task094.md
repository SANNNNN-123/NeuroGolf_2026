# task094 — 41e4d17e

**Rule:** Grid size 15, background cyan(8). Input holds 1-2 blue(1) 5x5 box *outlines*
(square perimeters) centred at (r,c), r,c in 3..11, boxes well separated (>=3 gap each
axis, crosshairs never touch). Output keeps the blue boxes and paints, for each centre
(r,c), the entire row r and column c pink(6). Blue is drawn after pink, so blue overwrites
pink at overlaps.
**Current:** 15.767 pts, ext:biohack_new, mem 10115, params 112
**Target tier:** A (separable label-map) — the crosshair mask is row-profile OR col-profile
(separable), output is a 3-colour label map → final Equal into free BOOL output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 5x5 outline-Conv (no pad) → centre profiles → label map | A | 7575 | 289 | 16.03 | (offset bug) | Conv top-left aligned, crosshair off by (-2,-2) |
| 2 | + pads=[2,2,2,2] to centre the Conv response | A | 7575 | 289 | 16.03 | 265/265 | correct, MARGINAL (+0.26) |
| 3 | drop dead 15x15 resp slice; ReduceMax full resp → crop 1-D profiles | A | 6690 | 287 | 16.15 | 200/200 | beats P by +0.38 ✓ |
| 4 | Conv on the 1-ch 15x15 blue SLICE (reuse blue_f) instead of 10-ch 30x30 input | A | 3750 | 58 | 16.76 | 500/500 | beats 16.15 by +0.61 ✓✓ |

## Best achieved
16.76 @ mem 3750 params 58 — adopted? N (orchestrator gates). Beats prior 16.15? Y (+0.61).
KEY: the Conv never needed the full 10-ch 30x30 input. `blue_f` (the [1,1,15,15] fp32 blue
slice, already computed for the overlay mask) IS the conv input → resp drops 3600B→900B and
the kernel drops from [1,10,5,5] to [1,1,5,5] (params 287→58). The 30x30 conv crop constants
(c0/c15/ax2/ax3) and the 30-wide ReduceMax/Slice all vanish.

## 1-D-only angle (re-attack hint) — REJECTED, INFEASIBLE for correctness
The suggested "drop the 2-D Conv, use 1-D ReduceSum blue-count row/col profiles" does NOT work.
Per-row blue count is 5 at box edges (r±2) and 2 at inner rows (r-1,r,r+1) — the centre row's
count (2) is identical to its neighbours, so no threshold isolates it. A 1-D conv [1,0,0,0,1]
peaks at 10 (5+5) at the centre, BUT two boxes whose edges align at row-distance 4 produce a
phantom 5+5=10 peak at a NON-centre row. Valid generator configs hit this (e.g. dr=8 → phantom
at r1+4; dr=4 with dc>=6 likewise). Verified false positive at row 7 for boxes (3,3),(11,11).
The 2-D Conv is REQUIRED to bind the full outline at one location.

## Irreducible-floor analysis (after attempt 4)
Per-tensor mem (total 3750): blue_f [1,1,15,15] fp32 900 + resp [1,1,15,15] fp32 900 +
L [1,1,30,30] uint8 900 (off-grid sentinel pad, required since output is 30x30, ORT Pad
rejects bool) + cross/is_blue/L_a/L15 (225 each) + profiles (~150). The two fp32 planes are
the floor: Conv input/output must be float and the active grid is 15x15. Off-grid blue is
confined to rows/cols 1..13, so a 13x13 crop (676B each, verified correct) would shave ~448B
(~16.87) but needs +1 offset Pads on the 1-D profiles back to 15 — marginal, not taken.

## OPEN ANGLES (untried)
- 13x13 blue crop ([1:14]) → resp/blue_f 676B each, +offset Pad on profiles → ~16.87 (+0.1).
- Cast resp fp16: adds a tensor, can't elide the fp32 Conv output → net worse.

## INSIGHT (transferable)
⭐ Box/ring CENTRE detection = one Conv with the ring's exact perimeter kernel; the response
peaks at the perimeter's pixel-count at the centre and is strictly lower elsewhere, so a single
Greater(thr) isolates centres with NO flood-fill. CRUCIAL: a default (no-pad) Conv aligns the
peak to the window's TOP-LEFT, not its centre — add `pads=[k,k,k,k]` (SAME) to land the peak on
the geometric centre. When the rule paints full independent rows AND full independent cols, the
2-D crosshair is SEPARABLE: reduce the centre plane to is_row[1,1,H,1] OR is_col[1,1,1,W] and
let the broadcast happen in the free final ops.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.001)
**Mechanism (op-census diff):** Detection conv kernels `row_w`/`col_w` shrunk **5→3 taps** ([1,1,5,1]→[1,1,3,1], [1,1,1,5]→[1,1,1,3]); 3 taps empirically suffice. −4 params, mem flat.
**Old→new:** mem 2662→2662, params 51→47.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task094_pre_s10.onnx`; source `public_candidates/bobmyers7186/task094.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
