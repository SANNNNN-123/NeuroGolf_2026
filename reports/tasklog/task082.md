# task082 — 3ac3eb23

**Rule:** Input has coloured pixels only in row 0; grid height is always 6, width 5..15. Each marked
column `c` (colour `color`) becomes a 3-wide vertical stripe in the output: on EVEN rows (0,2,4) the
centre col `c` shows `color` and the two sides `c±1` are black; on ODD rows (1,3,5) the centre is black
and the sides show `color`. Output is a separable row-parity ⊗ column-pattern, fully determined by row 0.
Marked columns are ≥3 apart so neighbour stripes never collide.

**Current:** 16.88 pts, small-active-canvas (15-wide) + per-column closed-form + parity-MatMul into FREE output, mem 3000, params 366
**Target tier:** A (separable per-column closed form, no detection/flood/argmax).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: 30-wide fp32 2-row template + MatMul | A | 4800 | 363 | 16.45 | — | baseline |
| 1 | fp16 everywhere, 30-wide | A | 3600 | 363 | 16.72 | — | +0.27 marginal |
| 2 | crop to 16-wide, Pad the CANVAS after MatMul | A | 11520 | 366 | 15.62 | — | WORSE (pre-pad canvas plane counts) |
| 3 | hybrid: 6-row conv stripes on full 30×30 + bg recon | A | 79200 | 1803 | 13.70 | — | exact but full 30×30 planes, far worse |
| 4 | crop 16-wide, Pad the TEMPLATE before MatMul | A | 3120 | 366 | 16.84 | — | +0.39 |
| 5 | **crop 15-wide (true max content col 14), Pad template** | A | 3000 | 366 | 16.88 | 500/500 | **ADOPTED +0.43** |
| 6 | depthwise colour conv (30p) + 1-ch bg conv | A | 3600 | 129 | 16.78 | — | param cut but +3 planes raises mem |

## Best achieved
16.88 @ mem 3000 params 366 — adopted? draft only (no commit, per constraints). Beats prior 16.45? **Y, +0.43**.
Fresh: ISOLATED 200/200 then 500/500; second seed 300/300; edge cases (w5 single col, w15 col13 → content col14) all exact.

## Irreducible-floor analysis
Dominant intermediate = the Pad'd 30-wide 2-row template `tpl` [1,10,2,30] fp16 = 1200B. It MUST be 30-wide
because the parity-MatMul `P[30,2]@tpl` carries the width axis straight into the FREE 30×30 output — the only
way to keep the 10-channel expansion free. The fp32 entry slice `x32` [1,10,1,15] = 600B is the irreducible
entry (Slice preserves the fp32 input dtype; it cannot emit fp16). params dominated by the cross-channel
odd-row conv kernel G [10,10,1,3] = 300 (shape-fixed: out_ch=10 incl bg, in_ch=10 since the bg channel sums
all colour channels, kw=3 for the ±1 neighbour shift). Splitting G into a depthwise colour conv (30p) + a
1-channel bg conv (30p) cuts params to 129 but the channel slice+concat to rebuild the 10-ch odd row adds
~600B of planes → net worse. So at this structure 3000B/366p ≈ the floor.

## OPEN ANGLES (re-attack backlog)
- Remove the Pad'd `tpl` (1200B): need a 30-wide free-output op that broadcasts a 15-wide template without
  materialising it. No idiom found — MatMul/Conv both carry the spatial axis, and routing into the free
  output via And/Or of parity⊗column broadcasts needs TWO full 30×30 And planes (worse). Likely a hard wall.
- Drop the fp32 entry (600B): only achievable if some op could emit a fp16 row-0 slice directly. Slice/Gather
  preserve fp32; casting the full 10-ch input is 18000B. No path.

## INSIGHT (transferable)
⭐ CROP-WIDTH + PAD-THE-TEMPLATE (not the canvas): when a net broadcasts a small per-axis template into the
FREE output via MatMul, crop the working axis to the active extent and `Pad` the *small TEMPLATE* up to 30
BEFORE the broadcast — so the broadcast op still writes the full 30×30 FREE output. Padding the *canvas*
AFTER the broadcast makes the pre-Pad 30×30 plane count (the classic crop-to-active trap) and is net-negative,
but padding the 2-row template is cheap (1200B) and the MatMul output stays free. This combined crop+fp16 took
task082 16.45→16.88 (+0.43). Also: for a per-column closed form the bg (ch0) channel is reconstructed inside
the SAME odd-row conv (`Σ_k x[k,w] − Σ_{k≥1}x[k,w±1]`) with no extra plane — splitting it out to save conv
params costs more in planes than it saves.
