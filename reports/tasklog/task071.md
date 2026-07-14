# task071 — 3345333e

**Rule:** 16x16 grid, two colours. The SPRITE (colors[0]) is a continuous creature
drawn MIRROR-SYMMETRICALLY about a vertical axis (every pixel placed at both
`scol+c` and `scol-c-off`, so axis2 = 2*scol-off). A SOLID width-4 box (colors[1])
is drawn ON TOP, occluding part of the sprite. Output = the sprite mirror-completed
(`sprite OR reflect(sprite)` about the axis), with the box removed entirely (flip
augmentation reflects the whole instance, which only changes the axis value).
**Current:** 13.79 pts, blank-note "confirmed-infeasible" (NO documented reason).
**Target tier:** A — closed-form symmetric reconstruction, no flood-fill/loop.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | reflection-matrix stack over 30x30 | A | 317865 | 13619 | 12.29 | - | works, too big |
| 2 | crop to 16x16, fp16 axis stack | A | 68091 | 2918 | - | - | fp16 upcast, 1 fail (axes too narrow) |
| 3 | anti-diagonal coincidence (no [NA,W,W] stack) | A | 46830 | 3932 | 14.17 | - | big drop |
| 4 | 1x1 Conv colf + slice (kill 10-ch product) | A | 35550 | 3970 | 14.42 | - | |
| 5 | slice occupancy + uint8 output pad | A | 29898 | 3424 | **14.59** | 200/200 | ADOPTED-candidate |

## Best achieved
14.59 @ mem 29898 params 3424 — beats prior 13.79 by **+0.80**. fresh 200/200.

## 2026-06-29 live-frontier refresh

Current live/source is much smaller: **17.100253 pts @ mem 2643 params 55**.  It
uses the generator's fixed active crop (`bg_active` [1,1,11,13]) rather than the
old full reflection-overlap stack.  Axis recovery comes from the top visible
sprite row, then a 13-wide reflected gather restores the mask; output is a
one-channel uint8 active label padded to 16 and then 30 before final Equal.

The obvious 900B `color30` label plane is not a free gain.  Making bool one-hot
before Pad costs 10x16x16 = 2560B, or 10x11x13 = 1430B plus pads, both worse
than the current one-channel label carrier.  The next largest tensors are
`bg_active` 572B and 143B bool active/reflection masks, so +0.3 would need a new
axis/output carrier, not local graph surgery.

## Key construction
- Box channel = SOLID (cnt == bboxW*bboxH) AND bboxW == 4 (0/5000 fresh).
  (`solid` alone fails ~1/3000; `width==4` alone ~1.8%; AND-ed = 0/5000.)
- Sprite channel = the other non-bg (ch>0) channel with pixels.
- Axis recovery WITHOUT a per-axis [NA,16,16] stack: overlap_a = sum over the
  anti-diagonal c+c'=a of the column-coincidence matrix `C = Isp^T @ valid`
  (valid = sprite OR box; the box fills occluded mirror cells). The true axis is
  the UNIQUE one where every visible sprite cell has an in-grid mirror landing on
  a valid cell => overlap_a maximal => argmax. (unique 0/2000 verified.)
- Reflection matrix `Rsel[k,c] = Equal(k+c, axis2)` built directly from the scalar
  axis (no stack); reconstruct `Isp OR (Isp @ Rsel)`.
- Output: uint8 colour plane padded to 30x30 with sentinel 99 (off-grid pad ->
  Equal-all-False -> all-zero target), Equal(L_u8, arange10_u8) -> BOOL free output.

## Irreducible-floor analysis
Dominant: the 3600B fp32 entry colour-index plane `colf30` (1x1 Conv over the full
30x30 input — Conv output spatial = input spatial, can't shrink without paying a
10240B input slice instead). Next: two [1,10,30,1]/[1,10,1,30] occupancy profiles
(1200B each) for the box discriminator, and ~12 small 16x16 fp32 working planes
(1024B each). fp16 on the 16x16 / [NA,16,16] planes does NOT help — ORT inserts a
PrecisionFreeCast that the scorer counts at fp32 (cost the same or worse).

## OPEN ANGLES
- Fuse the box discriminator's width/solidity to use only `cnt` + a single
  profile (e.g. detect the 4-wide constant-height column block) to drop one of the
  two 1200B occupancy planes (~−1.2KB, ~+0.04).
- Replace the per-channel bbox entirely by a 2-hypothesis self-consistency check
  (reconstruct for each candidate sprite colour, keep the one whose reflection is
  consistent) — removes occupancy planes but adds a second axis/reflect path.

## INSIGHT (transferable)
⭐ A "complete the mirror-symmetric shape behind an occluder" task is NOT a
connectivity/flood wall — it is closed-form. Reflection-overlap over K candidate
axes does NOT need a [K,H,W] stack: it equals the ANTI-DIAGONAL SUMS of the column
coincidence matrix `C = A^T @ B` ([W,W]), read out by ONE small matmul `Cflat @ D`
(D[flat(c,c'),a] = (c+c'==a)). And the chosen reflection matrix is built directly
from the recovered scalar axis as `Equal(k+c, axis)` — no per-axis stack anywhere.
Occlusion-robust axis = the unique axis where every visible cell's mirror lands on
(shape OR occluder). Re-confirms blank-note "confirmed-infeasible" labels are
~false-positive: this one broke +0.80.
