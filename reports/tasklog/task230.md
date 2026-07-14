# task230 — ARC-AGI 95990924

**Rule:** Input has several non-overlapping 2x2 GRAY (colour 5) blocks at top-left
(r,c). Output copies the gray blocks and adds four single coloured pixels at the
diagonal-outer corners of each block: out[r-1][c-1]=1, out[r-1][c+2]=2,
out[r+2][c-1]=3, out[r+2][c+2]=4. Satellites always fall on background (in-grid,
no collisions). Fully local closed-form stamp — NOT detection/connectivity.
**Current:** base net stored ~18.2 but FRESH-RATE 0.00 (does not generalize, real ~0).
**Target tier:** A — local separable conv stamp; output colours are FIXED {1..5}
so a label plane + Equal one-hot routes the 10-ch expansion into the FREE output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32: slice ch5, 2x2 count-conv ==4, 4x4 satellite-stamp conv, L=5g+sat, in-grid gate, int32 Equal | A | 37800 | 37 | 14.46 | 200/200 | works |
| 2 | same but fp16 chain (cast entry plane, Greater>3.5 instead of int Equal) | A | 25200 | 37 | 14.86 | 200/200 | adopted-candidate |
| 3 | count-conv on FULL input -> fp16 stamp-conv -> single fp16 label plane -> Equal | A | 11100 | 78 | 15.68 | 200/200 | works |
| 4 | + slice working canvas to 15x15 (in-grid square always <=15); rowany/colany in-grid mask | A | 8490 | 96 | 15.94 | 200/200 | works |
| 5 | + slice CHANNEL-5 corner FIRST (900B fp32 entry) instead of 3600B full-plane conv | A | 5914 | 63 | 16.30 | 200/200 | works |
| 6 | + uint8 label plane through Pad (halves the 30x30 output plane vs fp16) | A | 5014 | 63 | 16.47 | 200/200 | works |
| 7 | + SAME-pad count conv on 15x15 g5 (drop the 16x16 slice to 15x15) | A | **4890** | 63 | **16.49** | 200/200 | **best** |

Tried GC=13/14 detection-canvas < stamp-canvas (separate Pad of the detector up to
15): the extra Pad plane + larger-than-expected top-left (curated test uses TL=12,
exceeding the generator's randint(1,size-4)=<=11 bound) negated the saving. Reverted.

## Best achieved
**16.49 @ mem 4890 params 63, fresh 200/200** (single Slice-ch5 -> 2x2 count Conv ->
Greater -> fp16 stamp Conv -> uint8 label -> in-grid Where -> Pad -> Equal). Beats the
prior generalizing best (14.86) by +1.63. The stored single-Conv net scores 18.20 but
fresh=0.00 (overfit, does NOT generalize), so the real comparison is the generalizing
ladder, where this is the new best.

## Irreducible-floor analysis (v7, mem 4890)
Three ~900B planes dominate; everything else is tiny (<=450):
- g5 (900 fp32) — channel-5 corner Slice (15x15). The conv entry MUST be fp32 (Slice
  preserves the fp32 input dtype; casting to fp16 ADDS a 450 plane, no net win).
- cnt (900 fp32) — the 2x2 sum-Conv output (must be float; fp16 only via an added cast).
- L (900 uint8) — the 30x30 padded label plane feeding the FREE Equal output; uint8
  already halves it vs fp16, and the 30x30 shape is mandated by the output.
A generalizing net cannot drop below ~2700B (entry + count + 30x30 output label), so
~16.5 is the HARD floor for this rule. The stored 18.20 single-Conv is a non-
generalizing overfit (fresh 0.00); 18.50 is structurally unreachable while generalizing.

The data-dependent in-grid square (size 10 vs 15) is handled WITHOUT a crop: fix the
working canvas at 15x15 (always >= the in-grid square) and recover the in-grid mask as
rowany (x) colany from ReduceMax(input,[1,3]) / [1,2] (120B vectors, no 30x30 plane);
off-grid -> sentinel 200 via one Where, so off-canvas cells map to no colour channel.

## OPEN ANGLES (re-attack backlog — all marginal)
- Drop one 900-plane: would need to fuse the count-conv and stamp-conv (impossible, a
  Greater threshold must sit between them) or shrink the 30x30 output label (mandated
  by the output shape). No path found below ~2700B while generalizing.

## INSIGHT (transferable)
A "place fixed-colour markers at fixed offsets from a detected local shape" task is a
pure two-Conv pipeline: (1) a small all-ones Conv + threshold detects the shape's
anchor cell; (2) ONE weighted Conv whose kernel encodes each marker's COLOUR at its
relative OFFSET stamps all markers as distinct values in a single label plane (no
per-marker shift/add army). Disjoint markers + body never collide so the label is
just `body_colour*body + stamp`. opset-10 Equal needs int32 (float/fp16 both rejected
by shape-inference) so the final value plane must be cast to int32 before the FREE
output Equal — but the upstream arithmetic can all run in fp16 (half the bytes).

## S10 (2026-07-03) — knife-edge hardening ADOPTED (±0 pts, robustness)
Same phenomenon as task220 (logit grid {−.5,0,.5,1}, off-cells exactly 0.0; dirty-process
flip = 266/266 fail). Fix: subtract 0.25 from W[:,:,1,1] center taps (min_on 0.5) →
on ≥ +0.25 / off ≤ −0.25. mem 0 / params 900 UNCHANGED. Gates: fresh-process 0-fail,
dirty-process 0-fail (incumbent positive control 266-fail), fresh-2000 0/0.
Backup task230_pre_s10_knifeedge.onnx. See task220 S10 entry for the mechanism.
