# task341 — d6ad076f

**Rule:** Two solid colour blocks. A SHORT block (colour c0, thickness 2-4, length 4-6) and a LONG block / bridge (colour c1, length 6-9); the short block's column span is strictly NESTED inside the long block's span. A vertical CYAN(8) bar fills the gap *rows* between the two blocks, spanning the short block's columns shrunk by one cell on each side (its interior). `apply_gravity` then transposes/flips the whole figure into one of 4 cardinal orientations, so the two blocks may be stacked vertically or horizontally, in either order. INPUT = the two blocks; OUTPUT = input + the cyan bridge (cyan only overwrites background).
**Current (public):** 15.37 pts, ext:biohack_new
**Target tier:** detection (gap-fill between two blocks, 4 orientations) — reformulated to a clean reduction-based label-map (B-ish). Not separable globally (gap axis varies per instance), but per-instance it is rowmask⊗colmask, recoverable from 1-D occupancy + band reductions.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | reduction label-map: detect gap axis via occupancy hole; cyan rect = gap span (gap axis) × nested-intersection interior (perp axis); L=V+8·cyan; final Equal | B/detection | 10109 | 62 | 15.77 | 200/200 | WIN (+0.40) |
| 2 | RE-GOLF: all working 10×10/scalar planes → fp16 (0..9 + ±1e4 integer-exact); collapse colour-index plane to **uint8 Vu8** (serves mask `Vu8>0` AND label value); fold cyan value via ONE uint8 `Where(cyanB, 8, Vu8)` → pad-ready Lc (kills fp32 cyan/cyanv/Lcol + Mul + Cast) | B/detection | 7423 | 63 | 16.08 | 200/200 | WIN (+0.31) |

## Best achieved
16.08 @ mem 7423 params 63 — adopted? N (agent writes file only). Beats prior 15.77? **Y (+0.31)**.

## Irreducible-floor analysis (after RE-GOLF, mem 7423)
Dominant survivors: **V32 [1,1,30,30] fp32 = 3600 B** (colour-index Conv entry, forced fp32 by the fp32 one-hot input) + **V [1,1,10,10] fp32 = 400 B** (the Pad-crop bridge of V32 — Pad inherits the fp32 input dtype, so the cropped colour-index plane is unavoidably fp32 before its single Cast→uint8). Together V32+V = 4000 B is the colour-index ENTRY floor: cropping the *input* to 10×10 first costs 4000 B (10 ch × 100 fp32), and casting V32→uint8 *before* cropping costs a 900 B uint8 30×30 plane — both strictly worse. **L [1,1,30,30] uint8 = 900 B** is the final-Equal pad floor (output is fixed 30×30, label needs 0..9+sentinel ⇒ uint8 minimal). The 5 mask planes (M, Mabove/Mbelow/Mleft/Mright) are pinned at fp16 200 B each — they feed ReduceMax which rejects uint8/bool. Net floor ≈ 4000 (entry) + 900 (pad) + 1000 (masks) + ~900 (scalar/profile fp16) ≈ 6.8 kB ⇒ ceiling ~16.2.

## OPEN ANGLES (re-attack backlog, post-RE-GOLF)
- DONE: fp16 working planes + uint8 colour-index collapse + cyan-value Where fold (mem 10109→7423, +0.31).
- Eliminate the V [1,1,10,10] fp32 crop bridge (400 B): would need the colour index to be born narrow at 10×10. Pad/Slice inherit fp32 from V32; no cheaper path found (every alternative ≥ +500 B). The only escape is killing V32 itself, which requires an fp16 Conv ⇒ 18 kB fp16 input cast. Floor.
- Eliminate the 4 band-product [1,1,10,10] fp16 planes (Mabove/Mbelow/Mleft/Mright, 800 B): the band split is data-dependent (gap row at runtime) so it can't fold into a fixed row/col-sum Conv. A masked-min/max via index arithmetic on the per-line occupancy restricted to the band could in principle avoid the Mul, but the band membership is itself a runtime gate ⇒ still a 10×10 product. Likely marginal.
- Per-cell single-Conv (Tier S) is blocked: cyan width = *nested-interior* of one block, non-local (needs both spans + gap), no local hyperplane recovers it.

## INSIGHT (transferable)
`apply_gravity` in ARC-GEN is NOT physical gravity — it is a transpose/reflection applied IDENTICALLY to input and output, so the input→output rule is orientation-equivariant; handle it by computing BOTH axis branches and selecting via "which axis has the occupancy hole". ⭐ For two-block gap-fill tasks: the gap axis is the one whose 1-D occupancy has an internal empty run ((extent_len) > (#occupied lines)); the perpendicular cyan span is the nested intersection of the two flanking bands' extents (max-of-mins, min-of-maxes), shrunk by 1 — recoverable purely from ReduceMax/ReduceMin/Where on band-masked occupancy, no per-channel block identification needed.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 10. Flagged `bridge30` bool [30,30] 900B is a Where cond. A signed-einsum re-route WAS built and gates clean (bundled 266/266, fresh 2500 0-div) but prices **1491 vs incumbent 1429 (+62)**: the copy-term mixer [2,10,10]=200 params + Rt/Ct [2,30] 480B exceed the 900B saved. FLOOR — recorded explicitly so nobody rebuilds it. Candidate kept at scratchpad crop_b/cand341_trim.py, NOT adopted.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
