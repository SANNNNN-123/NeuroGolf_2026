# task308 — c8cbb738

**Rule:** A large bg grid holds several scattered "broken-border" elements. Each
element draws non-bg pixels in 4-fold (corner elements) or 8-fold (the on-axis
element) symmetry about its OWN centre; each element uses a DISTINCT colour and
sits on the perimeter of a Chebyshev square of radius h (= halfsize). The OUTPUT
is the single (2h+1)×(2h+1) square all the fragments reconstruct: bg everywhere
except colour c at the symmetric positions of its offset about centre (h,h).
Because each colour belongs to exactly ONE centre, the GLOBAL bbox of colour c
gives its offset magnitudes directly: ar=(rmax-rmin)/2, ac=(cmax-cmin)/2,
h=max_c max(ar,ac). Corner element → coloured at (h±ar,h±ac); on-axis element →
plus-shape at (h,0),(h,2h),(0,h),(2h,h). The two collide at ar==ac==h and are
disambiguated by the count of colour c in its top bbox row (corner=2, axis=1).
**Current:** 15.01 pts, gen:thbdh6332, mem 21698, params 101.
**Target tier:** detection/closed-form rebuild — no flood-fill, fully per-colour
bbox + separable ring pattern; same algorithm as the public net but far leaner.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | port public algo, fp16 downstream, bg=corner | A | 25178 | 116 | 0.0 | 199/200 | bg≠corner (coloured pixel can land at grid [0][0]); h=1 fails |
| 2 | bg = argmax channel count (most-frequent) | A | 25156 | 120 | 14.86 | 200/200 | correct but heavier than public |
| 3 | fp16 all full-canvas planes; one fp32 entry/axis | A | 17400 | 130 | 15.23 | 200/200 | halved the [1,10,30,*] planes |
| 4 | uint8 Lpad gateway; drop bg-mask planes (mask at scalar) | A | 16549 | 120 | 15.28 | 200/200 | bool Where(9) NOT_IMPLEMENTED → kept And/Or select |
| 5 | colour+1 packed into ONE fp16 [1,10,7,7]; scalar bg mask | A | 14409 | 121 | 15.42 | 500/500 | ADOPT |

## Best achieved
**15.42** @ mem 14409 params 121 — beats prior 15.01 by **+0.41** (≥+0.3). fresh 500/500.

## Irreducible-floor analysis
Two fp32 entry reductions dominate: `row_cnt_f32` = ReduceSum(input,axis=W) and
`cpres_f32` = ReduceMax(input,axis=H), each [1,10,30,1] = 1200B. They reduce the
fp32 input so ORT emits fp32 (cannot pre-cast the 10-ch input — that's 18000B).
Everything downstream is cast to fp16 (presence/min/max planes 600B) or kept bool.
The colour-index gateway is a uint8 [1,1,30,30] Lpad (900B) feeding the FREE bool
Equal output. The 7×7 ring lives entirely in fp16/bool [1,10,7,7] planes (~490–980B).

## OPEN ANGLES (re-attack backlog)
- The 6 fp16 [1,10,30,*] presence/min/max planes (~3600B) could shrink if the
  active grid extent (≤ ~5h+5 ≤ 20) were sliced before reduction — but the slice
  is data-dependent (variable grid size) → symbolic-dim trap. Static 30 stands.
- The 8 [1,10,7,7] diamond/corner bool planes (~3900B): the diamond pattern's
  three planes (dia_v,dia_h,dia_pat) are forced full by the [1,10,7,1]⊗[1,1,1,7]
  broadcast. A single banded-equality plane (pack |dr|==ar, |dc|==ac, dr==0, dc==0
  into one additive code read by thresholds) might fold these to 1–2 planes.
- rect-vs-diamond disambiguation still costs a [1,10,30,1] top-row-count chain;
  a (rmin,cmin)-corner-presence test could replace it but needs a 2-D gather.

## INSIGHT (transferable)
- ⭐ bg is the ARGMAX of per-channel pixel count, NOT the grid corner cell: a
  foreground pixel can legally land at input[0][0] (centres range over the whole
  interior), so corner-cell bg detection silently fails on small-h instances.
- ⭐ "argmin/argmax over K candidates that each carry a label, exactly one hits a
  cell" → pack label+1 into ONE additive plane: `Where(mask, colour+1, 0)` then a
  channel ReduceSum gives `colour+1` at hit cells / 0 elsewhere — recovers BOTH
  "any hit" (sum>0) and the label (sum-1) from a single [1,10,K,K] fp16 plane,
  killing the separate mask-cast + colour-Mul planes.
- ⭐ Exclude the bg channel's spurious full-grid bbox at the SCALAR [1,10,1,1]
  level (Where(not_bg, rh, -1)) instead of masking the full [1,10,30,*] count
  plane — drops two full-canvas planes for free.
- Pay ONE fp32 entry reduction per axis, then Cast→fp16 and run every downstream
  full-canvas op in fp16 (ORT ReduceMin/Max accept fp16); uint8 for the Pad
  gateway (Pad rejects bool but takes uint8 = half of fp16). ORT Where is NOT
  implemented for BOOL data branches under ORT_DISABLE_ALL — keep And/Or selects.

## 2026-07-05 — transfer probe follow-up

- `topk_k=4 -> 3` capacity shrink was tested as a task025-style relaxed-gate
  lever.  It is not a rare-case trade here: stored eval fails 129/266, so the
  fourth active colour/element slot is structurally required.
- Rechecked the final label expansion pattern.  This task correctly uses
  `Equal(labels_grid, channel_ids) -> Pad(output)` because the compact board is
  only 7x7: one-hot-before-pad costs `9*49=441B`, cheaper than padding a label
  plane to 30x30 (`900B`).  This is the useful small-board side of the
  label-pad ordering rule.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
