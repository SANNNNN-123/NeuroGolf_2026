# task287 — b8825c91

**Rule:** Grid is always 16x16 (2*size, size=8) and entirely in-canvas. The true
pattern has full D2 4-fold mirror symmetry (c->15-c AND r->15-r). The input
overwrites a few small (<=4x4, exactly 2) rectangular regions in the top-left 8x8
quadrant with yellow(4); the underlying pattern never contains yellow. Output
restores every cell to the unique non-yellow colour shared by its four symmetric
partners. Every orbit has 3 never-cut members (cutouts live in one quadrant), so
the cell recovers as the max over the orbit with yellow mapped to a low sentinel.

**Current (prior):** 15.61 pts, mem 11924, params 73 (the imported public net;
its src/custom draft mirrored it exactly: fp32 conv + fp32 slice + 4-way fp32 Max).
**Target tier:** A (orbit-reflect-gather into free output) — entry fp32 conv plane
(3600B) + 30x30 pad-back (900B) are the hard floor; everything else is uint8.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | original fp32 conv+slice+4xfp32 Max (prior draft) | A | 11924 | 73 | 15.61 | — | baseline |
| 2 | slice 10ch input to 16x16 BEFORE conv, uint8 folds | A | 14724 | 57 | 15.40 | — | WORSE: 10ch slice plane = 10240B |
| 3 | conv on full input, cast 30x30->uint8, slice; uint8 4-way orbit max | A | 7960 | 57 | 16.01 | — | +0.40 |
| 4 | + separable 2-fold max (assoc): s1=max(v,flipV); L=max(s1,flipH(s1)) | A | 7192 | 57 | 16.11 | 500/500 | ADOPTED, +0.50 |

## Best achieved
16.11 @ mem 7192 params 57 — adopted? Y (file only). Beats prior 15.61? Y (+0.50).
ISOLATED fresh 200/200 and 500/500.

## Irreducible-floor analysis
- vf30 (3600B): the 10->1 colour-index Conv MUST be fp32 over the full 30x30 free
  input. Slicing the 10-ch input first to crop the conv to 16x16 costs a 10240B
  slice plane (net-negative, attempt #2). Casting the 10-ch input to fp16 is 18000B.
  So 3600B is the entry floor.
- L (900B): the final label must be padded back to 30x30 (off-grid -> sentinel 10
  matches no channel) so the Equal writes the full free [1,10,30,30] bool output.
  uint8 Pad keeps it at 900B (vs fp16 1800B); opset-11 Pad accepts uint8.
- Middle planes all uint8 16x16 (256B each): v30(900 cast)+v(256)+vV+g1+s1+s1H+g2+L16.
  Two separable folds need only 2 Gathers + 2 maxes (vs 3+3 for the naive 4-way max).

## OPEN ANGLES (re-attack backlog)
- The v30 (30x30 uint8 cast, 900B) exists only to be sliced cheaply; if a future
  ORT allowed a uint8-producing crop directly off the conv we'd save it. Slicing
  fp32 then casting is worse (1280 vs 1156).
- vf30 + L (4500B combined) is ~63% of mem and both are structural; the remaining
  ~2700B of uint8 16x16 planes is near minimal for a 2-fold separable orbit max.

## INSIGHT (transferable)
- ⭐ SEPARABLE ORBIT MAX: a D2 4-fold mirror symmetrization needs only TWO folds,
  not a 4-way max — fold one axis (max(v,flipV)=s1), then fold the *result* on the
  other axis (max(s1,flipH(s1))). Saves one Gather + one max-pair (768B here) vs
  building all 4 partners. Generalizes to any associative orbit reduction over a
  product group.
- ⭐ CONV-THEN-CROP, NOT CROP-THEN-CONV: to crop a colour-index conv to an active
  KxK region, run the conv on the FULL free 10-ch input and slice the single-channel
  RESULT — never slice the 10-ch input first (that materializes a 10*K*K*4B plane,
  here 10240B, dwarfing the 3600B full-conv it was meant to shrink).
- uint8 elementwise max = Where(Greater(a,b),a,b) (256B/16x16 plane); ORT has no
  uint8 Max. Cast the fp32 conv to uint8 at 30x30 (900B) and slice uint8 (256B) —
  cheaper than slicing fp32 (1024B) then casting.
