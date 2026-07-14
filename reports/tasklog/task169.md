# task169 — 6e82a1ae

**Rule:** A 10x10 grid holds 4-6 small gray (colour 5) sprites placed with bounding-box
spacing>=1 (measured min chebyshev distance between gray cells of DIFFERENT sprites = 2,
so sprites are 8-connected-isolated). Each sprite has 2, 3 or 4 gray pixels; every pixel
of a sprite is recoloured `5 - count` (count2->3, count3->2, count4->1). Sprite chebyshev
diameter <= 3.
**Current (prior):** 15.16 pts (blank-note "confirmed-infeasible", no documented reason)
**Target tier:** B/A — closed-form, no flood-fill; the "per-component pixel count" looked
like a connectivity wall but the >=2 gap makes all small-radius window ops leak-free.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | leak-free multi-res mass features + Where one-hot chain, fp32, 3 iters | B | 10300 | 70 | 15.75 | 200/200 | win +0.59 |
| 2 | single colour-index plane + Pad(99 sentinel) + Equal->bool output; fp16 working planes | A | 5500 | 44 | 16.38 | 200/200 | win +1.22 |
| 3 | drop maxpool broadcast 3->2 iters (verified 0/3000) | A | 4700 | 44 | 16.54 | 200/200 | win +1.38 |

## Method
Sprites are isolated by a >=2 chebyshev gap, so ANY 3x3-window op on a gray cell touches
only that sprite -> leak-free. Build leak-free multi-resolution local "mass" features by
repeated gray-gated 3x3 sum-conv:
  F1 = (3x3-sum of gray)*gray   (= degree+1)
  F2 = (3x3-sum of F1)*gray     (radius-2 mass, still leak-free)
Broadcast each feature's per-sprite MAX to every sprite cell with 2 gray-gated MaxPool3x3
iterations (the max sits at a sprite-central cell, so all cells are within 2 hops). The
pair (M1=maxF1, M2=maxF2) separates the three counts EXACTLY:
  count2 <=> M1==2 ; count4 <=> M1>=4 OR M2==8 (M2==8 is the I-tetromino) ; count3 else.
A single colour-index plane is built with a 3-step uint8 Where chain, padded 10x10->30x30
with sentinel 99 (off-grid matches no channel -> all-zero, exactly like the benchmark),
then `Equal(idx30, arange)` emits the [1,10,30,30] BOOL output for free.

## Best achieved
16.54 @ mem 4700 params 44 — adopted? N (build-only). Beats prior 15.16 by +1.38. Y.

## Irreducible-floor analysis
Dominant intermediate: idx30 [1,1,30,30] uint8 = 900B (the entry to the free Equal output;
cannot shrink below 900B since the output op needs a 30x30 operand, and Equal-on-10x10 then
Pad is bigger + Pad rejects bool). Remaining ~3000B is ~15 fp16 [1,1,10,10] working planes.

## OPEN ANGLES (re-attack backlog)
- Broadcast the per-sprite SUM of F1 (cleanly disjoint {4},{7,9},{10,14,16}) instead of two
  MAX features -> would need only ONE broadcast feature + simple thresholds, dropping the
  whole M2 chain (~5 planes, ~1000B). Blocked only by lacking a cheap exact sum-broadcast
  for an isolated small component; a directed single-pass accumulation to a representative
  cell could work and would push toward ~17.
- Could fold the gray-gate Mul into the Conv/MaxPool (kernel already zero outside) to save a
  couple of planes.

## INSIGHT (transferable)
⭐ "Per-connected-component pixel COUNT -> recolour" is NOT a flood-fill wall when the
generator spaces components with a >=2 chebyshev gap: every bounded-radius window op is then
leak-free, so component-discriminating features come from REPEATED gray-gated 3x3 sum-conv
(multi-resolution mass) whose per-component MAX (broadcast by a few gray-gated MaxPool3x3
hops, count = sprite diameter) fingerprints the size. A single MAX feature can fail to
separate long-thin vs compact same-radius shapes (I-tetromino vs straight tromino both peak
at 3) — add a second (higher-radius) MAX feature to break that tie. Final colour-index plane
-> Pad with a NON-MATCHING sentinel (99) off-grid -> Equal(arange) gives the one-hot output
free, AND correctly emits all-zero off-grid (the benchmark has no channel-0 padding off the
active grid).
