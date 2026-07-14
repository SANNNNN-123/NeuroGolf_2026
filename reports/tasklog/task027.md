# task027 — 1b60fb0c

**Rule:** A small "creature" shape S is stamped FOUR times under the C4 rotation group about
a centre at index (9+off)/2 on each axis (off in {0,1}, a random single-pixel offset). Three
copies are drawn in blue(1) (the INPUT); the missing 4th copy is drawn in red(2) and ADDED to
make the OUTPUT. Since blue is exactly three of the four C4 copies, rotating blue by 90° about
the correct centre exposes the missing copy: **red = rot90_cen(blue) AND NOT blue**, cen=9+off.
The offset is recovered offset-free by picking the centre whose exposed copy is SMALLEST
(|rot90_cen(blue)\blue| minimal — one clean copy; the wrong centre scatters blue into a bigger
set). Grid is 10×10 at the top-left of the 30×30 canvas, so all work is on the 10×10 region.

**Current (prior stored):** 16.332 pts, ext:kojimar6275, mem 5777, params 37
**Target tier:** A — data-dependent rotation realised as reverse+transpose on a 10×10 plane;
the 10-ch expansion routes into the FREE Where output (no [1,10,H,W] intermediate).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | rot180 R-matrix MatMuls + max-overlap offset | A | 4206 | 229 | 16.60 | — | NON-EXACT (overlap pick fails ~2/10k) |
| 2 | rot90-closure asymmetry pick (2 branches) | A | 6226 | 51 | 16.26 | — | exact but heavy |
| 3 | rot90 via reverse+transpose, min-candsize pick | A | 4626 | 44 | 16.55 | 300/300 | exact |
| 4 | + drop Sub (Greater(g,blue)) + uint8 select | A | 4126 | 44 | **16.664** | 200/200×4 | **best** |

## Best achieved
16.664 @ mem 4126 params 44 — adopted? N (orchestrator gates). Beats prior 16.332? **Y, +0.332**.

## Irreducible-floor analysis
Dominant intermediate is the output stage: the [1,1,30,30] uint8 Pad carrier (900B) + the
[1,1,30,30] bool Where cond (900B) = 1800B. This is the structural floor for producing a
30×30 boolean mask to drive `Where(cond, red_onehot[1,10,1,1], input)` — Pad rejects bool so a
uint8 (smallest non-bool, 1B/elem) carrier is required, and Where requires a bool cond; neither
900B tensor is removable. Next is the 10×10 blue entry: f32 channel-1 slice (400B, forced by the
f32 input dtype) + its fp16 cast (200B). The two rot90 candidate branches (cb/cf/cu at 10×10)
add ~1000B.

## OPEN ANGLES (re-attack backlog)
- The two cand branches duplicate ~1000B. g10 is g9 shifted one column, so cand10 could be
  derived from cand9 by the same shift instead of an independent Greater/Cast chain (~300B?).
- Compute both candidate sizes from a single stacked [1,2,10,10] tensor (one ReduceSum) to halve
  the per-branch fp16 size planes.
- The 400B f32 blue slice is the canvas-entry tax; no way seen to read a fp16 blue plane directly
  (Slice preserves the f32 input dtype).

## INSIGHT (transferable)
⭐ For "complete the C4/Cn rotational symmetry" tasks where the input is k-of-n copies: the
missing copy = **rot(blue) AND NOT blue** for the correct rotation step, and the data-dependent
symmetry CENTRE (here a ±1px offset) is recovered offset-free by **minimising the size of the
newly-exposed set** over the candidate centres — the true centre yields exactly one clean copy
while a wrong centre scatters into a larger set (exact on 40k samples; a max-self-overlap or
rot180-candsize pick is NOT exact and silently fails genverify). Also: a 90° rotation needs NO
matrix — `rot90 = transpose(reverse_rows(x))` via a negative-step Slice + Transpose (0 params),
and the neighbouring-centre rotation is just a 1-column shift (Pad-left + Slice), avoiding a
second MatMul. Route the colour insert through `Where(cond30, color_onehot[1,10,1,1], input)`.
