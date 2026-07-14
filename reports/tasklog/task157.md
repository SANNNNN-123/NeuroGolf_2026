# task157 — 6a1e5592

**Rule:** 10×15 grid. Rows 0-2 are a red(2) bar. 3-4 "creatures" (contiguous black creatures
from `common.continuous_creature`, ≤4×4, often NON-rectangular). Each creature appears TWICE: (1) a
BLUE placement near the top at `(bluerow∈{1,2} + lr, bluecol + lc)` — but in the INPUT only the part
landing in rows 0-2 is visible (carved as black(0) into the red bar; the body below row 2 is NOT
drawn in the input); (2) a GRAY(5) placement bottom-anchored (bottom row = 9) at a DIFFERENT random
`graycol`, showing the FULL shape. OUTPUT = redraw each FULL creature in BLUE(1) at its blue location
(top visible carve extended downward by the full body), red bar kept, gray removed. The generator
enforces footprint-uniqueness so the blue↔gray correspondence is well-defined.
**Current (stored):** 13.76 pts, gen:wguesdon6315 (imported overfit, 75331 params), mem 75331.
**Target tier:** detection / NONE feasible — combinatorial shape-correspondence wall.

## Attempts (reference solvers in numpy, to confirm rule; none ONNX-feasible)
| # | angle | result |
|---|---|---|
| 1 | greedy per-gray-shape band match by width | 10/40 |
| 2 | exact-cover BACKTRACKING over (gray shape, bluerow, bluecol) placements | 30/30 ✓ (but needs backtracking — no feedforward equivalent) |
| 3 | greedy band match restricted to gray width | 25/40 |
| 4 | per-footprint-component normalized-shape match to gray top-portion | 38/40 (fails on merged/colliding footprints) |

## Why INFEASIBLE for feedforward ONNX (banned: Loop/Scan/NonZero/Unique/Compress)
The transform is a DATA-DEPENDENT ASSIGNMENT between two spatially-separated regions:
- The full body of each blue creature exists ONLY in the gray region, at an INDEPENDENT random column
  (bluecol ⊥ graycol). So the body cannot be reconstructed locally from the top footprint.
- The shapes are ARBITRARY contiguous creatures (60/128 non-rectangular), so there is no rectangle /
  local run-length shortcut (the lever that made task368 feasible — there every sprite shared ONE
  global pattern revealed once; here every creature is a distinct shape needing its own match).
- Correct reconstruction needs: connected-components of gray AND of the footprint, normalized-shape
  matching of arbitrary patterns across the two regions, and a one-to-one assignment. The only 100%
  reference needs exact-cover backtracking. Local greedy heuristics top out ~38/40 — exactly where the
  75k-param overfit base also lands (fresh 39/40), i.e. real-LB ≈ 0.
- This reduces to a graph-matching / exact-cover join over data-dependent variable-size shapes, which
  has no feedforward closed form under the allowed op set. No separability, no banded-Conv, no
  count-rank, no Kronecker, no runtime-onehot-Conv collapse applies (the binding op is the JOIN).

## Base-net status
`fresh_pass(157, n=40)` = 39/40 even for the 75331-param imported overfit → fails ALL-pass
generalization → scores ~0 on the real Kaggle LB despite stored 13.76 (gap-closer premise confirmed).

## OPEN ANGLES (low-confidence, untried)
- Runtime-weight Conv using an extracted gray shape as a data-dependent kernel could template-match
  ONE shape, but there are 3-4 shapes at unknown positions and the 2D assignment across them is still
  a join — no evidence this closes to 100%.
- If a future op budget allowed Loop/Scan, a bounded flood + per-component match would be exact.

## INSIGHT (transferable)
⭐ Two-region shape-CORRESPONDENCE with INDEPENDENT random placement columns (body visible only in a
distant region, matched per arbitrary shape) is a genuine feedforward-ONNX WALL — distinct from the
task368 "many copies of ONE revealed pattern → local run-length lookup" pattern. Discriminator: if the
output cell's value requires identifying WHICH distant object corresponds to THIS location (a join),
and objects are arbitrary variable shapes, it's an exact-cover/assignment problem with no closed form
under banned Loop/Scan/NonZero. Heuristic matchers plateau ~95% (38/40), which equals what a huge
overfit net achieves and still yields real-LB 0 (all-pass requirement).

## S10 (2026-07-03) — bobmyers7186 candidate NOT adopted
The bobmyers7186 net was **not adopted**: task157's generator is **UNGATEABLE**
(crash/timeout in fresh gen) and this task carries a known **private-LB-fragile**
flag (heuristic matcher plateaus ~95%, real-LB 0 on all-pass). Candidate Δ was only
≈ −5B — not worth the private-LB risk. Do not re-probe this candidate.


## S15b (2026-07-06) — ADOPTED from franksunp 7233.12: 7243 -> 7024 (+0.031); gate inc/cand=61/61 (equal, safe). See [[neurogolf-urad-7225-bundle-vein]].