# task076 — 36d67576

**Rule:** Several "rainbow sprites" (3-4) are stamped at random positions with random
rotations (∈{identity, rot90, rot180, horizontal-flip}) on a 13-15 grid. A sprite =
5 diagonally-connected YELLOW(4) pixels + 1 RED(2) + 1-2 BLUE(1) + 1-3 GREEN(3), each
non-yellow pixel placed orthogonally adjacent to a yellow pixel. In the INPUT only the
FIRST sprite is drawn fully; all OTHER sprites show only their yellow+red pixels.
In the OUTPUT every sprite is drawn fully. So: reconstruct the hidden blue/green pixels
of every partially-shown sprite by matching its (rotated) yellow+red anchor pattern to
the one fully-shown reference sprite and stamping the reference's blue/green offsets.

**Current:** 13.71 pts, 1330-node exact solver (TopK/ScatterND/ArgMax + heavy scalar
arithmetic: 222 Add, 154 Mul, 110 Min, 8 Mod/Div/Floor), mem ≈ 77,700, params 2115.
Passes 60/60 fresh — it is a correct, already heavily-golfed exact solver.

**Target tier:** detection/correspondence — the highest admissible tier is whatever the
exact per-sprite orientation-match + template-stamp costs; no closed-form collapse exists.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | local YELLOW-window → color (rad 2/3) | — | — | — | — | — | FAIL: 246/688 windows ambiguous |
| 2 | local YELLOW+RED-window → color (rad 2/3/4) | — | — | — | — | — | FAIL: ~48% of added pixels ambiguous at rad2 |
| 3 | mem-lever on existing net (route final one-hot to free output) | — | — | — | — | — | N/A: final Equal[1,10,30,30] IS already the free graph output (0 counted); no easy plane to drop |

## Best achieved
None > 13.71. No leaner correct net produced.

## Irreducible-floor analysis
The classification "what color/where to add" is provably NON-LOCAL: identical local
yellow+red neighborhoods (radius 2-4) map to different output colors depending on the
GLOBAL 5-pixel blob arrangement and sprite orientation (~48% of added blue/green pixels
are locally ambiguous). So no single fixed conv/separable closed form can decide the
output — you must (a) detect each of 3-4 sprites' orientation by matching its whole
yellow+red anchor against 4 rotations of the data-dependent reference blob, then
(b) stamp the reference's blue/green offsets. Both steps are over a per-instance-random
shape at per-instance-random positions, forcing many full-canvas correlation/stamp
planes. The existing net already pushes this onto scalar arithmetic (TopK positions,
ScatterND stamps, Mod/Div index math) to avoid full planes wherever possible; its
~77.7KB is distributed across ~14 medium fp16/fp32 working planes, not one removable
dominant plane. The 9000B Equal is the FREE output (not counted). Beating +0.3 needs a
~26% cut of an algorithm with no remaining single big lever.

## OPEN ANGLES (low-confidence)
- Per-channel grouped-Conv correlation of input yellow+red against the 4 rotated
  reference kernels to get a per-position orientation argmax, then a single ScatterND
  stamp of a rotation-indexed reference-offset table — MIGHT beat the existing TopK
  pipeline if the orientation map collapses to one fp16 plane, but the reference kernel
  is runtime/data-dependent and the 4-rotation stack + stamp still needs multiple
  full planes; unlikely to clear 26% under the already-golfed baseline.

## INSIGHT (transferable)
⭐ "Reveal hidden sprite cells via rotated copies of one fully-shown reference" is a
genuine shape-correspondence WALL: the per-cell answer is NON-LOCAL (same local
yellow+red window → different colors by global orientation, measured ~48% ambiguous),
so no closed-form local conv collapses it. When a re-probe target's existing net is a
1000+ node exact solver that already passes fresh and whose memory is spread across
many medium planes (no single dominant removable plane, and the final one-hot is already
the free output), the +0.3 lever almost certainly does not exist — re-confirm WALL fast.

## S10 (2026-07-03) — scout re-confirm: correspondence WALL, FLOOR
Full seam analysis vs playbook 14: flood already minimal uint8 MaxPool×3; detection already
minimal dilated-Conv read; bulk cost = distributed scalar TopK-position arithmetic (588B
candidate planes ×~12) — nothing to contract against free input; rule provably non-local
(~48% radius-2-4 ambiguous). No single ≥300B seam. Do not re-scout without new mechanism.

## S17 (2026-07-06) — WIN +0.0384: board_code chain fp16→uint8 recast
- Mechanism: board_code_f (Conv output) proven exact integers 0..5 on bundled+arc-gen. Recast
  the color-code carrier chain from fp16 to uint8 (1 byte): board_code Cast to=2; Equal consts
  → uint8 (added one/two/three/four/five_u8, pruned now-dead one/two/three/five_f + color4_f16);
  source_hidden_colors / hidden_update_code Where sentinels → zero_u8; source_red_i32 Cast to=2.
  ORT has no int8 Where kernel → used uint8 (values ≥0). Halved board_code(450→225) +
  board_code_flat(450→225) + candidate_board_code(252→126) planes.
- Before: mem 16254 / params 303 / pts 15.28544. After: mem 15629 / params 303 / pts 15.32392.
- Gate: bundled fail=0; fresh-gate 2700 instances (rebuilt vs incumbent) div=0 (exact recast).
  candidate_safe_indices (int32 Gather idx 0..224) stays int32 = FLOOR; board_code_f Conv f32 = FLOOR (input-tied).
