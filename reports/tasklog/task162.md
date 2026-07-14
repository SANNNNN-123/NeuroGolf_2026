# task162 — 6cf79266

**Rule:** Fixed 20×20 grid (one foreground colour K∈{2..9} over a dense black
static background, with 1–3 black 3×3 cutouts). The generator scans interior
cells (r,c∈[1,18]) in ROW-MAJOR order; whenever the 3×3 window centred at (r,c)
is currently all-black IN THE OUTPUT it paints that whole 3×3 block blue(=ch1).
Because it edits in place, an earlier-scanned all-black window that OVERLAPS a
later one (centres within Chebyshev≤2) suppresses the later fill. Closed form:
cand=all-9-black 3×3; fire = cand ∧ ¬(any row-major-EARLIER overlapping cand);
blue = dilate(fire, 3×3); out = Where(blue, ch1, input). Verified 0-mismatch vs
the in-place generator over 8000 fresh + all stored examples.

**Current (prior):** 13.94 pts, gen:thbdh6332 import, bloat ≈63291
**Target tier:** detection/B — 3×3 neighbourhood op needs a per-cell plane; the
win is cropping to the 20×20 active region + fp16/bool planes.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | parallel fill-all-black-3×3 (no causal blocker) | B | 7800 | 48 | — | — | FAIL stored ex2 (over-fills overlapping holes) |
| 2 | + causal-5×5 blocker conv (sequential→local) | B | 10600 | 73 | 15.72 | 200/200 | pass all |
| 3 | drop Not (Less-direct), keep u8 pad | B | 10200 | 73 | 15.76 | 200/200 | adopted |

## Best achieved
15.76 @ mem 10200 params 73 — beats prior 13.94 by **+1.82**. fresh 200/200.

## Irreducible-floor analysis
Dominant: the f32 ch0 slice `black32` [1,1,20,20]=1600B (Slice inherits f32) plus
~6 fp16 [1,1,20,20] conv/count planes (800B each) and the [1,1,30,30] pad-back
planes (u8 900B + bool 900B). Five neighbourhood convs (cand 3×3, blocker 5×5,
dilate 3×3) each force a per-cell plane; cropping to 20×20 already halves them
vs 30×30. Cannot drop the f32 slice (occupancy read), the cand-binary f16
(blocker conv needs a thresholded float input), or the 30×30 cond (Where must
broadcast to the full canvas).

## OPEN ANGLES (re-attack backlog)
- Fuse cand-threshold + blocker into one signed conv (e.g. `9·black_conv −
  causal_conv` banded) to drop the separate `cand` f16 plane (~800B → ~16.0).
- Replace the u8→pad→bool round-trip (2200B) with a direct bool dilation if a
  future ORT build pads bool.

## INSIGHT (transferable)
⭐ An IN-PLACE row-major generator scan with overlapping stamps is NOT a parallel
conv: overlapping all-X windows let the row-major-FIRST one suppress later ones.
This collapses to a PURELY-LOCAL closed form — `fire = cand ∧ ¬(causal-half
neighbourhood conv of cand)` — because the suppressing earlier window always
itself fires (no chains form when stamps are equal-size & the relation is
"overlap"). The causal kernel = the row-major-earlier offsets within the overlap
radius (here 12 of the 5×5). ⚠️ A 5000-sample fresh check can show ZERO
parallel-vs-sequential mismatches yet the rare overlap STILL appears in a stored
example → always reproduce the generator's in-place scan exactly, don't trust a
parallel approximation just because fresh-N agrees.

## S11 (2026-07-03) — FLOOR CONFIRMED at 4068B; both dossier levers refuted by measurement
- Lever 1 (occupancy fp32→u8): REFUTED — Slice preserves dtype (channel-0 crop = mandatory
  1600B fp32; Cast-before-Slice = 9000B input copy). Occupancy floor = 2000B.
- Lever 2 (block30 900B einsum-routed away): REFUTED — output COPIES the input field, so the
  final einsum needs the fp32 free input as operand → fire-mask forced fp32 [.,20,20]=1600B
  > the 900B bool it replaces. Where(block30, blue, input) is the routing floor.
- Key mechanism insight: incumbent's fused 4x4 QLinearConv det_weight (8/9-core + −1 L-border,
  rank 3) does detection AND row-major causal suppression in ONE 324B u8 plane. A free-input
  einsum is rank-1 separable → computes detection but provably cannot express the rank-3
  suppression; separate suppression ops cost ~4550B total (> incumbent).
- ⚠️ Negative artifact worth keeping: reports/candidates/task162_signed.py is bit-identical
  to the incumbent on 2000 fresh (0 div) and 136B cheaper, but fails original-ARC bundled
  train#2 (real H+V hole overlap — violates the generator's unambiguity filter). Public-LB
  fatal (bundled=LB), private-safe. Do NOT adopt under current rules.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k5(cost 4068): val gate fail (bundled train#2가 generator guarantee 위반 — S11, public-fatal anyway). 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
