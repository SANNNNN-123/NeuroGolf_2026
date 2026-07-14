# task271 — ae4f1146

**Rule:** A 9x9 black grid holds four NON-overlapping 3x3 cyan(8) boxes; each box
carries some blue(1) pixels at local positions. The generator samples 4 distinct
blue-pixel counts from `range(9)`, sorts them ascending, and assigns them to boxes
0..3 — so the four boxes carry strictly different blue counts and box 3 (the last)
has the MOST. The output is the 3x3 content of the box with the most blue pixels:
output[r][c] = blue(1) where that box has a blue pixel, else cyan(8). Winner is
unique (0 ties over 2000 fresh samples). Grid is always 9x9.

**Current (prior):** ~14.73 pts, tier B label-map.
**Target tier:** A/B — argmax-select-among-4-boxes then crop a separable 3x3
window. Output color is per-cell deterministic given the winning box, so it lands
in the label-map family but with a tiny working canvas (9x9/7x7), beating B.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 3x3 sum-convs (blue/occ) over 9x9 → score=blue·isbox → ReduceMax argmax → Gather 3x3 window → label-map Equal | A/B | 2880 | 70 | 17.01 | 500/500 | WIN |

## Best achieved
17.01 @ mem 2880 params 70 — adopted? N (orchestrator gates). Beats prior 14.73? Y (+2.28).

## Irreducible-floor analysis
Dominant intermediates: `L` = the 30x30 uint8 label plane (900B) — unavoidable
because the output must broadcast to [1,10,30,30] and the smallest per-cell label
dtype is uint8; and the two fp32 9x9 channel slices `blue9`/`cyan9` (324B each =
648B) — Slice preserves the fp32 input dtype so the fp32 footprint is paid before
the fp16 Cast. The rest is 7x7/3x3 fp16 + scalar argmax machinery (~600B). No
[1,10,*,*] plane is ever materialised; the 10-way expansion lands in the FREE bool
`output` via the final Equal.

## OPEN ANGLES (re-attack backlog)
- The two fp32 9x9 slices could collapse to one if box-detection used ch0
  (occ = 1 − ch0_9x9) instead of ch8 — but blue (ch1) is still a second slice, so
  no net win. A single 1×1 colour-index Conv would emit a [1,1,30,30]=1800B fp16
  plane (worse than 648B of two 9x9 slices), so that direction loses.
- L=900B is the genuine floor for a 30x30 uint8 label; only a single-Conv (tier S)
  reformulation would remove it, but the output color depends on a data-dependent
  box SELECTION (argmax over 4 positions) + a crop, which no fixed Conv can route.

## INSIGHT (transferable)
"Emit the 3x3 box with the most X pixels" = (a) a 3x3 all-ones sum-Conv over a
small slice gives per-top-left counts AND a box-validity count in one pass;
(b) score = count gated by (occ-conv==9) via Where; (c) UNIQUE-argmax position
recovers as scalar (minrow,mincol) = ReduceMax(iswin·rowramp) / ReduceMax(iswin·
colramp) — no NonZero/argmax-op needed; (d) data-dependent crop = Add scalar
offset to a [0,1,2] index const, chained Gather(axis=2)·Gather(axis=3). Whole
select-and-crop pipeline stays at ~9x9/7x7 fp16, well under the 30x30 label floor.
