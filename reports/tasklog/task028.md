# task028 — 1bfc4729

**Rule:** size=10 grid. Input has exactly two coloured pixels: colour0 at row 2, colour1 at row 7 (=size-3); their COLUMN positions are irrelevant to the output. Output is a FIXED 10x10 label template — top half (rows 0..4) draws colour0 on the outer frame plus full rows 0 and 2; bottom half (rows 5..9) draws colour1 on the outer frame plus full rows 7 and 9. Only the two colours are data-dependent; the geometry is constant.
**Current:** 16.52 pts (public net)
**Target tier:** A/S — fixed template parametrised by 2 recovered scalar colours; route the 10-ch expansion into the FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Where-chain over 30x30 fp16 masks + 30x30 sentinel | A | 7972 | 2730 | 15.72 | (not run) | full-size const planes too heavy |
| 2 | build L on 10x10, Pad(99) to 30x30 fp16, Equal->bool out | A | 3372 | 239 | 16.81 | 200/200 | +0.29 marginal |
| 3 | same but Cast L10->uint8 then Pad uint8, Equal(uint8) | A | 2572 | 239 | 17.06 | 200/200 | ADOPT, +0.54 |

## Best achieved
17.06 @ mem 2572 params 239 — beats prior 16.52 by +0.54. fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate is the padded colour-index plane L [1,1,30,30] uint8 = 900B
(the final Equal must broadcast L across the full 30x30 output). uint8 halves it vs
fp16 (1800B). Next are the two row slices [1,10,1,10] fp32 = 400B each (need all 10
channels to recover the colour one-hot; can't narrow without losing the one-hot).
On-grid bg(0) maps to ch0; off-grid sentinel 99 matches no channel → all-zero output
(correct: off-grid output cells carry no colour).

## OPEN ANGLES
- Combine the two row slices into one strided Slice (rows 2,7 step 5) — same 800B, no win.
- Could the L plane be avoided entirely via a separable row⊗col routing? The frame template
  is NOT row⊗col separable (frame + interior full-rows), so a single colour-index plane is
  the cheaper route. Likely near floor for this template.

## INSIGHT (transferable)
For a FIXED template parametrised by k recovered scalar colours: build the colour-index
plane on the small ACTIVE canvas (10x10), Cast to uint8, then Pad(sentinel=99) to 30x30 —
uint8 Pad + uint8 Equal(arange) is exact and halves the dominant plane vs fp16 (900B vs
1800B). Use an off-grid sentinel > 9 so off-grid matches no channel and stays all-zero,
while on-grid bg stays 0 -> ch0. Recover per-position colours from single-row Slice +
ReduceMax over cols (ch0 weight 0 in the k-ramp neutralises the all-bg cells).
