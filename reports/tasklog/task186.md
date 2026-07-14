# task186 — 794b24be

**Rule:** A 3x3 grid carries `count` (1..4, `randint(1,4)`) blue (colour 1) pixels at
random positions. Output is a 3x3 grid whose red (colour 2) cells form a FIXED
count-thermometer: out[0][0] red if count>=1, out[0][1] if count>=2, out[0][2] if
count>=3, out[1][1] if count>=4; every other cell black (colour 0). The ENTIRE
output is determined by ONE scalar (the blue count) => 4 possible outputs.
**Current (public):** 19.11 pts, Slice-blue→ReduceSum→Gather(red_bank[4,3,3])→Concat 10ch fp16→Pad, mem 314, params 46
**Target tier:** COUNT->FIXED-PATTERN (cheapest tier) — output is a pure function of one scalar count.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full 3ch bank[4,3,3,3] uint8, Gather→Pad | count→pattern | 79 | 127 | 19.67 | — | works, +0.56 |
| 2 | red-only bank[4,1,3,3], ch0=Equal(red,0), ch1=zeros, Concat 3ch uint8→Pad | count→pattern | 106 | 64 | 19.86 | — | works, +0.75 |
| 3 | #2 + int32 idx (was int64) | count→pattern | 102 | 64 | 19.89 | 200/200 | ADOPTED |

## Best achieved
19.888 @ mem 102 params 64 (total 166) — beats prior 19.11 by +0.78. Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate = the 36B fp32 blue Slice (channel-1 over the 3x3 grid). ORT
Slice preserves fp32 and ReduceSum rejects uint8/bool, so the count entry must pay
fp32 for the 9 active cells (36B) — this is the genuine entry floor for the count.
The Concat 27B (leading 3 channels of the small output) is the next item: ch0
(black) must be a NONZERO computed plane (= Equal(red,0)) because Pad can only
zero-fill, and the output needs ch0=1 at every black cell; ch1 (blue) zeros + ch2
(red) complete the 3-channel block that Pad expands into the FREE 30x30 output.
A full 3-channel bank (attempt 1) removes the Concat/Equal/Cast (mem 79) but costs
108 bank params (total 187 > 166), so the red-only bank + computed ch0 wins.

## OPEN ANGLES (re-attack backlog)
- The two ch0 planes (bool Equal 9B + uint8 Cast 9B = 18B) are the only obvious
  remaining fat; no uint8 Sub under ORT, and Concat/Pad reject bool, so a single-op
  ch0 was not found. A bank holding [ch0,red] (72 params) needs slices to re-insert
  ch1 and came out worse. Marginal headroom (~0.06 pts) below current.

## INSIGHT (transferable)
⭐ For COUNT->FIXED-PATTERN where the output has ONLY black + one fixed-colour
pattern, store a RED-ONLY bank[K,1,H,W] (not the full 10-ch or 3-ch bank) and
derive the black channel as `Equal(red,0)` — trades bank params for one small
Concat and a 9B bool plane, which is a net win because params and mem are weighted
equally in the score and the bank shrinks K× per dropped channel. Also: a uint8
Pad zero-fills BOTH trailing colour channels and the 30x30 spatial border in one op
(output is FREE). idx via int32 (not int64) Cast saves 4B on the Gather index.
