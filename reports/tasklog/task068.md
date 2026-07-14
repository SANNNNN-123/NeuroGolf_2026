# task068 — 31aa019c

**Rule:** 10x10 grid with several distinct colours placed; index-0's colour gets EXACTLY one pixel,
every other colour gets >=2 (random_colors samples without replacement, so per-colour count ==
per-index count). Hence exactly ONE colour has count 1 = min_color. Output is blank except: for the
single min_color pixel at (r0,c0) (generated in [1,8]x[1,8] so the box never clips), stamp a 3x3 RED
(colour 2) box centred there and put min_color back in the centre cell.
**Current (public):** 15.56 pts, Relu-count net (full 30x30), mem ~higher
**Target tier:** B — output colour per cell is deterministic but needs a per-cell colour-index plane
(centre colour) + a 3x3 dilation; not separable (centre colour != border colour, data-dependent
channel) so a label-map + final Equal is the admissible top tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map @ 30x30 (no off-grid sentinel) | B | 12735 | 43 | 0 | — | FAIL: target leaves off-10x10 region all-zero; L=0 there lit ch0 |
| 2 | label-map, work@10x10, Pad L w/ sentinel 10 | B | 6035 | 58 | 16.29 | 200/200 | WIN +0.73 |

## Best achieved
16.29 @ mem 6035 params 58 — adopted? N (build-agent; main adopts). Beats prior 15.56? Y (+0.73).

## Irreducible-floor analysis
Dominant intermediate = colf30 [1,1,30,30] fp32 = 3600B: the colour-index plane produced by the single
1x1 Conv (sum_k k*input_k) over the full 30x30 input. This is the documented 3600B fp32 colour-index
ENTRY floor — it cannot be narrowed by dtype (Conv output follows fp32 input) and cannot be produced at
10x10 without first slicing the 10-channel input ([1,10,10,10] = 4000B, WORSE). Second plane = padded L
[1,1,30,30] uint8 = 900B (needed: Equal requires a 30x30 label and Pad rejects bool, so the uint8 label
is padded with sentinel 10, not the bool output). Everything else is 10x10 (<=200B).

## OPEN ANGLES (re-attack backlog)
- MatMul channel-contraction to build `center` at 10x10 directly (avoid colf30): blocked — reshaping
  input to [1,1,10,900] or forming the uniq*input product still costs ~3600B. No net gain.
- Could the off-grid sentinel be folded so L stays 10x10 (Equal@10x10 then pad bool)? Blocked: Pad
  rejects bool. uint8 pad of L is the cheapest carrier.

## INSIGHT (transferable)
"Unique element = the count-1 channel": when a generator forces exactly one colour to appear once and
all others >=2, the centre/anchor is recovered with ZERO argmin ops via `Equal(ReduceSum(input,[2,3]),1)`
on the [1,10,1,1] per-channel counts — read its colour as `Sum_k k*uniq_k` scalar, locate it as
`Equal(colour_index_plane, ucolor)`. Combined with the slice-to-active-region + Pad-with-sentinel pattern
(target leaves the off-grid 30x30 region ALL-ZERO, not ch0=1 — so L must be sentinel>=10 there, never 0).
