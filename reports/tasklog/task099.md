# task099 — 444801d8

**Rule:** 10x10 input holds 1-2 vertical blue(1)-outlined width-5 "boxes"; each
carries exactly one coloured "seed" pixel inside. Output paints every interior
background cell of a box AND a full-width "lid" row directly above the box top
with that box's seed colour; blue stays blue, everything else stays background.
Boxes are always row-separable (upper box above lower box), so colour is a
per-row-band property — but the painted SHAPE (which cells fill) is a purely
local function of the binarised input (a painted cell's seed always sits within
rows [-1..+3], cols [-2..+2] of it).
**Current:** 16.84 pts, ext:thbdh6285 = single Conv 7x5 (10->10), mem 0, params 3510.
**Target tier:** S — output colour per cell is a local function of the input
one-hot window, so one Conv input->output (mem 0). Only lever is shrinking params.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map M(prefix-OR+lid) + region-gated MaxPool colour spread | B | ~4-6k | small | ~16.3-16.5 | exact | abandoned — colf 3600B floor + Pad 900B caps below the public Conv |
| 2 | per-channel `M AND dilate(input_k, r=3)` single-Conv | S | 0 | small | — | FAIL | cross-box colour leak (gap=1 row, seed reach 3) — dilation not region-gated |
| 3 | Conv 7x5 (sanity, = public) trained from scratch | S | 0 | 3510 | 16.84 | 1.0 | reproduces public; no gain |
| 4 | Conv 5x5 SYMMETRIC pads | S | 0 | 2510 | 17.17 | 0.884 | capacity OK but wrong reach — kernel can't see seed +3 down |
| 5 | Conv 5x5 pads=[top1,left2,bot3,right2], trained on FULL 30x30 | S | 0 | 2510 | **17.17** | **500/500** | WIN |

## Best achieved
17.172 @ mem 0 params 2510 — beats prior 16.84 by +0.33 (and P=16.74 by +0.43).
Single Conv 5x5 (10->10), asymmetric pads [1,2,3,2], weights SGD-fit (hinge,
positive-weighted) on fresh generator samples. evaluate() ok pass 265/265,
isolated fresh 500/500.

## Irreducible-floor analysis
Tier S, mem 0 (single Conv input->output). Cost is entirely the kernel params
10*10*KH*KW. Vertical reach must cover seed offset dr in [-1..+3] (lid is 3
rows above its seed) and dc in [-2..+2] (lid corners are 2 cols from the seed),
so KH>=5, KW>=5 is the minimum kernel that lets every painted cell "see" its
seed -> 2510 params is the floor for the direct single-Conv. Going below needs
KW<5, which provably can't reach a lid corner, or fewer channels, which can't
route the box-shape (blue/bg channels) into every colour output channel.

## OPEN ANGLES (re-attack backlog)
- KW=4 (2010 params, 17.68) IF some lid-corner reformulation removes the
  +-2 horizontal reach need (none found — lid corner is genuinely 2 cols out).
- Grouped/depthwise Conv to cut the channel-mixing params (blocked: M-shape must
  flow from blue/bg channels into each colour channel, needs full mixing).

## INSIGHT (transferable)
⭐ When the PUBLIC net is already a single Conv (mem 0), beating it = shrinking
the KERNEL. Measure the exact seed->painted offset envelope (dr/dc range over
fresh samples) to size KH/KW MINIMALLY and use ASYMMETRIC `pads` to place the
window over the asymmetric reach — a symmetric kernel wastes params on the unused
side (here 7x5=3510 -> 5x5=2510, +0.33). ⭐ Train Conv weights on the FULL 30x30
one-hot representation, NOT the cropped active grid: convert_to_numpy leaves
out-of-grid cells ALL-ZERO (channel 0 = 0, not 1), so a net trained only on the
10x10 region mis-handles the grid->zero boundary and fails fresh verification
even at "gridok 1.0" on the cropped eval.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
