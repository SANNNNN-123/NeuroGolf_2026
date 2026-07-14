# task220 — 913fb3ed

**Rule:** A square grid (size 6..16) holds 1..3 single coloured seed pixels, colours drawn
from {2,3,8}, each at an interior cell (rows/cols 1..size-2) with NO two 3x3 stamps
overlapping. OUTPUT: for every seed at (r,c) of colour `g`, fill the full 3x3 neighbourhood
with `colormap[g]` (2->1, 3->6, 8->4), then restore the centre to the original colour `g`.
Per cell: seed centre keeps original colour; the 8-cell ring takes the mapped halo colour;
everything else background. Stamps never overlap and never clip the grid, so per-colour
processing is fully independent and purely linear over the input one-hot.

**Current (pre-existing base):** stored ~18.2 pts but FRESH-RATE 0.00 (non-generalizing,
scores ~0 on real LB).

**Target tier:** S — the whole map is a fixed linear 3x3 stencil (dilate-minus-centre per
colour) so a single 10->10 Conv straight into the FREE `output` realises it with zero
intermediate memory.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | single 10x10x3x3 Conv -> output (halo=ring stencil, centre=identity, bg=ch0 minus rings) | S | 0 | 900 | 18.20 | 200/200 | PASS |

## Best achieved
18.20 @ mem 0 params 900 — adopted? leave to caller (recommend Y). Beats prior real (0, non-gen)? YES by ~+18.2.

## Encoding detail
`W[out_ch, in_ch, 3,3]`:
- `W[h(g), g] = ring` (3x3 ones with centre 0) — ring cells get the halo colour h(g).
- `W[g, g, 1, 1] = 1` — centre keeps original colour.
- `W[0, 0, 1, 1] = 1` — copy input background channel through.
- `W[0, g] = -ring` — ring cells leave the background channel (1 - 1 = 0).
SAME pad (pads=[1,1,1,1]). Output compared as `>0`. No-overlap guarantee means a background
cell is adjacent to at most one seed, so the bg subtraction is exactly 1-1=0 (never goes
negative-but-still->0 ambiguous, never double-subtracts). Verified evaluate 267/267 stored,
fresh 200/200.

## Irreducible-floor analysis
Not at a floor — this is Tier S (mem 0). The only cost is the 900 conv params (10x10x3x3),
which is fp32-free in the static scorer's element-count model anyway; score is dominated by
ln(0+900). A depthwise/grouped formulation cannot reduce params here because output channels
read a DIFFERENT input channel than their own (h(g) reads g), so the cross-channel dense conv
is required. Could trim params by zeroing unused channels, but params already give 18.20 and
ln(900)~6.8 so trimming to e.g. 81 nonzero only moves the score a little; not worth risk.

## OPEN ANGLES (re-attack backlog)
- Param shrink: only 3 colour channels are ever nonzero in/out; a 1x1 reshuffle + 3x3
  depthwise on a 3-channel sub-slice could cut params from 900 toward ~30, nudging pts up a
  fraction. Marginal (ln-scale), low priority.

## INSIGHT (transferable)
"Expand each seed pixel into a fixed recoloured K×K stamp" with NON-overlapping,
non-clipping guarantees is a pure linear stencil = a SINGLE dense Conv 10->10 straight to the
free output (Tier S, mem 0). The centre-vs-ring recolour is just two stencils in the same
kernel (ring ones with centre 0 routes to the halo channel; centre-only identity routes to
the seed channel); the background channel is maintained by copying input ch0 and subtracting
the ring contribution. No flood-fill, no argmax, no intermediates. ⭐ Whenever a task is a
per-seed fixed-shape stamp with an overlap-free generator guarantee, reach for conv_network
first.

## S10 (2026-07-03) — knife-edge hardening ADOPTED (±0 pts, robustness)
Single-Conv logits sat on {0,1} grid with off-cells EXACTLY at the >0.0 decode threshold →
any prior same-shape-Conv evaluate() in the process (ORT arena state) flipped ALL examples
to fail (local batch totals under-counted; grader currently unaffected — LB clean).
Fix: subtract 0.5 from W[:,:,1,1] (all 10 center taps) → in-grid logits shift −0.5
(on ≥ +0.5, off ≤ −0.5), padded cells stay exact-0/OFF. mem 0 / params 900 UNCHANGED.
Gates: fresh-process bundled 0-fail, dirty-process (pollutants 230,294) 0-fail vs incumbent
267-fail positive control, fresh-2000 0/0. Backup task220_pre_s10_knifeedge.onnx.
⭐ TRANSFERABLE: any mem=0 single-Conv net — screen max off-cell logit == 0.0 on a clean
run; fix via center-tap (no-bias) or bias epsilon shift. task193 screened healthy (gap 1.0).
