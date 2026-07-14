# task318 — ce4f8723

**Rule:** size=4 task. INPUT is 4 x (2*4+1)=9: TOP block (rows 0..3) holds only
colour-1 pixels, BOTTOM block (rows 5..8) holds only colour-2 pixels, middle row 4
is a yellow separator. OUTPUT is 4x4: cell (r,c)=GREEN(3) iff TOP has a pixel at
(r,c) OR BOTTOM has a pixel at (r,c); else background 0. Off-grid (>=4) is all-zero
(target is np.zeros; only the 4x4 region is set, so off-grid needs NO ch0=1).
**Current:** 18.40 pts, public slice/occ-OR/Concat/Pad net, mem ~734, params 1.
**Target tier:** A — pure overlay-then-recolour of two free input slices; no detection.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice ch1-top + ch2-bot, Or, Concat[1,10,4,4] uint8, Pad | A | 384 | 41 | 18.95 | 200/200 | adopt |

## Best achieved
18.95 @ mem 384 params 41 — adopted? Y. Beats prior 18.40? Y (+0.55).

## Irreducible-floor analysis
Dominant intermediates: the [1,10,4,4] uint8 Concat carrier (160B) + two fp32 4x4
input slices (64B each) + a handful of bool 4x4 planes (16B). The carrier is the
minimal way to assemble a 10-channel one-hot before the single Pad-into-free-output;
routing via Equal would need a 30x30 index plane (3600B, strictly worse). Everything
lives on the 4x4 active region, so there is no full-canvas plane anywhere.

## OPEN ANGLES (re-attack backlog)
- Fold the two fp32 slices into one (non-contiguous channel+row blocks make this
  hard); marginal (~64B) and not worth chasing below 384.
- Replace the uint8 zero initializer (16 elem params) with a cheaper construction —
  Sub rejects uint8, so a const init is the simplest legal zero plane.

## INSIGHT (transferable)
The public net inverted bg (1 - ch0) per block then OR'd; but each half-block only
ever contains ONE colour (top=1, bottom=2), so occupancy is just two direct
single-channel input slices OR'd — no Cast/Sub bg gymnastics. And off-grid does NOT
need ch0=1: the harness target is np.zeros with only the size x size region set, so a
plain value-0 Pad (all-zero off-grid) matches exactly. uint8 Sub/Mul/And are rejected
by ORT, so build a zero plane as a constant initializer, not via Sub(x,x).
