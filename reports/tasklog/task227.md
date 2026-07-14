# task227 — 94f9d214

**Rule:** Input is a size-tall × 2·size-wide grid (size=4 always → 8×4): TOP half (rows 0..3) holds green pixels, BOTTOM half (rows 4..7) holds blue pixels at random positions. Output is size×size (4×4): `output[r][c] = red` iff `grid[r][c]==bg AND grid[size+r][c]==bg` — i.e. a per-pixel NOR of the two stacked halves (red exactly where both halves are empty at column c), else background.
**Current:** 18.19 pts, generic conv9x1+b (910 params, mem 0)
**Target tier:** S/A — closed-form per-pixel logic on a tiny fixed 4×4 active region; no full plane needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice ch0 of both halves, red=top·bot, bg=1−red, Concat[1,10,4,4]+Pad | A | 448 | 153 | 18.60 | 200/200 | pass |
| 2 | Concat only [1,3,4,4], Pad channels(+7)+spatial(+26) | A | 336 | 41 | 19.07 | 200/200 | adopted |
| 3 | fp16 arithmetic on the 4×4 planes | A | 336 | 41 | 19.07 | — | no gain (extra cast planes offset the halving) |

## Best achieved
19.07 @ mem 336 params 41 — adopted? Y. Beats prior 18.19 by +0.88? Y.

## Irreducible-floor analysis
mem 336B is dominated by four tiny [1,1,4,4] planes: top/bot f32 slices (64B each) and red/bg f32 (64B each), plus the [1,3,4,4] uint8 block (48B) and two uint8 casts (16B each). These are floor for the logic: two halves must be sliced (f32, Slice preserves input dtype), one product + one complement, cast to uint8 for the Concat. fp16 casts add 2 planes that exactly offset the byte halving, so no net gain.

## OPEN ANGLES (re-attack backlog)
- Build the 3-channel block directly with a single op that yields uint8 (avoid the two separate red_u/bg_u casts) — marginal (~32B).
- Replace Mul+Sub with a single banded conv on a 1-channel occupancy slice — unlikely to beat the current tiny-plane floor.

## INSIGHT (transferable)
⭐ "background channel 0 == 1 ⇔ cell empty" turns any per-pixel logical rule over EMPTINESS into a direct slice of ch0 — no occupancy reduction (Σ_k k·input_k) needed. ⭐ When the output is a small fixed block on a known-empty canvas, Pad the CHANNEL dim too (pads `[...,7,...]`) so you Concat only the channels you actually set (here 3) instead of all 10 — cut the block plane 160→48B and params 153→41 in one move. The whole rule reduces to: slice ch0 of each half, Mul (=AND), Sub from 1 (=bg), Concat 3 channels, Pad to [1,10,30,30].
