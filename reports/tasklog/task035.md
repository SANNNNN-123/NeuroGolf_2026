# task035 — 1f642eb9

**Rule:** A cyan(8) rectangle ("pool") sits at rows 3..r1 (r1∈4..7), cols c0..5
(c0∈2..4); row-start 3 and col-end 5 are FIXED, only r1 and c0 vary. Scattered
non-cyan colour pixels lie on the four grid borders (row 0, row 9, col 0, col 9),
each ALIGNED with a pool row/column. The output keeps the entire input and ALSO
copies every border pixel inward onto the nearest pool edge cell. Verified exact
closed form (50000/0, no collisions between the four lines):
  out[3 ,c] = input[0,c] (where nonzero)   out[r1,c] = input[9,c]
  out[r,c0] = input[r,0]                    out[r ,5] = input[r,9]
i.e. the projected colour is simply a COPY of the border row/column — NO colour
detection of the border pixels is needed; only the two scalar pool bounds r1,c0.
**Current:** 16.20 pts, custom:task035 (label-map, 10×10 canvas), mem 6548,
params 76. Prior 15.36 (public ext:kojimar6275).
**Target tier:** B (label-map + final Equal). Not S: the projected colour at a
pool-edge cell is a NON-LOCAL gather (it comes from a border pixel in the same
row/column), so no single fixed Conv/permute over a local window produces the
output. Not separable A: the override is `cell-on-line AND projected-value≠0`,
where the projected value is a copy of a whole border line, not a row⊗col scalar
product. B (per-cell label map) is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colour-idx Conv→30×30 f32, cast 30×30 u8, cyan slice 30×30 f32, 4 broadcasting Where overrides, Pad, Equal | B | 10608 | 130 | 15.72 | 2000/2000 | works, not minimal |
| 2 | slice colour-idx to 10×10 f32 BEFORE cast; slice cyan only on 10×10 corner; bounds on 10×10 canvas | B | **6548** | 76 | **16.20** | 20000/20000 (all 12 r1,c0 combos) | candidate — beats 15.36 by +0.84 |

## Best achieved
**16.20 @ mem 6548 params 76 — fresh 20000/20000 across every (r1,c0) bound
combination.** Beats prior 15.36 by **+0.84**. Adopted? **N** (main adopts via
`python -m src.adopt 35`).

## Irreducible-floor analysis
At the Tier-B floor for this rule. mem_profile top:
- **3600 B f32 [1,1,30,30] Conv** — the colour-index gateway (`V=Σ k·onehot[k]`
  via a 1×1 [0..9] Conv). IRREDUCIBLE: the input is declared 30×30, a 1×1 Conv
  preserves spatial extent, and Conv output follows the float input dtype. Every
  cheaper alternative is worse: `Mul(input,kw)` = 36000 B; slicing the one-hot to
  [1,10,10,10] first = 4000 B (> 3600) before any conv; casting input to uint8/
  fp16 first = 9000/18000 B. The colour index for the whole 10×10 region is
  genuinely needed (output = input everywhere off the 4 override lines).
- **900 B u8 [1,1,30,30] Pad** of the label map (sentinel 10 outside the 10×10
  grid) — irreducible: the final Equal must span the 30×30 output region; Equal at
  10×10 then bool-Pad the output = 1000 B (worse).
- Everything else ≤ 400 B: two 400 B f32 10×10 slices (Vf colour crop; cyan crop
  for the bounds), 100 B uint8 Where label planes, ≤40 B 1-D bound aggregates.

## OPEN ANGLES (re-attack backlog)
- **Eliminate the 3600 B Conv via a non-Conv colour index.** No op produces a
  per-cell argmax of a 10-channel one-hot more cheaply than a 1×1 Conv whose
  output is the (free-input-sized) 30×30 — searched, none found. Would need the
  rule to be a single-Conv-into-output (Tier S), which the non-local projection
  forbids. This is the only path meaningfully below 6548 and looks closed.
- Drop the 900 B Pad by emitting Equal at 10×10 then padding the bool output —
  costs 1000 B bool, net worse. Not pursued.

## INSIGHT (transferable)
⭐⭐ **When a "paint projected pixels" rule projects a border pixel straight along
its own row/column to a target edge, the projected COLOUR is literally a copy of
that border row/column — there is no colour to *detect*.** Slice the colour-index
plane's edge rows/cols (toprow/botrow/leftcol/rightcol) and broadcast them into
the label map with `Where(on-line-mask AND copied-value≠0, copied-line, base)`.
The 4 lines here provably never collide (checked 50000), so the 4 Where overrides
commute — no priority logic needed. This collapses what looks like a detection
task to a clean Tier-B label map.
⭐ **Slice the float Conv output to the active canvas BEFORE Cast-to-uint8**: a
[1,1,10,10] f32 slice (400 B) + Cast (100 B) beats Cast-30×30 (900 B) + Slice
(100 B) — saves 500 B for free. Likewise slice the cyan channel only on the
10×10 corner (axes [1,2,3] → 400 B) instead of the full 30×30 (3600 B); compute
r1/c0 by `Where(present, arange10, ±sentinel)` + Reduce on the 10×10 canvas.
⭐ ORT `Greater` rejects uint8 (use `Not(Equal(x,0))` for a nonzero gate); cyan
bounds need a float plane because ReduceMax/Min reject uint8/bool.

## S16 (2026-07-06) — public bit-identical golf (franksunp) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
