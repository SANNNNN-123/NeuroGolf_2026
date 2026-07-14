# task034 — 1f0c79e5

**Rule:** A 2×2 colored seed square at top-left corner (R,C); its 4 corners map 1:1 to the 4 diagonal
directions. A corner painted RED(2) in the input sprouts an outward width-3 diagonal staircase to the
edge in the output. Closed form rel. to chosen corner: painted iff a>=0 & b>=0 & |a-b|<=1, where
a=(r-r0)·dr, b=(c-c0)·dc. size=9 grid.
**Current:** 15.16 pts (custom:task034, adopted from gen:thbdh6332 13.99), mem 18666, params 70
**Target tier:** B (label-map) achievable; rule is per-cell closed-form so B floor ~3600 is the real
target, NOT 18666. Possibly A (separable per-direction) with more work.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | closed-form predicate per 4 dirs on 9×9 fp32 planes → uint8 L9 → Pad → Equal | B-ish | 18666 | 70 | 15.16 | 200/200 | ADOPTED (+1.17) |
| 2 | shrink: scalars-only params + bool/uint8 predicates + 3-ch 7×7 crop | B | 5515 | 140 | **16.36** | 1000/1000 | candidate (beats 15.16 by +1.20) |

## Best achieved
**16.36 @ mem 5515 params 140 — fresh 1000/1000 + all 28 realizable edge cases (corner∈{1,6}²
× every 1/2/3-dir combo).** Beats adopted 15.16 by +1.20. Main should `adopt 34`.

## Irreducible-floor analysis (after attempt 2)
At/near the Tier-B floor for this rule. mem_profile top: **900B uint8 [1,1,30,30] final Pad of L**
(IRREDUCIBLE — must fill the 30×30 output region for Equal→[1,10,30,30] free BOOL output) +
**588B fp32 [1,3,7,7] input crop** (near-irreducible: the only per-cell view of the input; needs
contiguous channels 0..2 because occupancy=bg(ch0)-off and red=ch2-on, and Pad crops contiguous
ranges only — can't grab {0,2} alone without a full-res [1,2,30,30]=7200 Gather) + 2×196B fp32
single-output Convs (bg plane, red plane) + ~25 × 81B bool [1,1,9,9] predicate planes (the 4-direction
band logic: ab, |a-b|≤1 as two Greaters, AND-combine, OR-accumulate). The 81B bool planes are the
last soft target (~2025B) but each is a genuine logical step; merging the 4 directions would need a
unified diagonal-band reformulation (Tier A).

## What changed 18666→5515 (transferable recipe)
1. Params are SCALARS, not planes: colour from channel-presence `ReduceMax(input,[2,3])` →[1,10,1,1]
   (40B) → mask out ch0/ch2 → ReduceSum. R/C via `ReduceMin(bgplane)` along one axis → 1-D compare.
   No per-cell colour plane exists.
2. Crop the input to a 3-channel 7×7 window (588B) instead of materializing [1,10,9,9] (3240).
   Only ch0 (occupancy=bg-off, works for ALL colours) and ch2 (red) are needed; crop the contiguous
   0..2 range. Window offset (1,1) since the seed 2×2 always lives in rows/cols 1..7.
3. Every predicate/mask plane is bool/uint8 (81B), never fp32 (324B). `|a-b|≤1` = (a>b-1.5)&(b>a-1.5)
   — two bool Greaters of [1,1,9,1] vs [1,1,1,9], NOT Sub+Abs+Less (which forces fp32 9×9).
4. Corner-red flags via a TRUE 2-D `Gather(red, rows2, axis=2)` then `Gather(.., cols2, axis=3)` →
   [1,1,2,2] then Slice the [i,j] scalar. A row∧col outer-product flag is WRONG (two diagonal reds
   cross-talk into the off-diagonal corner — verified: 113/200 fresh).
5. Fold the chosen-flag scalar into the 1-D `age` (`age & flag`, [1,1,9,1]=9B) so no 9×9 gate plane.
6. Seed 2×2 square computed ONCE (shared by all dirs) not 4×.

## OPEN ANGLES (still untried after attempt 2 — diminishing returns)
- **Equal at 9×9 then Pad the OUTPUT (free):** L9→Equal→[1,10,9,9] bool (810B) then `Pad`(fill 0) to
  [1,10,30,30] named `output`. Replaces the 900B uint8 L Pad with an 810B intermediate (saves ~90B,
  ~0.01 pts). Needs ORT bool-Pad to work; not attempted (risk > reward).
- **Tier A unify the 4 directions:** a>=0&b>=0&|a-b|<=1 is a width-3 diagonal band; the 4 dirs are the
  same band reflected. A single signed-coordinate transform that emits one band, reflected/translated
  per corner via index tricks, could collapse the ~25 bool planes (2025B) to ~6. This is the only path
  meaningfully below 5515. Hard; the corner-specific deltas + per-corner chosen-flag gating resist a
  clean shared form.
- **Smaller crop:** ch1 in the 0..2 window is dead weight (196B of 588). Removing it needs non-contiguous
  channel selection (Gather→full-res→crop = worse) — no cheap route found.

## INSIGHT (transferable)
⭐ A "working adopt" at 15 with high memory is a HALF-win: agents tend to leave the predicate scaffold in
fp32. ALWAYS mem_profile after adopt; if a long tail of equal-size fp32 planes appears, they are
boolean/small-int and downcasting to uint8 is a free 4× cut.
⭐⭐ **The deepest cut is REFUSING TO MATERIALIZE PLANES AT ALL.** This task's params (colour, R, C, 4 red
flags) are all SCALARS — derivable by channel/axis reductions that never touch a per-cell plane
(`ReduceMax(input,[2,3])`→[1,10,1,1] presence; `ReduceMin(bgplane)`→1-D occupancy). The only per-cell
tensor you truly need is a tiny CROPPED window of the *minimal contiguous channel set* (here ch0..2:
occupancy=bg-off works for every colour, red=ch2), not the full [1,10,9,9]. 18666→5515 came mostly from
"what's actually a scalar?" + "what's the smallest channel/spatial slice that answers it?", and only
secondarily from uint8 downcasting.
⭐ `|a-b|<=1` → `(a > b-1.5) AND (b > a-1.5)` keeps everything bool; Sub+Abs+Less forces a fp32 plane.
⭐ Corner/point lookups in a 2-D field: use chained `Gather`(axis=2 then axis=3), NOT a row∧col outer
product — the outer product cross-talks whenever two marked cells share neither row nor col.
⭐ Crop offset trick: if the active region is a sub-window (rows/cols 1..7), crop to [.,.,7,7] at offset
(1,1) and add the offset back into the index `arange` const — saves vs cropping the full grid.
