# task039 — 2013d3e2

**Rule:** The 10x10 input holds a "pinwheel": 4 rotated copies of a 3x3 colour
pattern placed so the UPPER-LEFT 3x3 block at (row, col) equals the 3x3 output
exactly (`output[r][c] = grid[row+r][col+c]`, r,c in 0..2). The generator picks
row, col in {1,2,3}; the full figure spans rows row..row+5, cols col..col+5, so
all activity lives within rows 1..8, cols 1..8. The 3x3 crop therefore starts at
the bounding-box minimum (min nonzero row, min nonzero col) — verified equal to
the output on 3000+ fresh instances. ⇒ Pure spatial COPY of a 3x3 input window.
**Current (prior public):** 16.04 pts, find-first-row/col + 2-stage Gather + Pad,
mem 7744, params 28.
**Target tier:** S (output = a direct copy of input cells; only two scalar
offsets first_row/first_col are computed, no colour-index plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior public: slice ch1-9 to [1,9,10,10] (3600) for presence; gather rows full-width [1,10,3,30] (3600) | S | 7744 | 28 | 16.04 | — | baseline |
| 1 | presence from two TINY candidate-window slices [1,9,3,8]+[1,9,8,3] (864 each); gather rows full-width [1,10,3,30] (3600) | S | 5792 | 32 | 16.330 | — | kills the 3600 presence slice; just shy of +0.3 |
| 2 | + crop on the active 8x8 region: slice input[rows1..8,cols1..8]=[1,10,8,8] (2560), gather local → [1,10,3,8] (960) | S | **5696** | 36 | **16.346** | **200/200** | beats +0.3 |

## Best achieved
**16.346 @ mem 5696 params 36 — fresh 200/200.** Beats prior 16.04 by **+0.306**.
Adopted? **N** (main adopts via `python -m src.adopt 39`).

## Irreducible-floor analysis
Two roughly-equal contributors after the rework:
- **act = input[1:9,1:9] [1,10,8,8] f32 = 2560 B** + first gather [1,10,3,8] = 960 B.
  The crop must keep all 10 channels (output is a one-hot copy, incl. background
  channel-0 cells), and a two-axis gather always leaves ONE spatial axis full in
  the intermediate. Cropping to the 8x8 active region (rows/cols 1..8 cover the
  whole possible pinwheel) shrinks that full axis 30→8, turning the prior 3600 B
  [1,10,3,30] intermediate into 2560+960=3520 B.
- **two presence windows [1,9,3,8] + [1,9,8,3] = 864 B each (1728 B).** Because
  min_row, min_col ∈ {1,2,3}, the FIRST present row/col is already the global
  minimum, so presence only needs scanning rows 1..3 (cols 1..8) and cols 1..3
  (rows 1..8) — far cheaper than the prior full [1,9,10,10] = 3600 B slice.
Everything else ≤ 360 B (the [1,10,3,3] crop) or tiny [1,3]/[3] index/scalar
tensors. This is at the Tier-S floor for a copy that must materialise a 10-channel
gather intermediate.

## OPEN ANGLES (re-attack backlog)
- **Eliminate the [1,10,8,8] act + [1,10,3,8] gather (3520 B) entirely** by
  selecting the 3x3 window with a double {0,1}-MatMul Rmat@input@Cmat over the
  active region — measured to land in the same ~4kB ballpark (MatMul still
  produces a [1,10,3,8] intermediate), so no clear win; gather is simpler.
- Pack first_row/first_col into one tensor / one ArgMax — saves only a handful of
  bytes, won't move ln-mem score meaningfully.

## INSIGHT (transferable)
⭐ **"first nonzero row/col when the generator BOUNDS the min to a tiny set
{1,2,3}"**: presence scanning only needs the candidate window, not the whole
canvas — slice `input[ch1..9, rows 1..K, cols active]` → ReduceMax → ArgMax gives
the first (= minimum) index directly. This collapsed a 3600 B full-grid presence
slice to 864 B.
⭐ **Two-axis Gather floor = the larger spatial axis stays full** ([1,C,3,W]).
Pre-cropping the input to the generator-bounded active region (here 8×8) shrinks
that residual axis and beats gathering on the full 30-wide canvas, even after
paying for the crop slice. Reuse the same active slice for both gathers.
