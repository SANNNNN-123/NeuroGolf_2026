# task057 — 28bf18c6 ("crop the 3x3 sprite and tile it horizontally x2")

**Rule:** An 8x8 grid holds a single-colour 3x3 conway sprite at a variable
top-left `(row,col)` on a 0 background. The output is a 3x6 grid = the 3x3 sprite
tiled twice horizontally: `output[r][c] = output[r][c+3] = sprite[r][c]` for
`r,c in 0..2`. Verified over all 265 stored + 2000 fresh instances: grid is always
exactly 8x8, exactly one non-zero colour, the occupied bbox is exactly 3x3, and
`output == sprite[r0:r0+3, c0:c0+3]` tiled x2 with `r0/c0 = min occupied row/col`.
(Original triage "infeasible" was wrong — this is a clean data-dependent crop+tile.)

**Current (public):** 14.7 pts (mislabeled-infeasible).
**Target tier:** B — the crop offset `(r0,c0)` is data-dependent, so no fixed
Conv/permute window produces the output; but the crop is row/col separable and the
tile is a fixed pattern, so it collapses to two boolean shift-MatMuls (not a Gather
loop, not a 2-D detection net). Not S (offset varies per instance).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | presence `1-bg`; `r0/c0` via ReduceMax+ramp min-index; colour via spatial Mul+ReduceSum over [1,10,8,8] slice; shift double-MatMul `Rmat@P@CmatT` (CmatT folds the x2 tile via `j%3`); Where->label->Pad->Equal | B | 7445 | 93 | 16.07 | 200/200 | works |
| 2 | recover colour from per-channel presence `ReduceMax(input,[2,3])`*idx then ReduceMax — kills the two 2560 B [1,10,8,8] planes | B | 2149 | 84 | 17.29 | — | trim |
| 3 | presence + MatMul in fp16 ({0,1} exact): drop f32 `P` and the separate fp16 cast (Sub directly in fp16, min-index ReduceMax on fp16) | B | **1989** | **83** | **17.36** | **200/200** | FINAL |

## Best achieved
**17.36 pts @ mem 1989, params 83 — 265/265 stored, fresh 200/200, stress 2000/2000.**
Adopted? **N** (main adopts via `python -m src.adopt 57`). Beats public 14.7 by **+2.66.**

## Irreducible-floor analysis
Dominant intermediate is the **900 B uint8 `L` Pad [1,1,30,30]** — the 30x30 label
feeding the FREE final `Equal(L, arange[0..9])`. The output spans the full 30x30
canvas and only the Pad sentinel (10) makes off-grid cells match no channel
(all-zero). uint8 is already the cheapest dtype; irreducible (same floor as task250).
Remaining cost: one 256 B f32 `bg` slice (Slice preserves input fp32; feeds the
fp16 presence Sub), one 128 B fp16 `P16` (load-bearing twice — min-index ReduceMax
and the MatMul), the two fp16 shift matrices + intermediates (≤96 B each), and
fp32 scalar reductions (≤40 B). Colour recovery is now scalar-only (no plane).

## OPEN ANGLES (re-attack backlog)
- The 900 B Pad is the canonical output-canvas floor for "small output in a 30x30
  field". Could attempt building L directly at 3x6 and padding only once (already
  done) — no further win without a smaller-than-30x30 output, which the harness
  fixes. Effectively at floor.
- The 256 B f32 `bg` slice: could derive presence from `1 - ReduceMax(input[ch0])`
  but presence is needed as a full 8x8 plane for the MatMul, so the plane is
  unavoidable; fp16 already applied. Sub-0.1 pt at best.
- Tier S is blocked: crop offset `(r0,c0)` is data-dependent.

## INSIGHT (transferable)
⭐ A **data-dependent crop + fixed tile** is a Tier-B boolean **shift double-MatMul**,
not a Gather: `out = Rmat @ P @ CmatT` where `Rmat[i,r]=(r==i+r0)` translates the
bbox to the origin and the column matrix `CmatT[c,j]=(c==(j mod 3)+c0)` *folds the
horizontal x2 tiling directly into its index map* (`j mod tilew`) — the tile costs
zero extra ops. Offset scalars come from the standard ReduceMax-presence + ramp +
ReduceMin min-index. fp16 is exact for {0,1} presence MatMuls.
⭐ **Recover a single-colour scalar from per-channel presence, never a plane:**
`color = ReduceMax( ReduceMax(input,[2,3]) * arange[0..9] )` — collapsing spatially
first (input is FREE) turns a 2560 B [1,10,H,W] weighted plane into a 40 B
[1,10,1,1] vector (+1.2 pt here). Always reduce over space before weighting channels.
