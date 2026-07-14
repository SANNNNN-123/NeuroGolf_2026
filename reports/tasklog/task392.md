# task392 — f8c80d96

**Rule:** Nested concentric square "mats". Generator picks centre (row,col) with
ONE of {row,col}==0 (centre on an edge), thickness `thick`∈{1,2}, a colour, and
`show`∈{2,3}. It draws `size`(=10) concentric square ring perimeters: ring `i`
is the boundary of the box [A_i,B_i]² where A_i=thick−(thick+1)·i,
B_i=(thick+1)·i−1. INPUT shows only the first `show+1` rings; OUTPUT draws ALL
rings on a gray(5) background. Output is a pure function of (row,col,thick,colour),
independent of `show`. Closed form (verified exhaustively):
`painted(r,c) ⇔ max(need(r−row),need(c−col)) % (thick+1)==0`, `need(x)=max(x+1,thick−x)`.
Grid is always 10×10 (size fixed at 10).
**Current:** 14.66 pts (prior P), method = public net.
**Target tier:** B — per-cell deterministic; only 38 distinct full patterns exist
(one of row/col is 0, thick∈{1,2}), but recovering (row,col,thick) as direct
scalars is blocked by grid-edge clipping of the shown partial rings, so the
clean form is a const-candidate match → label plane → Equal.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | min-area-cover over 40 cand masks, dup const cube+flat | B | 8056 | 7694 | 15.34 | 266/266 | working, dup const |
| 2 | dedup to 38 masks, gather from flat (drop cube) | B | 8056 | 3894 | 15.61 | — | leaner params |
| 3 | painted via ch0-slice (not 9-ch sum) + colour via ReduceMax(input) | B | 4464 | 3895 | 15.97 | 200/200, 500/500 | ADOPT-CANDIDATE |

## Best achieved
**15.97 @ mem 4464 params 3895** — beats prior 14.66 by **+1.31**. fresh 500/500.

Method: the shown input rings are always a SUBSET of the (unique) full pattern,
and that full pattern is the smallest-AREA candidate that COVERS the shown input
(proven unique over 60000 fresh instances). Score_k = area_k + 1000·(|P|−covered_k)
where covered_k = MatMul(Cflat[38,100], Pvec[100,1]); ArgMin → Gather the winning
[100] mask → reshape → label plane (colour where painted, gray5 where in-grid &
unpainted, sentinel 10 off-grid) → Pad to 30×30 → Equal(L,arange[0..9]) into the
FREE bool output.

## Irreducible-floor analysis
params=3895 dominates: the [38,100] candidate-mask initializer (3800 elems). It is
the score floor for this formulation — the min-area-cover scorer must compare the
input mask against every distinct full pattern. Memory floor: padded label plane
Lp uint8 [1,1,30,30]=900B (must be 30×30 to broadcast against 10 colour channels
in the final Equal) + the 10×10 working tensors (≤400B each). Everything else tiny.

## OPEN ANGLES (re-attack backlog)
- **Transpose-symmetry halving (~+0.3):** full_mask(0,k,th)==full_mask(k,0,th)ᵀ, so
  store only the ~19 col0-family masks (1900 params), run the cover-match on P AND
  Pᵀ, pick the global min-area winner, transpose the output if the Pᵀ branch wins.
  → params ~1900, est ~16.25. Skipped: modest gain, added conditional-transpose risk;
  current win already comfortably >+0.3.
- **Direct scalar recovery (would zero the 3800 params, → Tier A/S):** recover
  (axis, centre, thick) as scalars then rebuild from the `need` predicate via ramps.
  BLOCKED so far: shown partial rings clip asymmetrically at the grid edge, so
  per-row/col occupancy signatures of (axis,thick) overlap (measured — col0/row0
  families collide). Would need a clip-robust scalar formula (e.g. ring-edge column
  spacing = thick+1 read off the un-clipped side).

## INSIGHT (transferable)
⭐ When a task's output is a pure function of a SMALL discrete parameter set (here
38 distinct full patterns) and the input is always a faithful SUBSET of the true
output, **min-area-covering candidate match** is exact and ONNX-cheap: covered_k =
MatMul(Cflat[U,HW], Pvec[HW,1]); pick ArgMin(area_k + BIG·(|P|−covered_k)); Gather
the winning flattened mask. Avoids any fragile per-scalar recovery when partial-
input clipping makes direct reductions ambiguous. Cost = U·HW params (the floor);
halve via transpose/rotation symmetry of the candidate family when present.
Also: recover painted-mask from the ch0 (background) slice `Less(bg,0.5)` — one
[1,1,G,G] slice, NOT a 9-channel ReduceSum (saves the 3600B fg plane); recover a
single fg colour scalar via `ReduceSum_k k·ReduceMax_spatial(input)` straight off
the FREE input (40B), no grid slice.
