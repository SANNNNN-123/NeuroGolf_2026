# task375 — ea786f4a

**Rule:** Input is a solid NxN square (N odd, 5..15) of one `color`, anchored top-left,
with the single CENTER cell (N//2,N//2) set to black(0). Output keeps the square but
paints BOTH diagonals black: out[r,c] = 0 iff (r==c OR r+c==N-1) else `color`, for
in-grid cells; off-grid cells stay all-channels-off. Output is a pure closed-form
function of N and color — the input center-black pixel is irrelevant.
**Current:** 15.51 pts, public net ext:wguesdon6304, → custom 16.91 pts, mem 2729, params 527
**Target tier:** B (label-map + Equal) — diagonal membership r==c OR r+c==N-1 couples r
and c, so it is NOT row⊗col separable (no Tier A) and not a local-neighbourhood linear
function (no Tier S single-Conv). The label-map fallback is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | scalars(color,s1) + separable in-grid + eye/RC diagonal masks → 15x15 uint8 L → Pad → Equal | B | 2729 | 527 | 16.91 | 200/200 | WIN +1.40 |

## Best achieved
16.91 @ mem 2729 params 527 — adopted? N (build-only; main adopts). Beats prior 15.51? Y (+1.40).

## Irreducible-floor analysis
Dominant intermediate is the padded label map L[1,1,30,30] uint8 = 900B; it cannot be
removed because the final Equal must emit the full [1,10,30,30] BOOL output, so the label
plane must span 30x30. All per-cell work (masks, in-grid bounds, diagonal Equals) is done
on a 15x15 working canvas (max grid side) as uint8/bool (~225B each) then padded with the
off-grid sentinel 10; only one 30x30 uint8 plane is materialized. color and N=size are
scalars (corner one-hot · arange; row-occupancy ReduceMax → max occupied index = N-1).

## OPEN ANGLES (re-attack backlog)
- Sub-900B: the 30x30 L is the floor for label-map+Equal. Eliminating it would require a
  Tier-A/S reformulation, but the diagonal relation is non-separable and global, so no
  cheaper exact encoding is apparent. Considered closed.

## INSIGHT (transferable)
⭐ "Diagonal-paint on an origin-anchored square" = label-map task where the two scalars
(color, N) are recovered for FREE: color = corner one-hot · arange (corner is never the
center); N-1 = max occupied row index via ReduceMax(input,axes=[ch,col]) · arange. The
diagonal masks are CONSTANT planes (eye for r==c; r+c sum-plane Equal s1 for anti), so the
only data-dependence is the scalar s1 broadcast into the anti-diagonal Equal. Work on a
15x15 canvas (max grid) + Pad-with-sentinel keeps every plane ≤900B uint8.
