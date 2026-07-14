# task170 — 6ecd11f4

**Rule:** Input (21..28 sq) holds (a) a `size`x`size` (size∈{3,4}) conway shape drawn in one
colour `scolor`, magnified by `smag`∈{3,4,5} at (srow=1, scol); every row/col of the shape is
occupied; and (b) a `size`x`size` colour box at (brow=H-size-1, bcol≥W//2) whose cell (r,c) holds
distinct colour `colors[r*size+c]`. Output = the colour box MASKED by the (downsampled) sprite
shape: `out[r][c] = colors[r*size+c]` if (r,c) is a sprite pixel else 0 (top-left aligned).
**Current:** 13.98 pts (675-node ArgMax/Concat net), mem ~61k
**Target tier:** B — data-dependent point lookups (chained Gather) into one colour-index plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | scalar-recover geometry + chained Gather, 7 full planes | B | 20512 | 117 | 15.07 | 200/200 | win |
| 2 | fold colored/sprite masks → 3 full planes (colf, eqsc, spv) | B | 11782 | 117 | 15.62 | 500/500 | BEST |

## Best achieved
15.62 @ mem 11782 params 117 — adopted? N (build-only). Beats prior 13.98 by **+1.64**. fresh 500/500.

## Irreducible-floor analysis
Three 30×30 fp32 planes dominate (3×3600=10800): `colf`=Σk·input_k (needed for box-colour gather,
bbot detection AND sprite equality); `eqsc`=Equal(colf,sc) and `spv`=Where(eqsc,toprows,0) recover
the sprite bbox (srow/scol/spcmax→smag). The top-row mask is LOAD-BEARING: a box cell whose random
colour happens to equal sc pollutes the unmasked col-profile 30%/scol 0.5% of the time (verified).
Both eqsc and spv are full planes because ReduceMax rejects bool, so a float position-plane must
materialise before reducing to the tiny row/col profiles. colf is irreducible (box values + bbot).

## OPEN ANGLES (re-attack backlog)
- Drop eqsc OR spv (→ ~8.2k, ~16.0): recover smag from the 1-D contiguous run-length of sc-rows
  (sprite occupies a gap-free row band srow..srow+smag·size-1; conway fills every row) instead of a
  2-D masked col-profile. Needs an ONNX "first-run-length-from-srow" on a [1,1,30,1] vector — awkward
  but plane-free. srow needs NO mask (box never above sprite); only scol/spcmax need the top mask.
- CROP-TO-ACTIVE not usable: brow/scol data-dependent → symbolic-dim Slice trap.

## INSIGHT (transferable)
⭐ A "two-objects-at-data-dependent-positions + spatial-correspondence" task is NOT a wall when both
objects reduce to closed-form scalar bboxes: build ONE colour-index plane, recover each object's
(row,col,size,scale) via ReduceMax/Min on tiny 1-D profiles, then read both objects with chained
data-dependent `Gather(colf, rows, axis=2)`→`Gather(·, cols, axis=3)` into a 4×4 block. Magnification
`smag = colspan/size` via exact float Div+Cast (no Mod). The col extent here is gen-guaranteed in-grid
(`scol ≤ W-size·smag-1`) so smag never needs clip logic — read the generator's coord bounds to find the
unclipped axis. Off-grid gather reads land on bg=0 (≠sc) so clipped bottom blocks self-zero — no H/W needed.
