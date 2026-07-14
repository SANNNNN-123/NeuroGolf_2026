# task196 — 810b9b61

**Rule:** Several axis-aligned rectangle OUTLINES drawn in blue (=1) on black (=0),
size ∈ {9,12,15}, boxes pairwise 8-separated. Some boxes have a single perimeter cell
knocked out to black (a "gap"). OUTPUT: each box outline is recoloured GREEN (=3) iff it
is a COMPLETE closed rectangle AND wide≥3 AND tall≥3; otherwise it stays BLUE. Flat
(w==1/t==1), thin (w==2/t==2, no interior) and gapped boxes stay blue. Black cells
(incl. the gap) stay black.
**Current:** 14.62 pts (public dilation net), mem 32175, params 66
**Target tier:** detection/connectivity — closure is a per-box GLOBAL property
(a single gap must un-fill the whole interior), so it needs propagation. NOT a closed-form
or separable tier-A/B form; the bounded per-direction enclosure (task125 lever) FAILS here
(314/400) because a gap on one wall still leaves interior cells seeing blue in all 4 dirs.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | parity of L-wall crossings (task204) | — | — | — | — | 11/400 | FAIL (shared rows merge box walls) |
| 2 | bounded per-dir enclosure reach=4/5 (task125) | — | — | — | — | 314/400 | FAIL (gap leaks past local window) |
| 3 | exterior flood (8-conn MaxPool) on 30-canvas | det | — | — | — | 400/400 | correct but big |
| 4 | + CROP-TO-15 + fp16 flood + uint8 label/Pad | det | 16875 | 266 | 15.25 | 200/200 | ADOPT-candidate |

## Best achieved
15.25 @ mem 16875 params 266 — beats prior 14.62 by +0.63. Fresh 200/200 (isolated).

## Irreducible-floor analysis
Dominant cost = the inward exterior flood: 11 iterations × (MaxPool3x3 + Mul-gate) = 22
full 15×15 fp16 planes ≈ 9900B. Irreducible because:
- closure is global → flood is required (convergence needs UP TO 11 iters; measured max over
  5000 fresh = exactly 11, matching the 15×15 8-conn diameter), so iters can't drop below 11.
- 8-conn (3×3) kernel is REQUIRED: a 5×5 MaxPool leaks the flood THROUGH 1px blue walls
  (it jumps over the wall before the per-step gate can block it) → 0/1000.
- MaxPool needs float; fp16 (2B) is the dtype floor (ORT MaxPool rejects uint8/int8/bool).
- canvas is fixed 15 (size ∈ {9,12,15}); data-dependent crop trips the symbolic-dim trap.
Tail (label + Pad→30) already collapsed to one 30×30 uint8 plane (900B) via the free
Equal(L, arange) output.

## OPEN ANGLES (re-attack backlog)
- **bad-flood instead of exterior-flood**: the public net dilates "bad box" over the blue
  loop and converges in ~6 iters (loop half-perimeter ≤ ~8 with diagonals) vs 11 here.
  Halving flood planes would save ~4500B (~+0.27). Blocker: needs a robust LOCAL "bad seed"
  (gap-adjacency + flat/thin detection) — gap detection from a single blue cell's
  neighbourhood is the hard part; worth a focused attempt.
- recover per-box closure as a scalar via per-component reductions (boxes are 8-separated)
  + broadcast — but broadcasting still needs the same flood, no net win.

## INSIGHT (transferable)
⭐ **The one-hot OFF-GRID representation is NOT black.** `convert_to_numpy` only sets a
channel for IN-GRID cells, so off-grid (beyond size×size within the 30-canvas) cells are
ALL-ZERO — ch0(black) is **0** there, not 1. Any flood/enclosure that treats "ch0==1" as
the floodable background SILENTLY BREAKS at the active-grid edge (a box's edge-touching
side never floods). Flood through `notblue = 1 - ch1` (in-grid bg ∪ off-grid) instead, and
mask the OUTPUT to in-grid via `ch0 OR ch1` with an off-grid sentinel (99) into the Equal.
This bit hard: numpy tests on a fully-painted 30-canvas passed 400/400 but the real harness
failed 21/266 until the representation was matched.
⭐ FILL-ENCLOSED / closed-loop detection: crop to the gen-bounded active region, run the
flood in fp16 with MaxPool3x3 (0 params, 8-conn respects 1px walls), interior = notblue−reach,
then green = blue·dilate8(interior) (corners reach centre interior diagonally → 8-conn dilate,
not 4-conn). Build the {0,1,3} label on the CROP and Pad once to 30×30 with a 99 sentinel.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-1/2 opus agents; occupancy/max-semiring reductions or sub-400B u8 banks). Do not re-attempt without a new mechanism.
