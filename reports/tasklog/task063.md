# task063 — 2bee17df

**Rule:** Square size×size grid (size ∈ {10,12,14}, always even) with inward
teeth (red=2/cyan=8) along all four edges (every tooth length ≥ 1, so the
perimeter ring is fully coloured), plus scattered coloured interior cells. For
every INTERIOR row r (1..size-2) whose interior cells (cols 1..size-2) are all
background, the whole interior of that row is painted green=3; same for every
interior column with an all-bg interior. flip/transpose applied afterwards
(structure-preserving). Compact: per-row coloured count rc[r] (channels 1..9)
== 2 ⇔ only the two perimeter endpoints coloured ⇔ row free. fill =
interior_r ∧ interior_c ∧ (rowfree ∨ colfree); the Or already implies the cell
is background so no bg mask is needed.

**Current:** 16.65 pts, mem 3600, params 618 (was 16.11 / mem 7200 / params 23).
**Target tier:** A — separable row⊗col routed into the FREE Where output; the
only full-canvas tensors are 3 bool masks.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | (prior) ch0 Conv 3600B fp32 occ + 3 bool planes + bg-mask | A | 7200 | 23 | 16.11 | — | baseline |
| 1 | drop ch0/bg plane: counts via 2 channel-weighted no-pad Convs; interior via 1-D neighbour Convs; fill=intr∧intc∧(rowfree∨colfree) | A | 3600 | 618 | 16.65 | 200/200, 300/300 | ADOPTED |

## Best achieved
16.65 @ mem 3600 params 618 — adopted. Beats prior 16.11 by +0.54 (≥+0.3 Y).

## Irreducible-floor analysis
mem 3600 = three [1,1,30,30] bool masks (freeor, intr2, fillb @ 900B each) +
~900B of 120B/30B row/col profile vectors. The "row OR col" 2-D structure with
independent interior gating on each axis needs an Or plane and at least one
And plane before the Where, and the Where carrier itself — 3 bool planes is the
floor for this shape. The 3600B fp32 ch0 occupancy plane of the prior version
was fully removed: coloured counts come straight from two channel-weighted
no-pad Convs (ch0 weight 0) and the in-grid/interior masks from 1-D neighbour
Convs on those 120B profiles — no per-cell occupancy is ever materialised.
params 618 ≈ Wrow(300) + Wcol(300) + small kernels/scalars; halving them only
buys ~+0.08, not pursued.

## OPEN ANGLES (re-attack backlog)
- Collapse the 3 bool planes to 2: tried float-score outer-products
  (R[r]·intc + C[c]·intr) but each is a 3600B fp32 plane (4× a bool plane) →
  worse. No clean 2-plane bool form found for "interior_r ∧ interior_c ∧
  (rowfree ∨ colfree)".
- Shrink the two 300-elem count Convs (e.g. ReduceSum tricks) — but bg-excluded
  per-row coloured count has no 0-param reduction (ReduceSum mixes ch0).

## INSIGHT (transferable)
⭐ A "fill the all-background interior row/col" task is NOT a flood/connectivity
problem: a row is free ⇔ its bg-excluded coloured count == 2 (the two always-
coloured perimeter endpoints), recoverable as a 120B profile via ONE channel-
weighted (ch0=0) no-pad Conv — no occupancy plane. Interior-vs-border is the
3-neighbour-in-grid test (length-3 same-pad Conv on the in-grid profile == 3,
i.e. >2.5), and the Or(rowfree,colfree) condition already guarantees the cell is
background so the explicit bg mask is droppable. opset-10 caveat: Equal rejects
float, so use Less/Greater range tests on integer-valued counts.
