# task268 — aba27056

**Rule:** A single hollow rectangular box of one random colour sits near one grid
edge with a NOTCH (a gap) on the edge facing the grid interior. The output draws
the box frame in its colour AND a yellow(4) "fountain" erupting from the notched
edge: a yellow interior fill, a vertical yellow band running through the inner
columns from the box to the far edge, and two yellow diagonal "arms" spreading
outward as they travel away from the box. Geometry = closed form of the box bbox +
notch direction, under flip/transpose orientation (8 params: size,wide,tall,col,
row,color,flip,xpose). Canonical frame (notch on top, box at bottom), bbox rows
[r0,r1] cols [c0,c1]: YELLOW = band(c0+2<=cc<=c1-2 & rr<=r1-1) OR interior OR
left-arm(cc-rr==c0+2-r0 & rr<=r0-1) OR right-arm(cc+rr==c1-2+r0 & rr<=r0-1);
FRAME = box-perimeter AND NOT YELLOW. dir->(flip,xpose): top(0,0) bot(1,0)
left(0,1) right(1,1).
**Current:** 15.15 pts, ext:vyank6322 (154 nodes), mem 18808, params 57
**Target tier:** detection/B — diagonal arms couple r&c (not row⊗col separable, so
not tier A); but fully closed-form, so beat the public net by shrinking planes.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | canonicalize via effective ramps, [1,10,10,10] Where routing, fp32 | det | 66980 | 81 | 13.89 | — | works but heavy |
| 2 | route via single value plane L + Equal; 1x1 Conv for colf | det | 19830 | 63 | 15.10 | — | tied |
| 3 | fp16 all working planes (entry fp32->cast fp16) | det | 15158 | 63 | 15.37 | — | +0.22 |
| 4 | FRAME = occ AND NOT yellow (drop perimeter geometry, ~17 planes) | det | 13558 | 63 | 15.48 | 200/200 | +0.33 |
| 5 | drop top-edge detection + dead `size` node | det | 13243 | 63 | 15.50 | 200/200 | +0.35 ADOPT |

## Best achieved
15.50 @ mem 13243 params 63 — adopted? recommend Y. Beats prior 15.15? Y (+0.35).

## Irreducible-floor analysis
Dominant intermediate is the padded value plane Lp[1,1,30,30] fp16 = 1800B
(irreducible: the final Equal(Lp, arange) must produce the [1,10,30,30] output, so
one 30x30 plane is required; fp16 halves the 3600B fp32 cost). The rest are ~25
[1,1,10,10] fp16/bool planes (100-200B) on the 10x10 active window. Cannot drop
below the one 30x30 plane without leaving the per-cell closed-form regime.

## Key structural wins
- `size` IS recoverable even though grids are zero-padded at top-left: IN-GRID bg
  cells carry ch0=1, OFF-GRID cells are all-zero, so the grid extent = the
  bounding extent of ReduceMax-over-channels. This was the apparent blocker
  (output depends on `size` for far-edge clipping) and it dissolved.
- EFFECTIVE-RAMP canonicalization: instead of transposing/flipping full output
  planes (4-way orientation select), fold flip+transpose into per-cell coordinate
  planes ER,EC (ER=flip? size-1-a : a; a/b=xpose? swap). All canonical predicates
  then evaluate directly in the OBSERVED frame — zero plane transposes.
- FRAME = (input box occupancy) AND NOT yellow — the output frame is just the
  input's box cells minus the cells the yellow fountain overwrites. Killed the
  entire perimeter geometry (~17 bool planes, -1600B).

## OPEN ANGLES (further reduction backlog)
- The diagonal-arm Equal planes (ecmer/ecper) + band/interior predicate chains are
  ~15 bool/fp16 10x10 planes. A banded single-plane encoding (pack band+arms into
  disjoint magnitude bands of one accumulation, read by thresholds) could merge
  several into one fp16 plane.
- The three edge-count ReduceSums (notch detection) cost 3 fp16 10x10 planes; a
  single packed count (e.g. 100*botfull+10*lftfull+rgtfull) read by bands could
  collapse to one plane.

## INSIGHT (transferable)
⭐ "Output depends on grid SIZE but the grid is zero-padded" is NOT a blocker:
in-grid bg cells set ch0=1 while off-grid cells are ALL-zero, so `size` =
bounding extent of ReduceMax-over-channels. Recover it and clip with it.
⭐ For flip/transpose orientation-equivariant tasks, fold the orientation into
PER-CELL effective coordinate planes (ER,EC) rather than transposing full output
planes — predicates then build directly in the observed frame, no 4-way select.
⭐ When the output's solid/frame region is a subset of the INPUT's marked cells
minus an overlay, compute frame = input-occupancy AND NOT overlay instead of
reconstructing the frame geometry from scalars.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.032)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 3356 (mem 3300 + params 56), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
