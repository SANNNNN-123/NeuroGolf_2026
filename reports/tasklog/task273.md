# task273 — af902bf9

**Rule:** 10x10 grid with 1 or 2 axis-aligned rectangles, each marked in the INPUT by its four
YELLOW(4) corners only. OUTPUT keeps the corners and paints every cell STRICTLY INSIDE the
rectangle (r0<r<r1 AND c0<c<c1) RED(2). With 2 boxes they sit in opposite (top-left / bottom-right)
quadrants, never sharing a row- or col-range. Verified 0/4000 fresh: a cell is red iff there is a
yellow corner in ALL FOUR strict quadrants around it (UL & UR & DL & DR) — the JOINT-quadrant test
(NOT the separable per-direction up/down/left/right OR, which bridges the two diagonal boxes).
**Current (prior):** 16.31 pts.
**Target tier:** A — closed-form bbox/enclosure via double triangular MatMul, no flood-fill, output
routed into the FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4-quadrant joint counts (SL@Y@{SU,SL}, SU@Y@{SU,SL}); Greater+And; uint8 pad -> bool cond; Where(red_oh,input) | A | 4400 | 429 | 16.52 | 200/200 | works |
| 2 | dedupe triangulars (only SL,SU distinct) | A | 4400 | 229 | 16.56 | 200/200 | -200 params |
| 3 | collapse Greater/And -> Min chain; let uint8->bool cast do the >0 threshold | A | 4300 | 228 | 16.58 | 200/200 | -100 mem |
| 4 | drop Where carrier: build colour-index L={0,2,4}=4Y+2redcnt; Pad SENTINEL 99; Equal->BOOL output | A | 4200 | 231 | 16.60 | 200/200 | carrier 1800->900 |
| 5 | drop redbit clip (redcnt is exactly 1 on interior) | A | 4000 | 230 | 16.650 | 500/500 | adopted-as-best |

## Best achieved
16.650 @ mem 4000 params 230 — adopted? N (per instructions). Beats prior 16.31? Y (+0.34). Fresh 500/500.

## Irreducible-floor analysis
Dominant: the single 30x30 uint8 colour-index carrier L30 (900B, the one plane the Equal one-hot
must broadcast against — can't shrink, output is 30x30; sentinel-99 pad is what let the BOOL-output
Equal route replace the 1800B Where uint8+bool pair). Then 6 fp16 10x10 spatial planes (1200B):
2 row-accumulations (SL@Y, SU@Y) feeding 4 quadrant col-products — INTRINSIC to the joint-quadrant
test (2 quadrants are insufficient: ~730/3000 fail). Entry yellow_f32 slice (400) + fp16 cast (200):
the 10->1 reduction must enter fp32 and matmul needs fp16. The 3 Min + 2 Mul + 1 Add tail (800) is
the minimal AND-of-4 + index-build.

## OPEN ANGLES (re-attack backlog)
- Fold the 4 quadrant col-products into 2 stacked matmuls Yabove@[SU|SL] -> [1,1,10,20]: same bytes,
  needs a Slice to split, no net gain unless the split is free.
- Derive Ybelow = colsum - Y - Yabove to drop one row-matmul: costs 2 Subs (400) > the 200 saved.
- Eliminate the fp16 Y plane by running the first matmul on the fp32 slice (fp32 triangulars): makes
  T1/T2 fp32 (800 vs 400) — net worse. No obvious sub-4000 path; ~16.65 is the structural floor.

## INSIGHT (transferable)
⭐ JOINT-QUADRANT enclosure (NOT the per-direction up/down/left/right OR): for corner-marked
rectangles whose interior cells are NOT row/col-aligned with any marker, "is this cell inside a box"
= a yellow exists in EACH of the 4 strict quadrants = `(SL@Y@SU)&(SL@Y@SL)&(SU@Y@SU)&(SU@Y@SL) >0`
(double triangular MatMul per quadrant). This handles multiple diagonally-separated boxes without
merging them, where the cheaper separable per-direction OR (task125 idiom) cross-talks.
⭐ When the WHOLE task uses only a few colours, build a single colour-INDEX plane and route via
`Equal(L, arange) -> BOOL output` instead of `Where(cond, oh, input)`: the carrier drops from an
1800B uint8+bool pair to ONE 900B uint8 plane. The catch on a sub-30x30 active grid: Pad OFF-GRID
with a SENTINEL (e.g. 99, not a valid colour) so Equal leaves those cells all-False (matching the
all-zero target) — padding with 0 would wrongly light channel-0 across the whole off-grid border.

## S10 (2026-07-03) — tasklog CORRECTION: net already resolved at 2109 mem
Scout re-measured live net: mem 2109 / params 51 / 17.322 pts, corner-detect + bbox-fill
(TopK(4) yellow rows → Greater/Less/And rect masks → Pad 900B → Where). The 4000-mem
quadrant-count description above is STALE (superseded net). Only 30×30 plane = the Where
cond (900B) = free-output-axis welded (un-croppable, S10 crop class). FLOOR at 2109.
