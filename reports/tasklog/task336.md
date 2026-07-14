# task336 — d4f3cd78

**Rule:** size-10 grid with only background(0) and gray(5). A gray "container" is a
closed rectangle of gray walls (top+bottom+left+right) EXCEPT one 1-cell gap in a
single wall; `apply_gravity` puts the gap on any of the 4 walls. OUTPUT keeps the
gray walls and ADDS cyan(8): the rectangle INTERIOR is filled cyan, and a straight
"drip" of cyan flows OUT through the gap in a straight line to the grid edge. Every
change is 0 -> 8.
**Current:** 16.86 pts, separable-rect-fill + scalar-Gather gap + rank-1 drip rays, mem 3372, params 55
**Target tier:** A — output colours are a FIXED set (gray passthrough + cyan); the
fill is a rank-1 rectangle and the drip is a rank-1 line, so it is closed-form
(no flood/connectivity wall despite looking like a "fill enclosed region" task).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-cell 4-dir enclosure + iterated MaxPool directional drip spread | A | 20600 | 242 | 15.06 | 5000/5000 numpy | works but plane-heavy |
| 2 | enclosure flags + 4 non-iterative directional OR-scan drips (MatMul) | A | 6900 | 433 | 16.10 | — | leaner, still below P |
| 3 | SEPARABLE bbox interior (rowin⊗colin) + per-wall gap detect + rank-1 drips | A | 4940 | 233 | 16.45 | — | interior plane removed |
| 4 | + scalar-Gather wall lines (drop 4×200B 2-D gap planes) + drop bg gate | A | 3372 | 254 | 16.80 | — | close |
| 5 | + exclusive CumSum scans (drop SU/SL matrices) + Gather on fp32 (drop gray_f) | A | 3372 | 55 | **16.86** | 200/200, 300/300 | ADOPT |

## Best achieved
16.86 @ mem 3372 params 55 — beats P=16.52 by +0.34 (>= +0.3). Fresh isolated
200/200 (seed 20260618) and 300/300 (seed 777).

## Irreducible-floor analysis
Dominant intermediates: the 30x30 uint8 colour carrier `L` (900B, the Pad output
feeding the free `Equal->BOOL output` — irreducible for any fixed-colour-set task)
and the single 10x10 fp32 gray Slice (400B, fp32 because Slice preserves the input
dtype; this is the ONE 2-D plane). Everything else is <=100B: the cyan assembly is
~9 bool 10x10 planes (rank-1 ANDs + OR chain) and the gap detection is 40B Gather
vectors. CumSum removed the triangular-MatMul matrices (200 params -> 0); gathering
the wall line on the fp32 slice removed the fp16 gray copy. ~at floor for the
fixed-colour Pad/Equal carrier pattern.

## OPEN ANGLES (re-attack backlog)
- Route cyan into the FREE output via a nested-Where on the input (gray/bg
  passthrough) to drop the 900B uint8 carrier — blocked by Where requiring X/Y
  dtype == declared output dtype (input is fp32, output BOOL); a bool passthrough
  Or double-counts channel-0 at cyan cells. If a cheap bool one-hot of the input
  could be had for free this would drop ~900B (-> ~18.4).
- Slice the gray plane to the true bbox extent (< 10x10) when the generator bound
  allows — marginal (active region is already the size-10 grid).

## INSIGHT (transferable)
⭐ A "fill the enclosed region + drip out the gap" task is NOT a flood/connectivity
wall: it is a SEPARABLE rectangle interior (rowin⊗colin from 1-D gray prefix/suffix
profiles) UNION a small number of RANK-1 drip lines. The drip is a straight ray, so
recover the gap's wall line by a SCALAR Gather (wall index = Σ one-hot·ramp) and read
the gap as a 1-D (line < 0.5) & span — this kills every per-wall 2-D plane. Two
further levers stacked cleanly: (1) exclusive/`reverse` CumSum gives strict
prefix/suffix-OR with ZERO matrix params (vs triangular MatMul's 100 params each;
CumSum needs fp32, rejects fp16); (2) run the wall-line Gather directly on the fp32
input Slice to avoid keeping a separate fp16 gray copy. The interior/drip terms are
inherently bg-only (the input interior and gap-line are empty), so the final `&bg`
gate is provably redundant — drop it (verified 0/10000).

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 1188/17.30; fill_weights [13,36]=full rank 13 (no low-rank), scatter_indices [2,52,4] clear-pass required for one-hot validity. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).
