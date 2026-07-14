# task119 — 508bd3b6 ("bounced diagonal ray" reconstruction)

**Rule:** Canonically a solid RED(2) wall fills the top `depth` rows. A 45° V-ray
lives below it: `row(c) = depth + |mid - c|` — comes down, bounces off the wall
edge at vertex `(depth, mid)`, goes back down. Ray cells are CYAN(8) for the
first `shown` columns (c < shown), GREEN(3) elsewhere. The INPUT shows only the
wall + cyan stub (green ray erased to bg); the OUTPUT redraws the full ray
(green) keeping cyan + wall. Then `flip` (h-mirror) + `gravity` (transpose /
row-reverse) rotate the whole figure into one of 8 orientations (wall a solid
band on any of the 4 sides). The full ray = union of two 45° diagonals through
the vertex `(r+c==a) OR (r-c==b)`, restricted to non-wall cells on the far side
of the vertex. Grid always 12×12.
**Current:** 16.12 pts, custom:task119 (geometric recovery + label-map+Equal),
mem 7088, params 106. Prior 15.37 (public ext:vyank6322).
**Target tier:** B (label-map + final Equal). Not S/A: output cells are two
data-dependent DIAGONALS (r+c==a OR r-c==b) clipped by a data-dependent wall
half-plane — not a fixed per-cell linear/permutation function (S) and not
row⊗col separable (A: a single diagonal is not a row-cond × col-cond product).
The reconstruction itself is fully closed-form (no detection floor): the ray is
recovered from the cyan stub + wall as exact scalars.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-fp32 12×12 planes, label-map+Equal | B | 22900 | 369 | 14.95 | 266/266 | correct but mem too high |
| 2 | fp16 per-cell planes | B | 14134 | 369 | 15.42 | — | +0.05 only |
| 3 | nrm/par/diag from 1-D ramps (drop 2-D coord inits) | B | 11494 | 105 | 15.64 | — | +0.27 |
| 4 | Equal-based masks (drop squared planes), merge near/far+wall | B | 8758 | 105 | 15.91 | — | +0.54 |
| 5 | uint8 chained-Where label (no fp16 colour planes) | B | 7894 | 109 | 16.01 | — | +0.64 |
| 6 | keep CY/RE fp32 (drop cast), par_near no-division | B | **7088** | **106** | **16.12** | **400/400** | WIN +0.75 |

## Best achieved
**16.12 @ mem 7088 params 106 — fresh 400/400** (isolated in-memory, all 8
orientations). Beats prior 15.37 by **+0.75**. Adopted? **N** (build-only;
main adopts via `python -m src.adopt 119`).

## Irreducible-floor analysis
Memory breakdown (7088): the 900 B uint8 [1,1,30,30] padded label `L` feeding
the FREE Equal (irreducible — Equal must write all 30×30); ten fp16 12×12 planes
(2880 B: nrm, par, RpC, RmC, two cyan-extreme `Where` masks, atmax fp16
plane/par/count); eleven bool 12×12 masks (1584 B: onA/onB/ondiag, cyB, atmin/
atmax, near_side, wallB, blocked, open_cell, greenmask); two fp32 12×12 slice
planes (1152 B: cyan CY and red RE channels straight off the one-hot input). The
diagonal masks (onA/onB) and `nrm` are irreducibly 2-D (a single diagonal is not
separable). Everything else (vertex scalars, slope, a/b) is already scalar.

## OPEN ANGLES (re-attack backlog)
- The two fp32 CY/RE slice planes (1152 B) are pure input reads; a Conv that
  emits cyan-and-red masks in one fp16 [1,2,12,12] plane could shave ~600 B.
- Several bool masks could fold (e.g. compute `greenmask` and the priority label
  in fewer ops) for maybe ~0.1 pt — diminishing returns vs complexity.
- Tier-A long-shot: blocked — a single 45° diagonal `r+c==a` is not a
  rowcond⊗colcond outer product, so the separable form cannot express it.

## INSIGHT (transferable)
⭐⭐ **An apparent ray-bounce / reflection "detection" task is a CLOSED-FORM
geometric reconstruction, not a detection floor.** The full ray = union of two
45° diagonals `(r+c==a) OR (r-c==b)` through a vertex; recover `(a,b)` as exact
scalars: the cyan stub's slope in (parallel vs wall-normal) coords is always ±1,
so `slope = (par_near − par_far)/(nrm_near − nrm_far)` and `vertex_par =
par_near + slope·(band − nrm_near)` (vertex sits on the wall inner edge). 0 /
40000 fresh.
⭐ **8 flip/gravity orientations collapse to a per-side coordinate blend:** detect
which full edge is all-red (4 scalar flags from edge sums), define wall-NORMAL
and wall-PARALLEL coordinate fields as a flag-weighted sum of 1-D ramps, and all
downstream geometry is orientation-agnostic — no per-orientation code path.
⭐ **`band = ReduceSum(red)/12`** recovers wall thickness in every orientation
(red total is band×12 regardless of side) — a count, not a spatial scan.
⭐ **nmin (cyan cell nearest the wall) is never tied → par_near needs no
division;** only nmax (farthest) can tie in the symmetric vertex-visible case.
Build nrm/par/diagonals by BROADCASTING 1-D [1,1,12,1]+[1,1,1,12] ramps so each
2-D plane materializes exactly once; keep input-channel slices fp32 (Greater/
ReduceSum accept fp32) to skip an extra cast plane.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.001)
**Mechanism (op-census diff):** Dropped one probe index (`side_rev_probe_idx` [1,1,4]→[1,1,3]) and one `Not` node. −1 param, −1B.
**Old→new:** mem 1244→1243, params 121→120.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task119_pre_s10.onnx`; source `public_candidates/bobmyers7186/task119.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
