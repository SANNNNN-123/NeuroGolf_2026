# task190 — 7ddcd7ec (extend diagonal seeds of a 2x2 box into full 45° rays)

**Rule:** A solid 2×2 box of one COLOUR k sits at (row,col) on a fixed 10×10 grid.
Up to three of the four diagonal corners carry a single SEED pixel (same colour) one
cell out from the matching box corner (d0 up-left (row−1,col−1); d1 up-right
(row−1,col+2); d3 down-left (row+2,col−1); d2 down-right (row+2,col+2)). The INPUT
shows box + present seeds (one cell each); the OUTPUT extends each present seed into a
full 45° ray out to the grid edge, box preserved.
**Current:** 16.69 pts, custom:task190 (closed-form scalar recovery + Equal output), mem 3981, params 85.
Prior 16.26 (custom conv-ray version).
**Target tier:** B (colour-index label-map + final Equal). Not S/A: rays are data-dependent
45° diagonals (r−c==Dmain / r+c==Aanti) clipped to a half-plane — not a fixed per-cell
permutation (S) and a single diagonal is not a row⊗col separable rectangle (A).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf30 1×1 conv entry + full-plane flag ANDs | B | 9353 | 82 | 15.85 | — | 3600B colour plane + 4×400B flags |
| 2 | ch0-slice occupancy, ReduceMax-k (drop colour plane) | B | 5833 | 85 | 16.31 | — | killed the 3600B entry |
| 3 | flags via per-row occupancy profiles (od/oa→ReduceMax cols) | B | 5193 | 85 | 16.43 | — | 1600B flag planes → 2×200B |
| 4 | Where for od/oa and L (drop ondiagf/onantif/fillf casts) | B | 4593 | 85 | 16.55 | — | −600B |
| 5 | box row/col via 1-D profiles (drop two 9×9 product planes) | B | 4341 | 85 | 16.60 | — | −252B |
| 6 | ondiag/onanti via [1,1,1,10] target vector (drop dval/aval) | B | **3981** | **85** | **16.69** | **200/200** | WIN +0.43 |

## Best achieved
**16.69 @ mem 3981 params 85 — fresh 200/200.** Beats prior 16.26 by **+0.43**. Adopted? **N** (build-only).

## Irreducible-floor analysis
mem 3981 dominated by: the 900B uint8 [1,1,30,30] padded label `L30` feeding the FREE Equal
(must write all 30×30 — the cheapest 10-ch output route); the 400B fp32 ch0 slice (Slice
preserves the f32 input dtype → 100 elems × 4B, the cheapest single-plane occupancy source);
three fp16 10×10 planes occf/od/oa (600B; od/oa are occupancy∩diagonal, irreducibly 2-D
because a 45° diagonal is not separable); L16 (200B); the two 9×9 box-conv planes bc/btl
(324B). Everything else is scalars / tiny 1-D vectors.

## OPEN ANGLES (re-attack backlog)
- ch0 400B: occupancy is fp32 only because Slice keeps the input dtype; no cheaper single-plane
  source found (channel-1..9 slice is 3600B; colour conv is 3600B).
- od/oa 400B: the per-row diagonal extract odrow[r]=occ(r, r−Dmain) is a Gather (data-dependent
  index → symbolic-dim trap risk); the Where+ReduceMax full plane is the safe form.
- L30 900B is the Equal-route floor; an And-broadcast output (ch0=ingrid&¬fill, chk=fill) needs
  a [1,10,30,30] bg one-hot intermediate (9000B) — strictly worse.

## INSIGHT (transferable)
⭐ **A "seed → grow ray to edge" task is closed-form, not a flood/connectivity wall:** the full
ray = a fixed 45° diagonal (r−c==Dmain or r+c==Aanti, both scalars from the 2×2-box top-left)
clipped to a half-plane (rows ≤ row−1 or ≥ row+2; the box rows split the line). Ray-present
flags are `ReduceMax(occ & ondiag & half) > 0` — scalar bools with NO Gather, reusing the same
masks. The box (the diagonal cells row,col & row+1,col+1) sits in NEITHER half so it never
fires a phantom ray.
⭐ **Build `Equal(rr−cc==D)` as `Equal(rr[1,1,H,1], (cc+D)[1,1,1,W])`** — fold the scalar into a
[1,1,1,W] target row vector so the Equal broadcasts straight to the 2-D bool, eliminating the
fp16 `dval=rr−cc` / `aval=rr+cc` full planes (−400B here).
⭐ **`Where(mask, occf, 0)` and `Where(mask, k, 0)` replace cast+Mul** (drops the bool→f16 cast
plane each time); reduce a detection plane to 1-D row/col profiles BEFORE the index-weighted
ReduceSum to recover (row,col) scalars without two full product planes.
