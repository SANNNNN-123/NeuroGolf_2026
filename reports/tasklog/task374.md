# task374 — ea32f347

**Rule:** A 10×10 grid holds 3 gray (color 5) line segments, each horizontal (up=0) or
vertical (up=1), of DISTINCT lengths sampled from range(2,10). Lines are kept isolated
(parallel lines gap ≥2; mixed-orientation case separated too) so each gray cell belongs to
exactly one line and no two lines are adjacent. Output recolors each line by its LENGTH RANK:
shortest→2, middle→4, longest→1.
**Current:** 15.52 pts (prior) → custom 16.28 pts, label-map B, mem 6064, params 79
**Target tier:** B — output color per cell is a deterministic per-cell rule but requires a
GLOBAL comparison of the 3 line lengths (rank), so not single-Conv (S) and not row⊗col
separable (the H/V membership couples r&c). B (label-map + final Equal) is the admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-cell run-length via product chains (2 axes × 2 dirs) | B | 28304 | 431 | 14.73 | — | too many full-canvas planes |
| 2 | neighbor-classify H/V cells; length = per-row/col sum of membership; min/max rank | B | 6464 | 79 | 16.21 | 200/200 | win |
| 3 | + base-4 Where (drop not_min/not_max/mid bool planes) | B | 6064 | 79 | 16.28 | 500/500 | adopted candidate |

## Best achieved
16.28 @ mem 6064 params 79 — beats prior 15.52 by +0.76. Fresh 500/500.

## Irreducible-floor analysis
Dominant intermediate is the 30×30 uint8 label plane L30 (900B, the standard B-tier carrier for
the final `Equal(L, arange)→bool output`). The rest is ~10 fp16 [1,1,10,10] (200B) working
planes for gray mask, the 4 directional shifts, neighbor sums, hcell/vcell membership, and
Lcell. The 900B carrier is irreducible without dropping out of B (the rule needs a per-cell
label, not a separable rect). Could shave ~400-800B more by fusing the H/V membership ops, but
already well past the +0.3 bar.

## OPEN ANGLES (re-attack backlog)
- Fuse hcell/vcell: use the 2L−2 reduction trick (sum of g·(gL+gR) = 2·len−2) to skip the
  Greater+Cast threshold per direction (~2 planes saved).
- Recover length per-row/col entirely in 1-D and broadcast a single fp16 vector, avoiding the
  full-canvas Lcell Mul/Add planes.

## INSIGHT (transferable)
⭐ "recolor isolated lines by LENGTH RANK" is NOT a run-length-scan task: because lines are
isolated, a horizontal line occupies one row and its length = Σ_row(gray ∧ has-horizontal-gray-
neighbour) — a per-cell NEIGHBOUR classification (2 shifts + Or + And) broadcast from a tiny 1-D
per-row/col length vector. This replaces O(W) product-chain run-length planes (28k mem) with ~6
cheap planes (6k mem). Distinct lengths ⇒ rank without sorting: color = (L==global-min)?short :
(L==global-max)?long : middle. Base-the-Where-chain-at-the-common-case trick (start label=4,
override min→2/max→1) drops 4 bool intermediates vs an explicit "and-not-min-and-not-max".

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2262/17.26; output5 [1,5,10,10] bool 500B beats 900B pad-then-Equal; gray_crop_f 400B forced-fp32; fp16 counts already minimal. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).
