# task030 — 1caeab9d

**Rule:** Three identical-shape clusters, one per colour in (1,2,4), are placed at
positions (megarows[i], megacols[i]) on a width-10, height-{5,10} background-0 grid.
The OUTPUT moves every cluster VERTICALLY so its top row aligns to megarows[0] (the
colour-1 cluster's row), keeping each cluster's COLUMNS unchanged. Equivalently per
colour k: out_k = in_k shifted DOWN by delta_k = minrow(colour1) − minrow(colour k)
rows; columns untouched. The three colours occupy DISJOINT columns (verified 5000
fresh: no column ever carries two colours), so a plain Sum combines them. Colour 1
has delta 0 (it is the alignment target) → never moves.
**Current (prior):** 16.28 pts.
**Target tier:** B (per-colour data-dependent vertical shift = row-axis boolean
MatMul). Not S: the shift amount is a global aggregate (per-colour min-row), no fixed
conv/permute. Not row⊗col separable A: each cluster is a 2-D shape, and the shift
couples nothing across columns but the per-colour 2-D plane must be materialised.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 3 colour slices, per-colour shift MatMul S@in, Where-priority uint8 label, in-grid row mask, Pad, Equal | B | 5484 | 84 | 16.375 | 266 stored | works |
| 2 | fold colour into S (Where), variadic Sum→colour-index, single uint8 cast | B | 5384 | 81 | 16.394 | — | trim |
| 3 | colour-1 delta==0 → drop its S+MatMul (use slice directly) | B | 4840 | 80 | 16.499 | — | big trim |
| 4 | Sum(s1,s2,s4) one op; uint8 in-grid mask (cast first, Where on uint8) | B | **4640** | **80** | **16.540** | **200/200, 500/500** | FINAL |
| 5 | OneHot shift-matrix (drop Equal+Where) | B | 5140 | 102 | 16.436 | 266 | WORSE (rank-5 Sm + Reshape plane) |
| 6 | Gather row-shift (drop S matrix) | B | 5452 | 83 | 16.381 | 266 | WORSE (validity Where + colour Mul cost more) |
| 7 | rebuild #4 + ArgMax top_k + row-9 height scalar | B | **4425** | **87** | **16.586** | **200/200** | **ADOPTED** |

## Best achieved
**16.586 @ mem 4425, params 87 — 266/266 stored, isolated fresh 200/200.**  Written to
src/custom/task030.py.  Beats prior 16.28 by **+0.308 (clears the +0.3 bar)**.
(Prior session reached 16.540 @ 4640 but self-gated as MARGINAL and never saved the file;
two extra tricks pushed it over: ArgMax-of-presence for top_k (drops the Where/ReduceMin +
BIG sentinel) and a 40B background row-9 strip + `r<5 OR height==10` for the in-grid mask
instead of a 120B ReduceMax(axes=[1,3]) profile.)

## Irreducible-floor analysis
Three fixed costs dominate and resist removal:
- **900 B uint8 [1,1,30,30] Pad** — the label map feeding the FREE final Equal must
  span the 30×30 output; uint8 (1 B) is the smallest dtype, Equal-at-10×10-then-pad-bool
  is rejected (Pad rejects bool). This is the Tier-B floor (matches task250/task035).
- **1200 B = three fp32 [1,1,10,10] channel slices** (colours 1,2,4). Slice preserves
  the fp32 input dtype; the colours {1,2,4} are NON-CONTIGUOUS so they cannot be sliced
  as one [1,3,…] block (slicing 1:5 = 1600 B includes empty ch3; concat-then-batch costs
  even more). Each plane is genuinely needed (min-row + shift source).
- **shift machinery ~1400 B**: for the two moving colours, per colour Sb bool(100) +
  coloured S fp16(200) + in16 cast fp16(200) + sh fp16(200) = 700. fp16 beats fp32 MatMul
  (S/sh would be 400 each) and ORT forbids mixed-dtype MatMul so the in16 cast is forced.
The remaining ~200 B (Lf sum, Lu/Lw uint8, anyR_full 120 B in-grid row mask, scalar
min-rows) are each ≤120 B and minimal.

## OPEN ANGLES (re-attack backlog)
- **Eliminate one fp32 slice.** Blocked: colours {1,2,4} non-contiguous; every cheaper
  multi-channel grab (Gather axis1, 5-channel slice) is larger. Seems closed.
- **Shrink the 900 B Pad.** The label-map→Equal route is the Tier-B floor; a separable
  row⊗col free-output route is impossible (2-D cluster shapes). Closed for Tier B.
- **Per-column single-shift via batched MatMul** (column on batch axis): the per-column
  shift tensor is [W,W,W] (huge) or needs GatherND — net worse. Not pursued past sketch.

## INSIGHT (transferable)
⭐ **A per-colour (or per-object) PURE VERTICAL shift to a common target row is a
row-axis boolean MatMul `sh = S @ in`, S[R,r]=colour·(r+delta==R)** — fold the colour
straight into S via `Where(Equal(srcramp+delta, outramp), colour, 0)` so the MatMul
output is already the colour-carrying plane and disjoint-column colours combine with a
single variadic Sum + one uint8 Cast (no per-colour bool/Where chain). The alignment
TARGET colour has delta 0 → skip its S and MatMul entirely and feed its raw slice into
the Sum (here −544 B / +0.10 pt).
⭐ **OneHot and Gather are NOT cheaper than Equal+Where+MatMul for a shift matrix**:
OneHot leaves a rank-(n+1) intermediate that needs a Reshape (extra full plane), and a
Gather-shift needs an explicit validity mask + colour Mul — both measured strictly worse.
⭐ **off-grid rows (height 5 vs 10 on a 10×10 canvas): the in-grid row mask = anyR>0
where anyR = ReduceMax(input, axes=[1,3])** (120 B [1,1,30,1]); gate the uint8 label to
sentinel 10 there so the final Equal emits all-zero off-grid (the Pad only handles
rows/cols ≥ canvas, not the short-grid interior rows 5–9).
