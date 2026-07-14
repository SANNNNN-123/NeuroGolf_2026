# task320 — ce9e57f2

**Rule:** Four bottom-anchored vertical red(2) bars in odd columns 1,3,5,7 of a
width-9 grid (height = max bar length + 1). For each bar of length L, recolour
its bottom floor(L/2) cells to cyan(8); top cells stay red(2). Background 0
stays 0. Output is the same-size grid; off-grid cells are all-zero (no channel).
**Current:** 16.81 pts (public net)
**Target tier:** A — fully separable closed-form per column, no flood-fill.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice ch2 11x9; below=triu MatMul; half via Mod; 10ch slice for in-grid | A | 7005 | 287 | 16.10 | — | passes, in-grid slice too heavy |
| 2 | in-grid row mask = rowhasred OR row0 (drop 10ch slice) | A | 3034 | 288 | 16.89 | — | +0.08 only |
| 3 | fold 2*below+1 into ONE matmul B (2 below-diag, 1 on-diag); cyan iff 2*below+1<L | A | 2440 | 286 | 17.09 | — | +0.28 |
| 4 | L per col via ReduceSum(axes=2) broadcast (drop 121-param ones matrix) | A | 2260 | 165 | 17.21 | 200/200 | ADOPT +0.40 |

## Best achieved
17.21 @ mem 2260 params 165 — beats prior 16.81 by +0.40. Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate is the 30x30 uint8 label carrier L (900B) feeding the FREE
Equal->bool output — irreducible because the 2-vs-8 distinction is per-cell (not
a separable rect), so a per-cell 30x30 carrier is required; uint8 is the minimal
dtype. Remaining ~1360B is the small 11x9 fp16 work chain (R / t=B@R / masks) and
the 396B fp32 red entry slice (the one unavoidable fp32 readout).

## OPEN ANGLES
- Replace 121-elem matmul B with a CumSum-suffix-sum (no matrix params) — would
  cut params but likely add an op + similar mem; marginal.
- Try emitting via three padded bool channel-masks reusing one carrier — almost
  certainly worse than the single uint8 900B carrier.

## INSIGHT (transferable)
⭐ "recolour the bottom floor(L/2) of a bottom-anchored run" is closed-form &
separable: cyan iff `2*below + 1 < L` (integer identity for below < floor(L/2)),
and BOTH the ×2 and the +1 fold into ONE strictly-below-triangular MatMix matrix
with the diagonal set to 1 (B@R = 2*below + R[r,c] = 2*below+1 at red cells) —
kills the entire half = (L - L%2)/2 Mod/Sub/Mul chain. Per-column total L is a
ReduceSum(axes=[2]) that broadcasts over rows (no ones-matrix). In-grid row mask
for a data-dependent height recovers from the run itself (rowhasred OR row0)
instead of a 10-channel slice.
