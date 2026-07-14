# task226 — 941d9a10

**Rule:** 10x10 black canvas partitioned into an R×C array of blocks by fully-gray separator rows/cols (gray=5), R,C each ∈ {3,5} (2 or 4 separators/axis). Output copies the input and fills three diagonal blocks: block(0,0)=blue(1), block(R//2,C//2)=red(2), block(R-1,C-1)=green(3). Fills cover only the block interiors; gray lines and background are untouched.
**Current:** 16.53 pts (public net).
**Target tier:** B — separable row⊗col block masks → single colour-index plane → Equal into FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 weighted-sum colour plane | B | 5056 | 237 | 16.43 | — | below bar (7 fp32 10x10 planes) |
| 2 | uint8 Where chain, full-plane consts | B | 2656 | 735 | 16.87 | — | consts bloated params |
| 3 | uint8 Where chain, scalar consts | B | 2656 | 240 | 17.03 | 200/200 | ADOPT-WORTHY |

## Best achieved
17.03 @ mem 2656 params 240 — beats prior 16.53 by +0.50. Fresh 200/200.

## Key construction
- separator-row/col indicators sr/sc = (axis-ReduceSum of the 10x10 gray slice == 10).
- inclusive prefix counts via a lower/upper-triangular MatMul on the [1,1,10,1]/[1,1,1,10] indicator vectors; exclusive = inc−s, suffix = nsep−inc (no 2nd matmul).
- block0 = (inc==0); midblock index = nsep/2 EXACT (nsep always even, 2 or 4) so red = (exc==nsep/2)∧¬sep; last = (suf==0)∧¬sep.
- three filled rectangles are separable AND-broadcasts of the 1-D masks (no 2-D box planes until the final 10x10 bool ANDs).
- colour-index L10 via uint8 Where chain on disjoint regions; Pad to 30x30 sentinel 10; Equal(Lp, arange) → FREE bool output.

## Irreducible-floor analysis
Lp (uint8 [1,1,30,30] = 900B) — must be 30x30 to broadcast against the 10 colour channels in the final Equal. gray10 (fp32 [1,1,10,10] = 400B) is the slice the reductions consume. Everything else ≤100B. Mem dominated by these two; ~2656 total.

## OPEN ANGLES
- gray10 400B could drop if rowsum/colsum came straight from per-axis Convs on the input gray channel (skip the 10x10 materialisation), but the saving is small vs the 900B Lp floor.

## INSIGHT (transferable)
"Partition by all-bg/all-line separators + label diagonal blocks" is fully SEPARABLE: per-axis block index = exclusive prefix-count of the separator indicator (lower-tri MatMul), middle-block index = nsep/2 when nsep is provably even — exact, no argmax. Build the colour-index plane with a uint8 Where chain (scalar fills broadcast) on disjoint regions, NOT a fp32 weighted sum (the float casts cost 7×400B of full planes). ⭐ scalar [1,1,1,1] Where fills broadcast over a [1,1,10,10] mask at ~0 params — never materialise full-plane colour constants.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.063)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 1534 (mem 1513 + params 21), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
