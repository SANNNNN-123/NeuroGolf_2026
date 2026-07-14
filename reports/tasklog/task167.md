# task167 — 6e02f1e3

**Rule:** Input is a 3x3 grid coloured `idx+color_offset` (idx in {0,1,2}, offset=2 -> colours {2,3,4}). Output is a 3x3 grid, all background except three GRAY (5) cells whose positions are keyed ONLY on `nc = #distinct colours present` (1..3): nc==1 -> top row (0,0)(0,1)(0,2); nc==2 -> main diag (0,0)(1,1)(2,2); nc==3 -> anti-diag (0,2)(1,1)(2,0). Outside the 3x3 grid: background.
**Current:** 18.47 pts, ext:wguesdon6304, mem 595, params 88
**Target tier:** COUNT->FIXED-PATTERN (cheapest tier) — the entire output is a function of ONE scalar nc.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | nc=ReduceSum-presence count; select 1-of-3 const 3x3 patterns by nc; Pad into free output | COUNT->FIXED | 263 | 79 | 19.17 | 200/200 | ADOPTED |

## Best achieved
19.17 @ mem 263 params 79 — beats prior 18.47 by +0.70. Y.

## Irreducible-floor analysis
nc recovery requires the [1,10,1,1] per-channel ReduceSum entry plane (40B fp32 — ReduceSum rejects uint8/bool). Remaining mem is small fp32 3x3 Where/Greater planes (36B each) for the pattern select + the [1,6,3,3] uint8 one-hot. Could shave ~0.2-0.3 by trimming the z4 zero-channel constant and the fp32->bool threshold, but already deep in tier-A; not chased.

## OPEN ANGLES
- Drop the `half`+Greater threshold and `z4` (36-elem) constant: build the pattern select directly on uint8 if a uint8-branch Where can be made to load under ORT_DISABLE_ALL (it could not here — both bool-branch and uint8-scalar-cond Where failed NOT_IMPLEMENTED / INVALID_GRAPH; fp32-branch Where works).

## INSIGHT (transferable)
Under ORT_DISABLE_ALL, `Where` with BOOL branches and `Where` with uint8 branches + a scalar-broadcast bool condition both FAIL (NOT_IMPLEMENTED / INVALID_GRAPH type error). The reliable pattern: do the 1-of-N constant select with **fp32 branches**, then threshold to bool with one Greater, then Cast to uint8 only for the final Pad-into-output one-hot. Distinct-colour COUNT = `ReduceSum(input,[2,3]) -> Greater(0) -> Slice off ch0 -> Cast fp32 -> ReduceSum over channels` (offset-agnostic, generalises).
