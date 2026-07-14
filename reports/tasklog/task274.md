# task274 — b0c4d837

**Rule:** Input draws a "cup": gray (5) side-walls at cols col_gap and col_gap+base-1, a gray base row, and cyan (8) water filling the bottom `water` cup-interior rows. Above the water are `space` (1..4) empty cup rows. The 3x3 output encodes ONLY the scalar `space`: out[0][0]=cyan if space>0, out[0][1] if space>1, out[0][2] if space>2, out[1][2] if space>3; all else background. space = grayrows − cyanrows − 1 (rows with gray = walls+base = space+water+1; rows with cyan = water).
**Current:** 0 pts (no manifest entry; public net P=16.00), method=scalar-tally, mem 2383, params 36
**Target tier:** A/B — output is a fixed 3x3 tally of ONE recovered scalar; no per-cell label map of the input is needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | space=grayrows−cyanrows−1 from ONE fp32 ReduceMax(input,[3]); 3x3 cyan tally via space>=T threshold; Pad→Equal | A | 2383 | 36 | 17.21 | 200/200 | ADOPTED |

## Best achieved
17.21 @ mem 2383 params 36 — beats P=16.00 by +1.21. fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate: the [1,10,30,1] fp32 row-presence reduction (1200B) — ReduceMax emits fp32 and we need per-row presence for two channels (cyan ch8, gray ch5). Slicing the two full channels [1,1,30,30] instead would cost 7200B, so the single 10-ch reduction is the cheaper entry. The padded [1,1,30,30] uint8 label map (900B) is the only other notable plane and is already minimal width. Both are at the structural floor for "recover one scalar + emit a fixed small one-hot tally".

## OPEN ANGLES (re-attack backlog)
- The fp32 rowmax could in principle drop if a single banded conv produced both cyan- and gray-row presence in a sub-30 plane, but the gain (<~0.3) is not worth the op churn at 17.21.

## INSIGHT (transferable)
"Output is a fixed-size tally encoding ONE structural scalar" tasks are tier-A regardless of input size: recover the scalar from cheap per-channel/per-row reductions, then build the tally as `Greater(scalar, threshold_const_plane)` over a tiny fixed grid routed into the FREE Equal/Where output. Here `space = grayrows − cyanrows − 1` turns a "measure the empty cup height" task into two ReduceSums of a slice of one ReduceMax — no per-cell plane of the input at all.
