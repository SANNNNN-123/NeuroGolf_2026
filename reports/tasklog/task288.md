# task288 — b8cdaf2b

**Rule:** size = neck + 2*shoulder; INPUT has only the bottom two rows filled (row s-1: shirt on each `shoulder`-wide corner + `neck` antenna cells in the middle; row s-2: `neck` shirt cells in the middle). OUTPUT keeps the input and ADDS two antenna-coloured 45° diagonals rising out of the shoulders: left `r-c == D` (D = s-2-shoulder), right `r+c == S` (S = D+s-1), each bounded to `r < s-2` and in-grid `c < s`. shoulder = (s-neck)/2 so D = (s+neck)/2 - 2.
**Current:** 16.06 pts, int32 RmC/RpC double-plane build, mem 5466, params 2137
**Target tier:** A — closed-form separable diagonal stamp routed into the FREE Where output; no per-cell colour-index plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp16 ramps + Equal | — | — | — | — | — | fp16 Equal unsupported (opset 10 INVALID_GRAPH) |
| 2 | fp32 ramps + Equal | — | — | — | — | — | fp32 Equal also unsupported (opset 10 Equal = int/bool only) |
| 3 | int32 ramps RI[1,1,30,1]/CI[1,1,1,30], Equal(RI-D,CI) & Equal(S-RI,CI) | A | 5136 | 64 | 16.44 | 200/200 | ADOPT |

## Best achieved
16.44 @ mem 5136 params 64 — beats prior 16.06 by +0.38 (≥+0.3 ✓).

## Irreducible-floor analysis
Dominant intermediate = the 30×30 bool diagonal masks. Five are materialised
(leftD, rightD, diag=Or, m1=And rowbound, mask=And colbound) = 5×900 = 4500B.
Two Equals are irreducible (left and right diagonals are distinct r∓c lines);
the Or + two bound-ANDs each add one 30×30 bool plane. The win over the prior
build was REMOVING the two int32 [1,1,30,30] RmC/RpC PARAMETER planes (900 elems
each → 2137→64 params) by testing r-c / r+c with two 1-D int32 ramps that
broadcast through the Equal. Equal must be int (opset-10 Equal rejects float16
AND float32), so the ramps and all scalar arithmetic are int32 (counts are
exact integers).

## OPEN ANGLES (re-attack backlog)
- Active canvas is always ≤9×9 (size ∈ {3,5,7,9}) anchored top-left. Building the
  masks on a 9×9 canvas (81B each) then Pad→30×30 could cut mask memory ~10×,
  BUT Pad rejects bool so it needs cast→int→Pad→recast, adding planes; net win
  uncertain (~5 small planes + 2 casts vs 5×900B). Worth a measured try for tier-A+.
- Collapsing the Or+two ANDs into fewer 30×30 planes (no obvious 4-plane route
  since the two Equals are independent and bounds differ per-diagonal).

## INSIGHT (transferable)
Two diagonals `r-c==D` / `r+c==S` are SEPARABLE: build them with two 1-D int32
ramps RI[1,1,30,1] and CI[1,1,1,30] via `Equal(RI-D, CI)` / `Equal(S-RI, CI)`
which broadcast straight to the 30×30 bool mask — NEVER materialise an int32
[1,1,30,30] RmC/RpC parameter plane (saves ~1800 param elems). ⚠️ opset-10
`Equal` accepts ONLY int/bool — both float16 AND float32 Equal raise
INVALID_GRAPH under ORT_DISABLE_ALL, so diagonal-membership arithmetic must run
in int32 (pixel counts and coords are exact integers anyway).
