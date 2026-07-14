# task352 — dc1df850

**Rule:** Active grid H×W (W∈5..10, H∈{W-1,W}). Sparse coloured dots at distinct (r,c),
non-overlap guard keeps every pair ≥2 apart in Chebyshev distance. Dot colours ∈ {2 (red)} ∪
random_color(exclude=[blue,red]) = {2,3,4,5,6,7,8,9}; blue(1) NEVER appears in the input. Output:
copy every dot; for each RED dot, paint BLUE on its 8 surrounding cells (common.draw clips to the
active grid) then re-paint the red centre. Because dots are ≥2 apart, a red halo only ever covers
background cells inside the active grid (never another dot). Closed form per output channel:
ch2..ch9 = centre copy; ch0(bg) = B0c − (#red in 3×3) > 0; ch1(blue) = B0c + (#red in 3×3) − 1 > 0,
where B0c = input ch0 at centre (=1 iff in-grid background, 0 off-grid) — the load-bearing in-grid
gate that kills off-grid-on-canvas neighbours of edge reds.
**Current:** 18.186 pts, conv3x3+b (dense [10,10,3,3]=910), mem 0, params 910
**Target tier:** mem-0 single-Conv — only ch0/ch1 need a 3×3 cross-channel footprint (red→blue/bg);
ch2..ch9 are pure 1×1 copies, so the dense [10,10,3,3] over-models. Any materialised 30×30 plane
(≥900B) drops below 18.19, so the win must stay mem-0 and only shave params.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | grouped Conv group=2 (split [0..4]\|[5..9]) [10,5,3,3] | conv | 0 | 460 | 18.87 | 0 fail/500 | ADOPTED |

## Best achieved
18.869 @ mem 0 params 460 — beats prior 18.186 by **+0.68** (Y).

Key: all cross-channel deps (red ch2 → blue ch1 / bg ch0) live among channels {0,1,2,3,4}; channels
{5..9} are pure copies. So group=2 (equal groups; size-5 is the smallest dividing 10 that can hold
{0,1,2}) suffices: weight [10,5,3,3]=450 (+10 bias)=460 vs dense 910. The harness scores (out>0) and
every channel's integer Conv response is >0 exactly on target.

## ⚠️ ORT-1.26 grouped-Conv BUG + workaround (load-bearing)
onnxruntime 1.26.0's grouped Conv silently MISCOMPUTES when group-0's weight block is sparse / has
off-centre taps: it corrupts group-1's output channels (e.g. out ch5/ch6 returned bias-of-ch0/ch1
instead of the copy). The ONNX **reference evaluator computes the correct result**, so it is an ORT
bug, not a spec issue. Diagnosed by probing: center-only group=2 ✓, dense-random group=2 ✓, but ANY
single off-centre weight in group-0 → diverges on group-1. FIX: densify group-0's weight block by
adding dummy weights on group-0 LOCAL input index 1 == global input channel 1 (BLUE), which is
structurally ALWAYS 0 in every input grid (verified 0/400 fresh) → the dummies multiply a guaranteed-
zero plane and change nothing, and params count ELEMENTS (already 450) so they cost nothing. After the
fix ORT == reference on 400/400 and fresh = 0 fail/500.

## Irreducible-floor analysis
params dominate (mem 0 — the Conv output IS the free graph output). 460 is the grouped-Conv optimum:
the 3×3 dilation forces k=3 on ch0/ch1; group-0 must contain {0,1,2} (red feeds both blue and bg) so
the minimal EQUAL group size dividing 10 is 5 → I/group=5 fixed → 10·5·9=450 weights irreducible.
Any further shrink needs a non-Conv mem-0 single op (none produces a 3×3 cross-channel relabel) or a
materialised plane (≥900B → score < 18.19). So 460/mem-0 is the floor for this structure.

## OPEN ANGLES (re-attack backlog)
- A future ORT upgrade that fixes the grouped-Conv sparse bug would let the dummy ch1 weights be
  dropped from the analysis (no params change — purely cosmetic).
- If a 3-way unequal grouping (size {3,7}) were ever expressible (it is NOT in standard ONNX Conv,
  which requires equal groups), group-0={0,1,2} would give 3·... but ONNX forbids it — dead end.
- Drop bias (only ch1 needs −1): no, removing it would mis-fire bg-far cells; saves only 10 params.

## INSIGHT (transferable)
⭐ GROUPED CONV beats a dense [O,I,k²] conv when the cross-channel coupling is LOCALISED to a channel
block (here red→{blue,bg} all in {0,1,2}) while the rest are per-channel copies: group=g cuts params
to O·(I/g)·k² with mem still 0 (output = graph output). Pick the smallest equal group size dividing
the channel count that still co-locates every coupled (target,source) channel pair. THIS is the
sub-floor escape for the "mem-0 single conv emitting bg ch0" case when only a FEW output channels
truly need the neighbourhood.
⭐ ORT 1.26.0 GROUPED-CONV BUG: ORT miscomputes grouped Conv when a group's weight block is sparse /
off-centre (corrupts OTHER groups' outputs); ONNX reference is correct. WORKAROUND: densify the
sparse group's weight block with dummy weights on an input channel that is structurally ALWAYS ZERO
(here ch1/blue) — inert, free (params count elements), and restores correctness. ALWAYS cross-check
grouped-Conv builds against onnx.reference.ReferenceEvaluator, not just ORT.
