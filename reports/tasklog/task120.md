# task120 — ARC-AGI 50cb2852

**Rule:** N=2..4 solid filled rectangles (colours from {1,2,3}, >1 distinct), non-overlapping with a
1-cell gap, painted on an active grid (W in 12..14, H=W±2). In the OUTPUT each rectangle keeps its
1-pixel OUTLINE in its own colour and its INTERIOR (cells not on the rect's top/bottom row or
left/right col) is recoloured to CYAN(8). Verified closed-form (0 mismatches / 300 fresh):
- ch0 (bg)  = input ch0 (copy)
- ch_c (c∈{1,2,3}) = (centre is colour c) AND (NOT interior) = border pixel of that colour
- ch8 (cyan) = INTERIOR = centre and all 4 orthogonal neighbours are non-bg (occupancy of in0)
- interior test depends ONLY on in0 (occupancy = NOT in0); colour channels not needed for cyan.
Input colours observed = {0,1,2,3} only.

**Current:** 18.19 pts, single dense `Conv[10,10,3,3]` (910 params, output IS graph output → mem 0).
**Target tier:** sub-floor grouped-Conv (the task352-style escape) — only admissible lever here.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | grouped Conv (group=2/5/10) sub-floor escape | — | — | — | — | — | INFEASIBLE (channel-coupling) |

## Best achieved
none — 18.19 stands. The mem-0 dense Conv is at HARD floor for this coupling.

## Irreducible-floor analysis
Score 18.19 ⇒ params budget = exp(25−18.19) ≈ 910; beating by +0.3 needs **mem+params ≤ 672**, i.e.
STRICTLY FEWER than 910 elements. The only mem-0 way to shrink params is an equal-group Conv (mem stays
0 because the conv output is the graph output). The cross-channel coupling, with output channel placement
FIXED by the harness (it scores `out>0` per canonical channel; cyan is locked at ch8):
  out0←{0}, out1←{0,1}, out2←{0,2}, out3←{0,3}, **out8←{0}**, out{4,5,6,7,9}=∅.
In an equal-group Conv (O=I=10), output channel `o` lives in group `o//(10/G)` and may only read the
input channels of that same group. Two couplings break EVERY partition:
  • **cyan ch8 → in0**: needs a group co-locating output index 8 with input index 0 ⇒ group span ≥9 ⇒
    group size ≥9, and 9∤10. Only group size 10 (the dense conv itself) works.
  • border out{1,2,3} → in0 also needs in0 grouped with output indices 1..3 (group≥4 → size 5), but even
    that secondary constraint can't be satisfied simultaneously once cyan forces span 0..8.
Tested G∈{2,5,10}: all INFEASIBLE (see analysis). No equal group <10 contains the coupling.
A DECOMPOSITION cannot help either: cyan's interior mask is an inherently full 30×30 plane. The cheapest
split (group=2 Conv[10,5,3,3]=460 for bg+border, then a separate ch0 occupancy conv → [1,1,30,30] cyan
plane routed to ch8) materialises ≥900B (fp32 conv output) → 460+900 ≈ 1360 ⇒ score ≈17.8, WORSE than
18.19. Conv outputs float so cyan can't be produced as free uint8. Every decomposition pays a ≥900B plane
that only beats mem-0 below the existing score — the BUILD_PROMPT MEM-0 SINGLE-CONV-AT-FLOOR rule exactly.

## OPEN ANGLES (exhausted)
- None buildable. The wall is the fixed cyan@ch8 ← bg@ch0 long-range channel coupling under
  equal-group constraint; relabelling cyan to a low channel is impossible (harness scores canonical order).

## INSIGHT (transferable)
⭐ The grouped-Conv sub-floor escape is blocked not only by "how many input channels a target needs"
but by the **distance between the target output channel index and the lowest input channel it needs**:
an interior/occupancy channel that lives HIGH (cyan=8) but reads bg=ch0 spans 0..8, and no equal group
<10 can bridge that span (would need size ≥9∤10). This is a distinct failure mode from task352 (red→blue/bg
all clustered in 0..2) and complements the prompt's "ch5→{0,1,5} needs group≥6∤10" example: here it is
"ch8→{0} needs group≥9∤10". Discriminator to add: compute span = max(target_idx, max_src) − min(min_src,
target_idx); feasible only if some g|10 with g ≥ span+1 exists, i.e. span ≤ 4 (g=5) or span ≤ 9 with g=10.
A required occupancy channel sitting at a HIGH canonical colour index (cyan=8, …) is therefore a structural
floor even when it reads only ONE input channel.
