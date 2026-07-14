# task097 — 42a50994

**Rule:** A grid (w,h in [5,20]) is sprinkled with single-colour pixels (one random colour c). A coloured cell survives iff it has >1 coloured cell in its 3x3 neighbourhood INCLUDING itself, i.e. >=1 coloured 8-neighbour. The ONLY cells that change are in-grid coloured cells with ZERO coloured 8-neighbours -> cleared to background. bg and off-grid cells unchanged.
**Current:** 15.70 pts (prior friends-Conv + ReduceMax in-grid + Where), mem 10800, params 102
**Target tier:** A (closed-form local-Conv mask routed into the FREE Where output)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior: friends-Conv + ReduceMax(in-grid) OR + Where | A | 10800 | 102 | 15.70 | 267/267 | baseline (two fp32 planes) |
| 2 | BANDED 3x3 Conv (centre=10, nbr=1), kill = (C==10), opset-11 float Equal, Where(kill,bg,input) | A | 4500 | 101 | 16.57 | 200/200 | ADOPTED |

## Best achieved
16.57 @ mem 4500 params 101 — beats prior 15.70 by +0.86. fresh 200/200 isolated.

## Irreducible-floor analysis
ONE fp32 [1,1,30,30] Conv plane (3600B) is irreducible: the Conv must consume the fp32 input
one-hot, and the centre-banded count value distinguishes "coloured & isolated" (==10) from
bg/off-grid (<=8) and connected-coloured (>=11) in a SINGLE plane, so no separate occupancy
plane is needed. The only other intermediate is the bool kill mask (900B). The 10-ch output
expansion is routed into the FREE Where output. Casting the Conv plane to fp16 would ADD a
1800B plane (the fp32 entry is mandatory), so 4500B is at floor for this structure.

## OPEN ANGLES (re-attack backlog)
- Could shave the bool kill plane by folding the C==10 test into the Where condition directly,
  but Where needs a bool mask anyway -> no saving. 4500B is effectively floor.

## INSIGHT (transferable)
⭐ "Remove isolated single-colour pixels" (Conway-style friends>1) collapses to ONE banded 3x3
Conv: centre weight = base (10), 8 neighbours = 1, over colour channels 1..9. The single value
C separates ALL cases by MAGNITUDE BAND — C==base ⟺ coloured-and-isolated, C<base ⟺ bg/off-grid,
C>base ⟺ coloured-with-neighbours — so the kill mask is one float Equal (opset 11) with NO
separate occupancy / in-grid plane. Eliminates the second fp32 plane (ReduceMax in-grid detector)
that the naive friends-count + Where approach needs.
