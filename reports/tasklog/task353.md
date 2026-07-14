# task353 — dc433765

**Rule:** Input has exactly two coloured pixels: green (3) at (gr,gc) and yellow (4) at (yr,yc),
chebyshev distance >=2 (generator excludes "too close"). Output: yellow stays put; green moves ONE
cell toward yellow on each axis independently: ng_r=gr+sign(yr-gr), ng_c=gc+sign(yc-gc). Rest is
background (ch0) inside the HxW (<=14x12) origin-anchored grid; off-grid all-channels-off.
**Current:** 16.40 pts (public net) → custom **16.75** pts, mem 3110, params 719
**Target tier:** A/B (single label map + Equal) — output has 3 distinct colours (0/3/4) so a separable
per-channel route would need 3 full 10-ch Wheres; one 30x30 uint8 label plane is the lean form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice input to 14x12 window + label map | B | 10630 | 62 | 15.72 | 271 ok | window slice = 6720B fp32 plane, bad |
| 2 | ReduceSum profiles (no window) + label map | B | 5946 | 116 | 16.29 | 271/271 | profiles [1,10,30,1]=1200B x2 dominate |
| 3 | green/yellow sig via no-pad Conv (1*g+100*y) | A/B | 4102 | 713 | 16.52 | 200/200 | replaces profiles, Conv kernel=600 params |
| 4 | fp16 coord decomposition | A/B | 3614 | 713 | 16.63 | 200/200 | halve small fp32 working planes |
| 5 | slice occupancy vec for in-grid (drop H/W counts) | A/B | 3342 | 719 | 16.69 | 200/200 | removes Less/ReduceSum-count planes |
| 6 | green coord = total - 100*yc (drop green-profile planes) | A/B | 3110 | 719 | 16.75 | 500/500 | scalar arithmetic vs full-vector green profile |

## Best achieved
**16.75** @ mem 3110 params 719 — adopted? pending main. Beats prior 16.40 by **+0.35** (>=+0.3 YES).

## Irreducible-floor analysis
Dominant intermediate = the 30x30 uint8 label plane L (900B) padded before the final Equal(L,arange).
Irreducible because the output has 3 distinct colours (bg=0, green=3, yellow=4) plus an off-grid
sentinel — a separable row⊗col route into the free bool output can carry only ONE colour rectangle;
multi-colour needs the label carrier. fp16 working planes + Conv-signature + scalar green-coord put
everything else under ~200B each.

## OPEN ANGLES (re-attack backlog)
- Tier-S copy route: output ~= input with green pixel moved 1 cell. ch4/ch1-9 == input exactly;
  only ch0 (±2 cells) and ch3 (shifted) differ. A data-dependent Gather-shift of input ch3 + a
  Where(editmask, ...) could avoid the 900B label, but editmask is itself a 30x30 plane — unclear win.
- Could fold occupancy (rowocc/colocc, 240B fp32) into the signature Conv as a 2nd output channel,
  but that doubles the kernel (+300 params) — net loss. Left as-is.

## INSIGHT (transferable)
⭐ Two coloured single-pixels of known colours c1,c2: recover BOTH per-axis profiles in ONE no-pad
Conv `W[1,10,1,30]` with channel weights {c1:1, c2:100} (band separation) — output [1,1,30,1] (120B)
beats the [1,10,30,1] ReduceSum profile (1200B). Decompose by `yellowprof=sig>50`, then derive the
OTHER coord arithmetically: green = Sum(sig*ramp) - 100*Sum(yellowprof*ramp) — never materialise the
second profile plane (saves a full-vector plane per axis). "move toward" = sign(Δ) via
Where(Δ>0,1,Where(Δ<0,-1,0)). In-grid mask for an origin-anchored solid grid = SLICE the per-axis
occupancy vector (ReduceMax>0) to the working WRxWC — no H/W count / Less needed.
