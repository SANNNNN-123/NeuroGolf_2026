# task049 — 23b5c85d

**Rule:** N (2..5) solid axis-aligned rectangles of DISTINCT colours are drawn on a
width×height (10..20) black canvas; boxes may overlap and the LAST box drawn is the
one with strictly-smallest width AND height, whose colour `colors[-1]` is guaranteed
(generator rejection-sampling) to be the RAREST by visible pixel count and is never
occluded. OUTPUT = a solid `talls[-1]×wides[-1]` rectangle filled with `colors[-1]`,
anchored at top-left (0,0); the harness one-hot encodes it so every cell outside the
output grid is all-channels-0.
**Current:** 15.93 pts (public CumSum-scan net), beaten.
**Target tier:** A (separable origin-anchored rectangle) — output[ch,r,c] =
rare[ch] AND (r<tall) AND (c<wide); only the per-instance rare-plane extraction needs
a single 30×30 plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel ReduceMax occupancy + rare-gate Mul + Less⊗Less | A | 5952 | 72 | 16.30 | 200/200 | works |
| 2 | rare plane via 1×1 Conv (runtime one-hot weight) + 1-D reductions | A | 4996 | 72 | 16.47 | 500/500 | ADOPT-candidate |

## Best achieved
16.47 @ mem 4996 params 72 — adopted? N (build agent; main adopts). Beats prior 15.93? Y (+0.54).

## Irreducible-floor analysis
Dominant intermediate is the single fp32 rare-colour plane [1,1,30,30] = 3600B produced
by the 1×1 Conv. This is the documented colour-plane floor (cannot go below fp32 30×30
for a per-cell channel-collapse). Everything else is ≤900B (the rect bool [1,1,30,30]) or
1-D/scalar. The alternative (per-channel [1,10,30,*] occupancy + gate-Mul) costs 4800B, so
the Conv-collapse is strictly better.

## OPEN ANGLES (re-attack backlog)
- Slice the active region to ≤20×20 before the Conv to drop the plane to 1600B — blocked
  because slicing the fp32 input to [1,10,20,20] materialises a 16000B intermediate (worse).
- Recover tall/wide from per-channel COUNTS alone (count_rare = tall·wide) would need a
  factorisation of the product into its two factors — no cheap closed form; the bbox-span
  route is needed, which needs the plane.
- True Tier S impossible: output colour/size is a global rank+bbox function of the input,
  not a per-cell neighbourhood function.

## INSIGHT (transferable)
⭐ A 1×1 Conv whose WEIGHT is a RUNTIME-computed one-hot [1,10,1,1] (ORT accepts a
non-initializer Conv W) collapses "select channel k AND extract its 30×30 plane" into ONE
op producing a single fp32 [1,1,30,30] (3600B) — strictly cheaper than per-channel
ReduceMax occupancy ([1,10,30,1] = 1200B ×2) plus a rare-gating Mul (another 1200B ×2 =
4800B total). Use this whenever you've recovered a per-channel selector one-hot and need
its spatial plane. The "smallest box + rarest colour → solid origin-anchored block" rule is
fully separable (rare one-hot ⊗ (r<tall) ⊗ (c<wide)) and beats the public scan floor.
