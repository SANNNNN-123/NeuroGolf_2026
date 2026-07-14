# task389 — f76d97a5

**Rule:** Input is a size×size grid (size 3..5) filled entirely with one non-gray colour `color`, with some cells overwritten by gray (5). Output is black (0) everywhere with `color` placed at the former gray positions. Per cell: gray(5)→`color`, background(color)→black(0); off-grid stays all-zero. So output[color] = input[5] and output[0] = input[color]; all other channels 0.

**Current:** 17.45 pts, gen:thbdh6332, mem 1856, params 42
**Target tier:** S (pure 1×1 Conv with a runtime-built 2-entry weight; output is the FREE tensor — no full-canvas intermediate ever materialised).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | rebuild weight as 2 outer-products in 4D (drop W_2d reshape copy), fp32 | S | 1416 | 42 | 17.72 | — | MARGINAL (+0.26) |
| 2 | build termA/termB/W16 in fp16, single Cast→fp32 for Conv weight | S | 1156 | 42 | 17.91 | 200/200 | ADOPT (+0.46) |

## Best achieved
17.91 @ mem 1156 params 42 — beats prior 17.45 by +0.46. Fresh isolated 200/200.

## Irreducible-floor analysis
The Conv weight W is [10,10,1,1] and MUST be fp32 (ORT Conv requires the weight dtype to match the fp32 input), so the final W tensor is a hard 400B. W has two nonzeros in distinct rows AND columns (W[color,5]=1, W[0,color]=1) so it is rank-2 — it needs a sum of two rank-1 outer products, i.e. at minimum two product tensors + one combine tensor. Built in fp16 those three are 200B each (600B); plus the 400B fp32 cast = 1000B of weight machinery. The colour-index recovery chain (ReduceMax/mask/ArgMax/OneHot/reshapes) is all ≤40B tensors (~156B). The output one-hot expansion is free (it is the Conv's `output`).

## OPEN ANGLES (re-attack backlog)
- The 400B fp32 W cast is the single biggest item. Only removable if the whole Conv could run in fp16, which needs an fp16 input — and casting the 10-ch input is 18000B, far worse. No cheaper escape found.
- Folding term B into a second op (e.g. a separate Conv or a Where on input ch5) costs a 3600B slice plane — strictly worse. Two-entry weight via a single OneHot-2D (depth-100) costs ≥1200B (the [2,100] one-hot) — also worse.
- Could try collapsing presence_raw+presence (40+40B) but the saving is <40B and would not change the tier.

## INSIGHT (transferable)
⭐ For a "recolour mask→colour on black canvas" task where output channel assignment is a fixed permutation of input channels parameterised by ONE recovered scalar colour, the entire net is a single 1×1 Conv whose runtime weight is a tiny [10,10,1,1] permutation matrix — the 10-channel output expansion lands in the FREE output for zero mem. Build the (rank-2, two-nonzero) weight as fp16 outer-products and pay ONE fp32 Cast at the end (Conv forces fp32 weight to match fp32 input); the fp16 construction halves every [10,10] working tensor. This is a clean Tier-S floor (~1156B); the 400B fp32 weight cast is the irreducible item.
