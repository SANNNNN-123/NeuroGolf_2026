# task139 — 60b61512

**Rule:** A 9x9 grid carries two 3x3 yellow(4) Conway sprites, each spanning its full 3x3 bbox. An `xpose` (inverted-transpose) flag selects one of exactly two grid-aligned window layouts: xpose=0 → windows rows1-3/cols0-2 and rows4-6/cols5-7; xpose=1 → rows1-3/cols2-4 and rows6-8/cols5-7. Output: inside each 3x3 window every background(0) cell becomes orange(7), yellow stays 4; outside both windows stays 0.
**Current:** 16.74 pts (prior)
**Target tier:** B — label-map + final Equal (window mask is fixed per layout, no per-cell colour recovery needed).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 9x9 colour fills + bool masks | B | 1706 | 530 | 17.29 | 200/200 | works |
| 2 | scalar fills + 7-or-0 label masks, fold 2 Where into 1 select | B | 1544 | 207 | 17.53 | 500/500 | adopted |

## Best achieved
17.53 @ mem 1544 params 207 — beats prior 16.74 by +0.79. fresh 500/500.

## Irreducible-floor analysis
Dominant intermediate = padded label map Lp [1,1,30,30] uint8 = 900 B, irreducible (must be 30x30 to broadcast against the 10 colour channels in the final Equal that writes the FREE bool output). Next is the fp32 yellow slice Y [1,1,9,9] = 324 B (the 10→1 entry read). This is the standard ~17.5 tier-B label-map floor.

## OPEN ANGLES
- Could fold yellow-overlay differently or shrink the active slice, but the 30x30 pad floor dominates — diminishing returns.

## INSIGHT (transferable)
xpose (inverted-transpose) layout flag is recoverable as a SINGLE scalar from where signal can ONLY appear under one orientation (here: yellow in rows7-8 ⇔ xpose=1). Combined with FIXED per-layout window label maps, the whole task is `Where(xp_scalar, M1, M0)` then a yellow overlay — no per-cell transpose/reflection matrices, no 2-D detection. When windows are fixed-grid-aligned per a finite set of orientations, precompute one label-mask per orientation and select with a scalar.
