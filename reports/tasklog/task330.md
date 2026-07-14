# task330 — d2abd087

**Rule:** 3..6 gray "continuous_creature" sprites on a 10x10 black canvas, bounding boxes
separated with spacing>=1 (no two sprites within Chebyshev distance 1). Each sprite has a
pixel COUNT in {4..8}. Output recolors every gray pixel RED (2) if its sprite has exactly 6
pixels, else BLUE (1); background stays 0. Per-connected-component count → color.
**Current:** 15.22 pts (public net: MaxPool flood + ScatterND/Gather histogram, mem 16604, params 1133)
**Target tier:** detection/B — needs exact per-component count (connectivity), but isolated
sprites + tiny canvas make it a closed-form flood, not a wall.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp16 10x10 id-flood (MaxPool×7) + ScatterND-add histogram + Gather | B | 8904 | 238 | 15.879 | 200/200 | ADOPT-CANDIDATE +0.66 |

## Best achieved
15.879 @ mem 8904 params 238 — beats prior 15.22 by **+0.659**. fresh 200/200 isolated.

## Irreducible-floor analysis
Dominant intermediates are the seven fp16 [1,1,10,10] flood planes (200B each ≈ 1400B) plus a
couple of small fp32 helpers and the [101] histogram buffer (404B). No 30x30 fp32 plane, no
[100,100] all-pairs matrix. The public net wasted memory keeping fp32 (400B) 10x10 planes and
30x30 (3600B) planes; halving to fp16 + a leaner histogram dropped 16604→8904.

## Key insights
- ⭐ Component counting is NOT a wall when sprites are ISOLATED (gen guarantees bbox spacing≥1):
  an **8-connected 3x3 MaxPool** id-flood never leaks across components, so 7 iters (max creature
  size 8) converge exactly. One MaxPool + one Mul(gray) per iter — far cheaper than 4 Pad/Slice
  shifts for 4-connectivity.
- ⭐ Per-component COUNT = histogram via **ScatterND with reduction='add'** (opset-16 op, but the
  scorer checks DOMAIN not VERSION so it's allowed; verified working under ORT_DISABLE_ALL). A
  tiny [101] buffer + Gather-back beats both the [100,100] all-pairs equality matrix (≥20kB fp16)
  AND a 37-offset id-gated window Conv (~18kB). This is the lever that made the whole task cheap.
- ⭐ Equality-gated counting needs NO explicit gray gate: after the id-flood, only same-component
  cells share the unique id (bg id=0), so `id==id` already excludes neighbors and background.

## OPEN ANGLES
- Could drop a couple more flood planes by checking whether 5 or 6 iters suffice (6 already
  verified 0-mismatch in proto; used 7 for margin). Marginal (~−ln saving).
- The ScatterND updates plane is fp32 [100] (ScatterND wanted f32); a uint8/f16 updates path
  could shave a few hundred bytes if ORT accepts it.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2800/16.99; root_idx int32 400B=min index buffer, ch5_f 400B forced-fp32 crop; no input+delta route (in gray5→out ch0/1/2). No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).
