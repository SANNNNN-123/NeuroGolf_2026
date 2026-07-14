# task080 — 39e1d7f9

**Rule:** A block-lattice grid: bitmap (size 5..10) upscaled with each cell -> an (sp x sp)
solid block separated by single `linecolor` lines, period p = sp+1 = 6-(size-1)//2 + 1
(only p in {3,4,5} stay <=30 and are scored; size 8 -> 31 is dropped). (size-1)//2 "pixels"
each carry a colors[0] CENTER block. The INPUT shows exactly ONE fully-decorated pixel: its
4 orthogonal neighbour blocks = colors[1] (edge), its 4 diagonal neighbour blocks = colors[2]
(corner, only when 3 colours). The OUTPUT decorates EVERY center the same way, clipped at the
bitmap border. Stamps never overlap so fill lands only on background cells.
**Current (prior):** 14.358 pts (ext:biohack_new, 122-node brute force), mem ~41638.
**Target tier:** B — data-dependent period needs a runtime downsample/upscale (Gather), so
not a Tier-S spatial copy; but it IS clean closed-form (NOT the "data-dependent PERIOD = needs
runtime" wall the blank note feared).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-res 8-dir shift-by-p via Gather, colour planes fp32 | B | 192644 | 170 | 12.83 | 200/200 | correct but bloated (16 gather/pad fp32 planes) |
| 2 | full-res shifts, fp16 working planes | B | 28558 | 285 | 14.73 | 200/200 | better; still per-shift 30x30 planes |
| 3 | BITMAP-resolution: downsample (Gather block-tops)->tiny 10x10 ops->upscale | B | 25246 | 285 | 14.83 | 231/231 | magnify lever; planes shrink 9x |
| 4 | row-count via no-pad Conv (drop occ plane) + uint8 final | B | 19422 | 598 | 15.10 | 231/231 | removed occ + a tail Where |
| 5 | TWO-SENTINEL-BLOCK upscale (line+offgrid folded into one double-Gather) | B | 18222 | 488 | 15.16 | 231/231 | no tail Where, no line/ingrid 30x30 masks |
| 6 | pad-once-per-source shifts + uint8 outB12 (kills PrecisionFreeCast) | B | 17106 | 487 | 15.22 | 500/500 | **best** |
| 7 | bitmap count Convs as uint8 QLinearConv + onnxsim | B | **10834** | **454** | **15.67** | 300/300 | ADOPTED |

## Best achieved
15.67 @ mem 10834 params 454 — adopted as `custom:task080+qconv+onnxsim`.
Beats prior live 15.63 by +0.043; fresh 300/300.

2026-07-01 source/live parity cleanup: source was already ahead of live after the
latest qconv simplification.  Fresh verify passed (`428/500` eligible
instances, 0 failures) and `src.adopt 80` promoted the source-owned build:
`memory=10198`, `params=454`, `points=15.726497` stored 231/231.  This is a
small parity adoption, not a new task001-style mechanism.

## Irreducible-floor analysis
Dominant = colf32 (3600B, the ONE fp32 30x30 colour-index Conv entry — irreducible per
FLOOR_RESEARCH) + the downsample Gather Bg2 [1,1,10,30] (1200B fp32, inherits colf dtype)
+ the uint8 final 30x30 plane big (900B). The remaining ~11kB is ~30 tiny 10x10/12x12
bitmap planes (fp16 200B / 288B each). Not strictly at floor — see open angles — but each
remaining lever is <0.1 pt and risks the 600s watchdog.

## OPEN ANGLES (re-attack backlog)
- Convert the {0,1} bitmap masks (bocc/seed/vbar/iscenter + dilations) from fp16 to bool
  (100B). Blocked by ORT Pad-rejects-bool on the pad-once buffers and uint8 having no
  Mul/And; needs uint8-pad -> bool-cast -> And. ~10 planes x 100B ~= +0.06 pt, fiddly.
- Bg2 1200B fp32: cast colf32 -> uint8 (900B) and gather uint8 (Bg2 300B). Net ~0 unless
  colf32 itself becomes removable (it isn't — only feeds downsample + linecolor gathers).
- Cheaper c0/c1/c2 recovery without the 4-neighbour iscenter AND (fewer bitmap ops).

## INSIGHT (transferable)
- ⭐ A data-dependent block PERIOD is NOT a wall: collapse to BITMAP resolution with a
  runtime-stride **downsample Gather** (block-top indices i*p) -> tiny K x K ops -> a runtime
  **upscale Gather** (uidx = i//p). All intermediate planes shrink 30x30 -> 10x10 — the
  task159/task195 magnify lever applied to a whole lattice rather than one sprite.
- ⭐ TWO-SENTINEL-BLOCK upscale: fold "overlay grid lines" AND "off-grid -> 99" INTO the
  upscale table — append a linecolor block (idx 10) and a 99 block (idx 11, padded LAST so it
  wins the corner) to the K x K block table, then `uidx[i] = 11 if i>=A else 10 if (i+1)%p==0
  else i//p`. ONE double-Gather emits the entire final plane: no tail Where, no separate
  line/in-grid 30x30 masks.
- ⭐ The harness `calculate_memory` counts a tensor at its INFERRED (declared) dtype x the
  TRACE shape, NOT the ORT runtime-upcast dtype — so fp16 working planes DO score at half
  even though the ORT profiler shows them upcast to fp32 via InsertedPrecisionFreeCast. BUT a
  fp16 plane feeding the graph-output op (Equal) adds an fp32 cast plane that IS counted;
  making that feeder **uint8** removes the cast (uint8 isn't upcast). uint8 Gather + uint8
  Equal both run under ORT_DISABLE_ALL.
- Row/col occupancy COUNT as a no-pad Conv over channels 1..9 (W[1,10,1,30]) replaces the
  30x30 occupancy plane ReduceSum would force (per-row count = full-line detector).
⭐ 2026-06-28 dtype update: small bitmap count Convs that only feed `Equal(count,k)` or
`Greater(count,0)` should use uint8 `QLinearConv` with scale=1/zp=0, not fp16 Conv. Here it
halved `occ_cnt`, `edge_cnt`, `corner_cnt`, and the seed/occupancy cast planes.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 10563 -> 9834 (+0.072); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].