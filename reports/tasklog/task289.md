# task289 — b91ae062 (nearest-neighbour upscale by a data-dependent factor K)

**Rule:** Input is a 3x3 grid (size=3) with 3..5 scattered coloured pixels on
black. `enhance = len(colors) = K` = the number of DISTINCT non-bg colours used
(1..5). Output is a (3K)x(3K) grid where each input cell (r,c) of colour v becomes
a solid KxK block of colour v at (r*K, c*K) — i.e. `output[u,v] = input[u//K, v//K]`
for u,v < 3K, and EVERY channel is 0 outside that top-left footprint. (common.grid_enhance)
**Current (public):** 14.87 pts.
**Target tier:** B (label-map + final Equal). Tier S/A blocked: the upscale factor K
and hence the output footprint are DATA-DEPENDENT (depend on the distinct-colour
count), so `u//K` is a data-dependent index map → needs a Gather, not a fixed
Conv/permute, and the footprint is not a fixed separable rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 3x3 colour-index Conv→fp16, K=Σpresence, ri=clip(floor(u/K),0,2) double-Gather→fp16 plane, And-mask + uint8 Cast + Where(sentinel) + Equal | B | 5988 | 87 | 16.29 | 200/200 | works |
| 2 | src→uint8 (gathered plane uint8 not fp16); nested Where instead of And+Where | B | — | — | — | — | folded into #3 |
| 3 | PAD src to 4x4 with sentinel 99 at row/col idx 3; ri=clip(floor(u/K),0,3) → out-of-footprint cells read the sentinel cell → NO mask/Where plane at all | B | 2245 | 86 | 17.25 | 200/200 | big win: removed 2 full planes |
| 4 | fuse Max+Min into one Clip(qf,0,3) | B | **2125** | 86 | **17.30** | **500/500** | BEST |

## Best achieved
**17.30 @ mem 2125 params 86 — fresh 500/500 (isolated, all K=1..5 covered).**
Beats public 14.87 by **+2.43**. adopted? N (build-only per brief).

## Irreducible-floor analysis
The lone 30x30 plane is the gathered label map `L [1,1,30,30] uint8 = 900 B` — the
canonical label-map floor that drives the final `Equal(L, arange)→BOOL` output. The
double Gather is intrinsically a full-canvas 30x30 read, so one 30x30 plane is
forced; uint8 is already the smallest dtype for it. Next: `in33` slice [1,10,3,3]
fp32 = 360 B (the 10-channel 3x3 read for the colour Conv), the [30] fp32 index
vectors (q/qf/ric_f ≈ 360 B) and `ri` int64 [30] = 240 B, `g_rows` [1,1,30,4]
uint8 = 120 B. Everything else ≤90 B.

## OPEN ANGLES (re-attack backlog)
- Shave `in33` (360 B): derive the 3x3 colour index without a 10-channel slice
  (e.g. MatMul contracting the channel axis of a 3x3 input window) — but every
  channel-contraction route still needs a ≥90-elem 10-ch read; net neutral, <+0.1.
- Shave the [30] index-vector chain (~600 B over q/qf/ric_f/ri): compute ri in
  integer arithmetic (int Div by an int K) to drop the float Floor/Clip vectors;
  fp16 on these crashes ORT Min/Max under ORT_DISABLE_ALL so int is the only route.
  Marginal (~+0.2) and the 900 B L plane caps the task near ~17.6.
- The 900 B L is the label-map floor; Pad rejects bool so a 9x9/15x15 bool Equal +
  Pad to 30x30 is not available — no clean sub-900 final.

## INSIGHT (transferable)
⭐⭐ **Data-dependent nearest-neighbour UPSCALE by a recovered scalar factor K =
double Gather of the small source by a [30] index map `ri[u]=clip(floor(u/K),0,n)`,
with the OUT-OF-FOOTPRINT region handled for FREE by PADDING the source with one
sentinel row+col (index n) and clipping the index to n** — cells beyond u≥(src·K)
clip to the sentinel index and read 99, so NO validity mask / And / Where plane is
ever materialised. This removed two full 30x30 planes (5988→2125, +1.0 pt) vs the
naive mask-then-Where approach. The footprint is the UNION (row-idx==n OR col-idx==n)
and a sentinel placed in BOTH the pad row and pad col reproduces that union exactly
after the two axis-separable Gathers.
⭐ K (distinct-colour count) is a pure scalar: `ReduceMax(input,[2,3])` per-channel
presence → slice ch1..9 → ReduceSum. No per-cell plane.
⭐ This is the task195/task011 Kronecker/upscale family generalised to a
DATA-DEPENDENT scale factor — the const-index Gather becomes a `floor(u/K)` Gather.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 15 (in 3×3, out ≤15). Flagged `mask_float` [30,3] 360B: its 30 IS the einsum free-output row dim. Cropping → Pad → counted 9000B fp32; the P[15,30] re-embed adds +450 params, which exceeds the 180B saved. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
