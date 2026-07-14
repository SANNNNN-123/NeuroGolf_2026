# task132 — 56ff96f3

**Rule:** The grid holds 1..2 axis-aligned rectangular "boxes". Each box has a UNIQUE
colour and appears in the INPUT as exactly its two DIAGONAL corner cells (TL+BR or
TR+BL depending on a per-box flip). Boxes never overlap (`overlaps()` margin 1). The
OUTPUT fills each box's full bounding rectangle with that box's colour; every other
in-grid cell is background (colour 0). Grid is 6..15 × 6..15, placed top-left.
Because the two marks are the diagonal corners, each colour's pixel bbox = its box rect.

**Current:** 14.709 pts, `ext:kojimar6275`, mem 29394, params 64
**Target tier:** A (separable per-channel: output[c,r,c'] = rowband_c[r] AND colband_c[c'])
but background channel 0 (= in-grid AND not-covered) is the complement of a UNION of
boxes → not row⊗col separable, so it collapses to a label-map (B) with a separable
in-grid mask. Tier S impossible (fill is non-local).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map via weighted channel-MatMul L=A@B, full 30×30, fp32 | B | 29560 | 84 | 14.70 | 200/200 | correct, no gain (ch0 needed in-grid mask) |
| 2 | + fp16 working tensors + crop to WORK=16 window | B | 18052 | 70 | 15.20 | 200/200 | win |
| 3 | + in-grid mask = rowany⊗colany (kill [1,1,30,30] ReduceMax-over-ch chain) | B | 11156 | 69 | 15.67 | 200/200 | win |
| 4 | + Greater-before-Slice; rowany from fp32 occ reduce; wrow via one Where; Or/Not bands; WORK=15; uint8 Lg | B | 9395 | 67 | **15.84** | 500/500 | **WIN** |

## Best achieved
**15.84** @ mem 9395 params 67 — adopted? **N** (orchestrator gates). Beats prior 14.709 by
**+1.14** → clear WIN (≥+0.3). Generalizes 500/500 fresh (max grid observed 15×15).

## Irreducible-floor analysis
Dominant intermediates:
- **2 × 1200B fp32 occupancy** `rowocc[1,10,30,1]`, `colocc[1,10,1,30]` = ReduceMax of the
  fp32 input over each spatial axis. PER-CHANNEL is mandatory (the two boxes carry distinct,
  instance-random colours, so all 10 channels must stay separated to recover per-colour bbox).
  ReduceMax/MatMul emit fp32 from the fp32 input; casting to fp16 ADDS a tensor (1200+600) so
  fp32 is the floor. Cropping doesn't help (the reduce output keeps the full 30 on the *other*
  axis). This 2400B is the hard structural floor of any per-channel-bbox formulation.
- **900B uint8 label map** `L[1,1,30,30]` (padded) — required so `Equal(L,arange)` broadcasts
  the 10-way one-hot into the FREE bool output. Can't be <30×30 (output must be 30×30; Pad on
  bool is rejected, so pad the uint8 label not the bool output).
- Remainder (~6000B) = the per-channel band machinery: 4× `[1,10,15,1]` fp16 ramp-mask Where
  sources (rmin/rmax/cmin/cmax, 300B each) + Or/Not band bools + `colcond_f`/`wrow` + the two
  `[1,1,15,15]` fp16 MatMul/Where planes. Each is needed once; counts are near-minimal.

## OPEN ANGLES (re-attack backlog)
- Collapse the four `[1,10,15,1]` Where-sources: derive rmin via a negated ramp reusing the
  rmax reduce — does not cut peak tensor count (still one Where each), ~0 payoff. Tried mentally.
- CumSum band reconstruction (prefix≥1 AND suffix≥1 over the 2-mark profile) — same `[1,10,15,1]`
  cost as the ramp-mask, no win; CumSum is allowed but doesn't beat min/max-bound bands.
- Value-plane (V=Σ k·input_k) instead of per-channel: a single [1,1,30,30] fp32 V is 3600B >
  2×1200 occupancy, AND loses colour separation for the bbox → strictly worse. Rejected.
- The ~2400B fp32 occupancy floor is the gate between ~15.8 and ~17; no reformulation found
  that recovers per-channel bbox without two fp32 spatial reductions.

## INSIGHT (transferable)
⭐ **Per-colour bbox-FILL = label-map + weighted channel-axis MatMul.** With disjoint boxes,
`L[r,c'] = MatMul(A[r,c]=c·rowband_c[r], B[c,c']=colband_c[c'])` contracts the 10-channel axis
into a single [1,1,H,W] colour-index plane WITHOUT ever materializing a [1,10,H,W] product —
disjointness guarantees no double-stamp and background (weight 0) falls out for free.
**The in-grid background channel is the separable rectangle `rowany ⊗ colany`** (background ch0
fills every in-grid cell, so "any channel occupies row r" = "r < grid-H"); reducing the existing
fp32 occupancy over the channel axis gives it for ~120B, killing the naive [1,1,30,30]
ReduceMax-over-channels in-grid plane (saved ~6500B here). Floor of this class = two fp32
per-channel spatial reductions (~2400B) + the 900B uint8 label plane ⇒ ~15.8 pts ceiling.
