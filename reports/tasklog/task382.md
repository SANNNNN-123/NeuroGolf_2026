# task382 — f15e1fac

**Rule:** Grid is W×H (both in [10,20]). Red dots on the left edge at rows `rows`
(spaced 4–7 apart); cyan dots on the top edge at cols `cols` (spaced 2–4). The
output replicates the top cyan pattern p down every row, each row r shifted RIGHT
by S[r] = the running count of red rows passed at/above r (`Out[r,c]=p[(c+1)-S[r]]`,
clipped to width). Reds stay put. The whole figure may be horizontally flipped
(`flip`) and rotated/reflected by `gravity` (0..3 = optional vertical flip then
optional transpose).
**Current (was):** 14.00 pts, custom fp16 round-trip net, mem 59042, params 1098
**Target tier:** A (orientation-dependent per-row shift; needs a canonicalise→
solve→uncanonicalise round-trip + one data-dependent gather index plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | crop full pipeline to 20×20 active region (gen bounds grid ≤20×20, top-left anchored) + Pad label map back to 30×30 (uint8 sentinel 10) | A | 35092 | 562 | 14.518 | 400/400 | win |
| 2 | replace 3600B ReduceSum occ plane with channel-0 Slice (bg0) OR (M>0) | A | 32608 | 571 | 14.590 | 400/400 | win |
| 3 | derive W/H from bg0 fp32 reductions directly (every in-grid row/col has a bg cell) → drop the occH fp16 plane | A | 31892 | 571 | 14.612 | 500/500 | win, deployed |

## Best achieved
14.612 @ mem 31892 params 571 — adopted? Y (overwrote deployed). Beats prior
14.00 by +0.612 (prior re-attack only reached 14.155). Isolated fresh 200/200.

## Irreducible-floor analysis
After cropping, mem is dominated by fixed-cost entries that can't shrink:
- 3600B fp32 — the single Conv entry plane (`Σ k·input_k`, must be fp32, output is
  full 30×30; the 3600B "pay-one-entry" floor).
- 1600B fp32 ×2 — the 20×20 Slice of the conv (→M fp16) and the channel-0 bg Slice
  (in-grid extent signal). fp32 20×20 is the slice floor.
- 1600B int32 — the per-cell cyan-shift Gather index plane `Dc` (Gather indices
  reject uint8/fp16, int32 30×30→20×20 is the floor).
- 900B uint8 — the 30×30 padded label map before the free BOOL Equal.
- ~12 fp16 @800B (~9.6KB) — the canonicalise/uncanonicalise chain (M, M_t, M1, M2,
  Mc, Oc, O1, O2, O2t, O, Dp, Dcf). This is the real remaining bulk.

## OPEN ANGLES (re-attack backlog)
- The ~12 fp16 working planes (each 800B) are the round-trip canonicalisation
  chain. A reformulation that solves in the ORIGINAL frame (no transpose/flip
  round-trip) — e.g. orientation-equivariant per-axis selection of the shift
  direction — could delete ~5–8 of those planes (~4–6KB → +0.15–0.25). Not yet
  found; the rule's shift axis flips under transpose so it's non-trivial.
- The int32 Dc shift plane (1600B): a per-row shift expressed as a MatMul with a
  data-dependent banded shift matrix would replace int32 with fp16, but a 2-D
  matrix is the same byte cost — no clear win.

## INSIGHT (transferable)
⭐ CROP-TO-ACTIVE on a gen-bounded grid is a large, low-risk win even on an
already-custom fp16 net: here a full 30→20 crop of EVERY working plane + uint8
Pad-back cut mem 59042→31892 (+0.61) with the logic untouched. PITFALL that cost a
debug cycle: keep the FINAL in-grid mask in the ORIGINAL orientation — the
canonical-frame rectangle mask (`row<W_canon ∧ col<H_canon`) is WRONG when applied
to the un-transposed output; use the raw input-frame occupancy (`bg-ch0 OR M>0`)
for the off-grid sentinel. Also: when every in-grid row/col is guaranteed to
contain a background cell, the channel-0 background slice alone gives exact W/H via
1-D reductions — no separate occupancy plane needed.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 5741 -> 5732 (+0.002); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].