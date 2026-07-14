# task107 — 469497ad (variable-factor kron upscale + red corner-rays)

**Rule:** Input is always 5x5. factor f = (#distinct colours in the last row)+1,
f in 2..6. Output is (5f)x(5f) = the kron upscale of the input by f*f, overlaid
with red(2) at four diagonal corner-rays of length f emanating from the corners
of the upscaled 2x2 "box". Red is only ever drawn on background cells (0
conflicts over 3000 fresh). The box sits at (0,1)/(1,0)/(1,1); colours are
mirrored across the diagonal in the input.
**Current (prior custom):** 14.61 pts, custom:task107, mem 23733, params 8821
(a [15,24,24]=8640-param red lookup table + four 30x30 fp32/int32 index planes).
**Target tier:** B (label-map + final Equal). Tier S/A blocked: the output is a
data-dependent variable-factor kron upscale (f-dependent index = a Gather, not a
fixed conv/permute) overlaid with non-separable 45-degree diagonal rays.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior table-based (baseline) | B | 23733 | 8821 | 14.61 | — | baseline |
| 2 | separable double-Gather upscale of lab5[1,1,5,5] (kill 4x 30x30 index planes); keep red table | B | 8687 | 8794 | 15.23 | 200/200 | killed the index-plane floor |
| 3 | arithmetic red mask (kill 8640-param table) on 30x30 fp32 | B | 17421 | 151 | 15.23 | 200/200 | params->151 but fp32 diag planes cost 2x3600 |
| 4 | fp16 diagonal planes (R-C, R+C integer-exact in fp16) | B | 13945 | 151 | 15.45 | 200/200 | 2x3600->2x1800 |
| 5 | red on 24x24 fp16 canvas + Pad to 30 (red coord<24) | B | 13651 | 207 | 15.46 | 200/200 | small (Pad overhead ate most) |
| 6 | SENTINEL-VIA-GATHER: pad lab5->6x6 with 10, clip gidx to 0..5 (kills ingrid+final sentinel Where) | B | **11853** | **154** | **15.61** | **500/500** | BEST |

## Best achieved
**15.61 @ mem 11853 params 154 — fresh 500/500 (isolated temp-net).** Beats prior
14.61 by **+1.00**. Adopted? N (build-only per brief).

## 2026-06-28 source-control update
The deployed live graph is now substantially ahead of this old semantic note:
`memory=2924`, `params=1154`, `points=16.686638`.  `src/custom/task107.py` was
converted to a live-exact source builder, so future work should treat the older
11.85KB semantic builder as historical, not the active baseline.

## Irreducible-floor analysis
Dominant intermediates (total ~9.2KB of 11.85KB):
- **RmC, RpC [1,1,24,24] fp16 = 1152 each** — the two diagonal-distance planes
  for the arithmetic red mask. fp16 is exact here (R-C in [-23,23], R+C in
  [0,46]); diagonals are NOT row/col separable so a 2-D plane is required. The
  24x24 crop is the true active red extent (max red coord 23).
- **in5 [1,10,5,5] fp32 = 1000** — slicing the 10-channel 5x5 colour grid to feed
  the 1x1 colour-index Conv. Colours are arbitrary 0..9 so all 10 channels are
  needed; this is the irreducible "read the colour grid" cost.
- **L0, L [1,1,30,30] uint8 = 900 each** — the upscaled label and the
  red-overlaid label that feeds the final Equal. The 900 uint8 label map is the
  canonical label-map floor (cf task195).
- **red30u(900 u8) + red_b(900 bool)** — Pad emits uint8 (ORT Pad rejects bool),
  then a Cast to bool for the Where condition; two unavoidable 900 planes for the
  padded full-size red mask.

## OPEN ANGLES (re-attack backlog)
- Collapse red30u+red_b (2x900): apply red on a 24x24 SLICE of L0 with the 24x24
  bool directly (576), then reassemble to 30x30 — but the reassembly (Pad the
  24x24 + select against the rest of L0) likely re-introduces ~2 planes; net
  probably neutral. Worth a measured attempt for ~ -1000 -> ~15.7.
- Merge ondm/onda/ondiag (3x576): no obvious single-op form for an OR of two
  Equals against two different scalars.
- in5 1000: a per-channel MatMul that contracts the 10-ch axis straight off the
  free input into a [1,1,5,5] colour-index could dodge the 1000 slice, but the
  contraction weight still reads 10 channels — likely no win.

## 2026-06-29 re-attack: diagonal table arithmetic

Current source/live exact graph is **16.686638 @ mem 2924 params 1154**.
The largest apparent lever is `diag_points_table` (720 initializer elements).
I derived the exact formula for the 15 cases:
`f = num_changes + 2`, `k = min(arange(6), f-1)`, with four rays whose
row/col affine coefficients depend on `pos_idx`.  A temp ONNX replaced the
720-entry table gather with coefficient gathers and vector arithmetic:
`/tmp/task107_diag_arith.onnx`.

Stored result: exact `266/266`, but **16.360943 @ mem 4756 params 892**.  The
params drop (1154 -> 892) is more than erased by the new 24-wide int32
intermediates and coefficient-gather tensors.  Conclusion: the dense diagonal
point table is currently the better compression for ONNX's memory model.

Checked related decode angle: replacing the `GatherND -> MatMul -> Cast`
colour decoder with `ArgMax`/LUT or QLinearMatMul is structurally worse here.
`ArgMax` emits int64 and still needs a non-contiguous colour-value map
({1,3,4,5,6,7,8,9}); QLinearMatMul would require casting the gathered fp32
one-hot block to uint8 before multiplication, adding more counted memory than
the existing tiny fp32 `[1,6]` MatMul output plus uint8 cast.

## INSIGHT (transferable)
⭐⭐ **SENTINEL-VIA-GATHER**: for a variable-size upscale/crop whose out-of-grid
cells must be "no channel on", pad the small source grid with ONE extra
sentinel row/col (value = an unused index like 10) and CLIP the gather index to
include that sentinel slot. Out-of-grid positions clip to the sentinel index and
gather the sentinel automatically — this kills the entire separate in-grid mask
(rowin&colin, ~900B) AND the final sentinel Where (~900B). Saved ~1800B here.
⭐ **Variable-factor kron = separable double Gather of the SMALL source grid**:
upscale by data-dependent f via gidx=floor(arange30/f) (a tiny [30] vector),
Gather(src5x5, gidx, axis=2) then axis=3 — NEVER build a 30x30 fp32/int32 index
plane (that was the prior 4x3600 floor). Combined with the sentinel-gather trick
the whole upscale costs only the final 900B label plane.
⭐ **Diagonal-ray decoration = arithmetic R-C/R+C equality on an fp16 plane
cropped to the true active extent**: replaced an 8640-param [15,24,24] lookup
table; opposite corner-rays share one diagonal constant ((row-col)*f and
(row+col+2)*f-1), so only TWO diagonal planes (not 4) are needed.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 3813 -> 3694 (+0.032)
Mechanism: Einsum vs FREE input. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].