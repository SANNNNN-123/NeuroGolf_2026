# task183 — 77fdfe62

**Rule:** Input is an (size+4)×(size+4) grid (size∈{2,4,6}) with a blue frame at rows/cols {1,size+2}, four corner colours at the outer corners (TL=in[0][0], TR=in[0][s3], BL=in[s3][0], BR=in[s3][s3] with s3=size+3), and cyan pixels inside the frame at in[r+2][c+2]. Output is a size×size grid where every cyan pixel (r,c) is recoloured by the QUADRANT it falls in relative to h=size//2 (TL/TR/BL/BR); non-pixel cells are background 0; everything outside size×size is unset.
**Current (prior):** 14.88 pts, gen:biohack_new, mem 24816, params 38
**Target tier:** A (separable quadrant recolour + scalar-driven small-region rebuild, no full-canvas working plane)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | scalar size + corner row-conv + 6×6 label → pad → equal | A | 10492 | 78 | 15.73 | 200/200 | ok |
| 2 | single-slice cyan + combined-row corner gather | A | 5732 | 80 | 16.33 | 200/200 | ok |
| 3 | GatherND(batch_dims=2) corner read (kills 2400B plane) | A | 4172 | 92 | 16.64 | 200/200 | ok |
| 4 | fp16 6×6 working planes | A | 3340 | 92 | 16.86 | 200/200 | ok |
| 5 | equal-then-pad-into-output (720B vs 900B carrier) | A | 3160 | 92 | 16.91 | 200/200 | ok |
| 6 | size = sqrt(total)−4 scalar (kills 120B row-profile) | A | 3044 | 93 | 16.95 | 300/300 | ok |
| 7 | nested-Where qcol (3 ops vs 9 muls) | A | 2432 | 93 | 17.17 | 300/300 | ok |
| 8 | TL via fixed Slice + 3-point GatherND (480B vs 640B) | A | 2184 | 106 | 17.26 | 500/500 iso | ADOPT-candidate |

## Best achieved
17.26 @ mem 2184 params 106 — beats prior 14.88 by **+2.38**. Isolated fresh 200/200 and 500/500.

## Irreducible-floor analysis
Dominant intermediates after optimisation:
- **Expand 480B** (int64 [1,10,3,2]) — GatherND batch_dims=2 needs one [row,col] index per (channel,corner); ORT rejects int32 indices and does NOT broadcast index batch dims, so the 10× channel replication is structural for reading data-dependent corner cells across all colour channels without a 30×30 plane.
- **Equal 360B + Cast 360B** ([1,10,6,6]) — the 10-channel one-hot must materialise once before Pad-into-output; Equal is always bool and ORT Pad rejects bool, forcing the uint8 Cast. 720B is the floor for routing a per-cell-coloured one-hot into the free output (beats the 900B pad-then-equal carrier).
Everything else is ≤144B (the cyan fp32 6×6 Slice) or fp16/uint8 small planes.

## OPEN ANGLES (re-attack backlog)
- Build the one-hot only over the 8 possible output channels {0,2..7,9} (never 1/8) and scatter-pad the two interior gaps — saves ~120B but needs interior-channel insertion (non-contiguous Pad), likely net-neutral.
- Reduce Expand: a GatherND formulation that contracts channels before the gather would kill the 480B, but every channel-reduction creates a ≥3600B plane — no win found.

## INSIGHT (transferable)
⭐ **GatherND(batch_dims=2) reads K data-dependent cells across ALL 10 channels with NO wide plane** — the index tensor [1,10,K,2] (Expand of a [1,1,K,2] built from a recovered scalar) is the only cost (~K·160B int64), vs ~2400B for a Gather of full rows. The classic "read a data-dependent corner/marker cell" pattern (which normally forces a 1200–2400B row/col slice) collapses to this. Reduce the point count by reading any FIXED-position cell with a plain Slice instead.
⭐ **size = sqrt(ReduceSum(input,[1,2,3])) − const** when the grid is a full k×k of one-hot cells (every cell sets exactly one channel ⇒ total sum = side²) — a 4B scalar, no row/col profile plane.
⭐ **equal-then-pad beats pad-then-equal** for per-cell-coloured outputs: build the [1,fewch,K,K] one-hot at the small active size, Cast uint8, then Pad-into-the-free-output — 720B vs the 900B full-canvas label carrier.
⭐ **nested Where for a separable 2×2 quadrant colour map**: qcol = Where(rowhalf, Where(colhalf,TL,TR), Where(colhalf,BL,BR)) — ONE K×K plane + two tiny row vectors, vs 8 outer-product Muls + a Sum.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 1048 -> 1025 (+0.022)
Mechanism: single Einsum. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].