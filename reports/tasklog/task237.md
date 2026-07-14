# task237 — 99fa7670

**Rule:** A (width×height) grid (both ∈ 3..9) holds scattered seed pixels, at most one per row
(seed rows strictly increase by 2 or 3; seed cols ∈ [0, width-2]). For each seed (r,c,color):
(a) horizontal fill rightward — `output[r][col]=color` for col∈[c, width-1]; (b) vertical fill down
the last column — `output[row][width-1]=color` for row∈[r, height-1]. Vertical fills are applied in
ascending seed-row order, so the last column at row `row` carries the color of the seed with the
LARGEST seed-row ≤ row (forward-fill, most-recent wins; running-max matches only ~52%). Off-grid stays bg.
**Current:** 15.43 pts (P), ext:kojimar6275, mem 14283, params 41 (blank "skip-marginal" note).
**Target tier:** A (CumSum row-fill + tiny segment-MatMul forward-fill; routed into FREE bool output).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 (prior session) | suffix-fill `colf@triu` + plateau forward-fill `E@vals`, 9-ch slice | A | 7636 | 221 | 16.03 | 500/500 | logged, NOT adopted to manifest |
| 2 | CumSum horizontal + segment-MatMul forward-fill, 10-ch slice + occ plane | A | 7920 | 55 | 16.02 | 500/500 | superseded |
| 3 | + slice channels 1..9 only; separable rowin/colin via ReduceMax(input,[1,3])/[1,2] (no 2-D occ plane); 1-D last-col mask | A | 6886 | 58 | 16.15 | 500/500 | **ADOPT (+0.72)** |

## Best achieved
16.15 @ mem 6886 params 58 — beats P (15.43) by **+0.72** (≥+0.3 ✅). Fresh 500/500, stored 266/266.

## How it works
- Off-grid is all-zero across channels; in-grid bg sets ch0=1. `rowin=ReduceMax(input,[1,3])`→[1,1,30,1]
  and `colin=ReduceMax(input,[1,2])`→[1,1,1,30], sliced to 9, give a separable solid mask with NO 2-D
  occupancy plane. `lastmask = rowin AND NOT shift_left(colin)` = rightmost in-grid col per in-grid row.
- `colf` = Conv(input[:,1:10,:9,:9], weight=[1..9]) → [1,1,9,9] colour index (ch0 dropped from the slice).
- HORIZONTAL ray = `CumSum(colf, cols)` — exactly one nonzero per row ⇒ prefix sum is the rightward fill;
  mask by `ingridf` to stop at the grid's right edge. ONNX has CumSum but no cum-max, and CumSum suffices.
- `lastcol = ReduceMax(hor, cols)` = per-row seed colour (no need to know W).
- FORWARD-FILL down last column = segment MatMul: `grp=CumSum(lastcol≠0)`, `M[i][j]=(grp[i]==grp[j])`,
  `ff = M @ lastcol` selects each row's segment-start seed colour (later/lower seed wins). `ff *= rowin`.
- `res = Where(lastmask, ff, hor)`; uint8; `Where(ingrid, res, 99)` sentinels off-grid; Pad→30×30 with 99;
  `Equal(L, arange[0..9])` → BOOL output (off-grid + padding all-False = all channels 0).

## Irreducible-floor analysis
Dominant counted intermediate: the 9-channel 9×9 fp32 input slice `x19` (2916B) — reading colour needs a
fp32 multi-channel region; ORT upcasts fp16/uint8 and Conv rejects uint8, so it can't shrink. Second: `L`
(uint8 30×30 = 900B, the padded label needed for the final Equal broadcast). Everything else is ≤324B 9×9
or ≤120B vectors. `output` is FREE (graph output). mem+params=6944 → ~16.15.

## OPEN ANGLES (re-attack backlog)
- Cast the six 324B fp32 9×9 planes (colf/hcum/hor/ingridf/M_f/res) to fp16 (162B each) → ~−1KB ≈ +0.13.
  Risk: CumSum in fp16 under ORT_DISABLE_ALL untested; MatMul/Equal-chain fp16 9×9 already proven (attempt 1).
- Drop `x19`'s 2916B by contracting channels on the FULL input then slicing a single-channel colf — net
  worse (colf30 = 3600B). No sub-2916 single-channel colour read at 9×9 currently exists.

## INSIGHT (transferable)
⭐ A "ray from a single per-row seed to the row's end" = plain `CumSum` along that axis (prefix sum =
suffix fill when there is exactly one nonzero per row) — dodges the missing cum-max op entirely.
⭐ FORWARD-FILL (carry most-recent nonzero down a sparse axis, later-wins) is closed-form with NO Scan:
`grp = CumSum(nonzero)` then `ff = Equal(grp_i, grp_j) @ values` — the segment-id equality matrix groups
each plateau and `values` (nonzero only at each segment's seed) selects it. Exact on length-≤K axes.
⭐ Another blank-note "skip-marginal" label that was wrong (6/7 wins this wave).


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 1901 -> 1838 (+0.034)
Mechanism: Einsum. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].