# task365 — e50d258f

**Rule:** A 10x10 grid holds 2-3 solid, gap-separated, axis-aligned rectangles ("boxes")
filled with blue(1)/cyan(8) plus a few red(2) cells. Each box has a DISTINCT red count
(sampled without replacement from {1,2,3,4}); the output is the box with the MOST reds,
cropped to its bounding box at the top-left of a fresh grid (rest all-zero / off-grid).
**Current (prior):** 14.81 pts, gen:vyank6322, mem 26613, params 68
**Target tier:** detection/B — needs per-box red ARGMAX + variable-size crop; selection is a
global argmax over data-dependent-count components, so it lands in the run-sum / detection band
(B-ish), NOT a clean separable (A) or single-op (S) rule.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | linear 5-step run-sum recurrence (L,R,U,D), winmask, gather-crop | B/det | 21914 | 474 | 14.98 | 266 stored | works, heavy |
| 2 | + segmented-doubling run-sums (offsets 1,2,4) | B/det | 20314 | 674 | 15.05 | — | leaner |
| 3 | + fp16 colour slice, drop redundant *nb / And / eqmax masks | B/det | 19614 | 674 | **15.08** | **200/200 + 500/500** | best, generalizes |

## Best achieved (NEW SESSION 2026-06-19 — WIN, crosses +0.3)
**15.455 @ mem 13282 params 691** — beats deployed 15.08 by **+0.38** (and prior 14.81 by +0.65).
Fresh ISOLATED 200/200, 500/500, AND genverify.fresh_pass 200/200 — GENERALIZES.
src/custom/task365.py. (Earlier session best was 15.0822 @ mem 19614, MARGINAL +0.27.)

### What cracked it (BR-corner two-scan, NOT the 4-sweep segtotal)
The 4-sweep box-red plane was the wall. KEY pivot: box-red is only needed at each
box's BOTTOM-RIGHT corner, and the winner is the UNIQUE-max BR (distinct counts).
So compute box-red at BR via TWO FORWARD segmented prefix scans (not 4 segtotals):
  rH = H-fwd-prefix(red)  (= box-row red at the run's right end)
  boxred = V-fwd-prefix(rH)  (= full box red at the box's BR cell)
Then winner = ReduceMax over BR corners; brBR==M marks the unique winning BR.
Box H,W come from TINY 1-D run-length scans of the winning row/col occupancy
(gathered at r1/c1) — NOT full-plane wid/hei scans. TL = (r1-H+1, c1-W+1).

### The plane-cost levers that stacked to -6.3KB
- ⭐ SHIFT AS ONE fp16 MatMul with a constant [G,G] shift matrix (M_d @ v for rows,
  v @ M_{-d} for cols): ONE plane per shift vs Pad+Slice (2) or Slice+Concat (2);
  matrix is sparse params (100 ea, cheap). Zero-fill is automatic (shifted-in rows
  of M are all-zero) — Gather can't zero-fill, MatMul can.
- ⭐ GATED HILLIS-STEELE WITHOUT HEAD FLAGS: precompute g1,g2,g4 (g_d[i]=1 iff cells
  i..i-d all occ) by idempotent doubling g2=g1*sh(g1,1), g4=g2*sh(g2,2) (occ in
  {0,1} so squares vanish). Then v += g_d * shift(v,d). Far fewer ops than the
  head-flag segmented scan; the >=1-cell gap means g_d alone stops cross-box bleed.
- ⭐ NO 30x30 COLOUR PLANE: colours are only {0,1,2,8}. colf = Conv(input[:,0:3]
  weights[0,1,2]) + 8*input[ch8], all on the 10x10 slice (1.2KB+small) — kills the
  3600B full-canvas colour Conv. red=(colf==2), occ=(colf>0).
- single-cell scalar reads via data-dependent Gather (occ row/col at r1/c1; run
  length at c1/r1) instead of mask-mul + ReduceMax full planes.

### Floor now
mem 13282: ~38 fp16 [1,1,10,10] scan/gate planes (~7.6KB, the two red prefix scans
+ their gates = irreducible 2x3-step doubling), + fp32 10x10 colour-build planes
(~2KB), inLo [1,3,10,10] 1.2KB, L uint8 30x30 900B (one-hot output carrier floor).

## Irreducible-floor analysis
Dominant memory = 68 fp16 [1,1,10,10] planes (13600 B) from the FOUR contiguous-run all-reduce
sweeps (L+R for horizontal red-run total, U+D for the vertical roll-up). Box-red total per cell
requires a segmented all-reduce on each axis = (forward scan)+(reverse scan)−self = 2 one-directional
run-sums per axis → 4 sweeps, each ~14 tensors even with log-step doubling (offsets 1,2,4; needed
because boxes reach 6 wide/tall). The ≥1-cell gap means a non-gated shift-by-2 would bleed across
adjacent boxes, so the per-step link/gate chain is mandatory.
Secondary: colf30 [1,1,30,30] fp32 = 3600 B — the standard "read colour from the 30x30 one-hot"
floor (slice-then-cast is cheaper than slicing input channels); + final padded uint8 label 900 B.

## OPEN ANGLES (re-attack backlog)
- **Halve the run-sums (the real lever, ~+0.3 to clear B threshold):** an integral-image of red via
  2 CumSum planes + box extents from run-LENGTHS (2 sweeps) would still be 4 sweeps; the genuine cut
  is a 2-CumSum area-sum evaluated per cell at its (corner, far-corner) — blocked by needing a
  data-dependent 2-D Gather (GatherND) of II at per-cell indices. If a cheap per-cell II-rectangle
  read exists, box-red drops from 4 sweeps (13.6 KB) to ~2 CumSum planes (→ ~mem 7 KB ≈ 16.3 pts).
- Reversal-batch trick (pack base+reverse on channel axis, share link) verified correct but
  BREAKS EVEN on bytes (2-ch plane = 2x size cancels the half-count).
- Avoid the 3600 colour read: only possible if the crop colour came from scalars (it doesn't —
  the box keeps its full 1/8/2 texture), so the 3600 stays.

## INSIGHT (transferable)
⭐ "global argmax over solid axis-aligned rectangles + variable crop" is FEASIBLE and beats the
gen-net WITHOUT flood-fill: per-component reductions become **contiguous-run all-reduces** (segmented
doubling, offsets 1,2,4 cover runs ≤8). Distinct per-box counts make the winning box the UNIQUE argmax,
so `winmask = (boxred == ReduceMax(boxred))` recovers its exact bbox FOR FREE (no run-length pass) —
1-D occupancy profiles of winmask give (min_row,min_col,H,W) scalars, then the task036 Gather-shift
crop idiom. The wall is that a 2-D segmented SUM still needs 4 one-directional sweeps (~13.6 KB of
fp16 10x10 planes); it lands at ~15.1, a MARGINAL +0.27 over the gen-net. To cross +0.3/reach B≈16.8
you need a CumSum integral image with a cheap per-cell rectangle read (blocked here by data-dependent
2-D Gather).


## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.024, policy-gated)

Clean adoption (candidate ≤ incumbent on every gate). Same BR-corner two-scan box-red
argmax + variable crop mechanism as the incumbent; only the detection Conv realization changed.

**Mechanism diff (op census, retired vs new):** single `Conv` → `QLinearConv` (int8),
`Cast` count 7→8. The two forward segmented prefix scans (rH, boxred), the unique-max BR
selection, the 1-D run-length H/W reads and the `Einsum`/`ArgMax`/`Where` crop assembly are
all unchanged (85→86 nodes). The int8 detection-conv output flows into the `TopK` rank via an
fp16 score (`safe_name_37_topk_f16`), so ranking stays exact.

**Cost:** mem 3908→3808, params 120→124, pts 16.6990→16.7231 (**+0.024**, cost 4028→3932 −96).

**Gate evidence:** bundled 266/266 fail=0 (both nets). Fresh 2×2000: candidate 0 fails,
incumbent 0 fails, 0 divergence. TopK audit: 1 TopK, data-input `safe_name_37_topk_f16`
= **FLOAT16** (grader-safe; not uint8).

**Backup + provenance:** incumbent → `reports/retired_networks/task365_pre_s10.onnx`;
candidate source `public_candidates/bobmyers7186/task365.onnx` → `networks/task365.onnx`;
source regenerated via live_to_exact_source --write-src, src↔live reconciled fail=0.

⭐ TRANSFERABLE: same int8-`QLinearConv`-on-a-fixed-detection-Conv-feeding-fp16-`TopK` lever
as task264 — a detection/B net with one fixed detection Conv whose output only feeds a TopK
rank can int8-quantize that Conv bit-safely. Selection: single fixed detection Conv → fp16
TopK/ArgMax, currently fp16/fp32. Confirmed twice this session (264, 365) on the bobmyers pack.
