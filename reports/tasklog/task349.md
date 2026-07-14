# task349 — db93a21d ("death stars")

**Rule:** Input shows only MAROON(9) square centers, one per death-star; each is a
solid 2r×2r block (WIDTH always exactly 2r — `col∈[0,size-2r]`; HEIGHT may be
clipped at the top/bottom edge). Output redraws, per star: the MAROON center
(copied), a GREEN(3) halo = the center block Chebyshev-DILATED by r (→ a 4r×4r
square centered on the block), and a BLUE(1) beam filling the center's columns
from just below the block to the bottom edge. Per-cell priority MAROON>GREEN>BLUE>bg.

**Current installed/source-owned exact:** 14.826867 pts, mem 26100, params 90.
2026-06-28: `src/custom/task349.py` was re-synced to the installed live graph;
the older semantic source scored 14.389 / mem 39600 / params 960 and is
superseded for implementation, though its analysis below remains useful.
**Target tier:** detection/B — variable-per-object dilation + downward beam over a
full-size (10..30) canvas; no crop (grid IS the data region).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | conical grayscale dilation of a radius field (R=Σ ge(2k)) | B | 79890 | 1868 | 13.69 | — | correct but heavy |
| 2 | green = OR_d dilate_d(run≥2d) via single fused MaxPool/d; inclusive-blue | B | 39600 | 960 | 14.39 | 200/200 | best |

## Best achieved
14.826867 @ mem 26100 params 90 — current installed/source-owned exact baseline.
The 14.39 semantic implementation is retained only as analysis.

## Irreducible-floor analysis
GREEN dominates: it needs per-radius detection AND per-radius dilation.
- `green = OR_{d=1..5} dilate_d( run-length-of-maroon ≥ 2d )`. The cleanest form
  fuses spread+2D-dilate into ONE MaxPool per d:
  `cv_2d = Conv(maroon, ones[1,1,1,2d])` (window sums) and
  `green_d = MaxPool(cv_2d, kernel[2d+1,4d], pads[d,3d-1,d,d]) ≥ 2d`
  (a width-2d window sums to ≤2d, so "≥2d within reach" == "a full 2d run lies
  within Chebyshev d"; horizontal reach window x'∈[x0-3d+1, x0+d]).
- That is **5 conv fields + 5 maxpool fields = 10×900 fp16 = 18000B**, and this is
  the measured floor: radius r∈{1..5} all occur at size 30 (verified by fresh
  histogram), so all 5 levels are mandatory. Element count (not dtype) is the wall
  — fp16 is already the min for Conv/MaxPool (uint8 rejected); stacking d on the
  channel axis keeps the same element count. Conical-dilation and cumulative-
  dilation reformulations BOTH cost MORE (they un-fuse the spread from the dilate).
- Plus one mandatory fp32 entry plane (3600): `colf = Conv(input, w)` with
  w[ch0]=1, w[ch9]=2 → {0:off-grid, 1:in-grid bg, 2:maroon} gives BOTH the maroon
  mask and the in-grid mask from a single 10→1 reduction.
- The remaining ~16k is: m(fp16,1800), colcum blue-OR (fp16,1800), the 5-way bool
  OR for green (9×900=8100), and the uint8 compose chain (~3600).

## OPEN ANGLES (re-attack backlog)
- Halve the GREEN dilation by computing it at 15×15 and upscaling ×2 — green
  regions are ≥4×4 solid squares, but their EDGES are not 2-aligned, so naive
  downsample/upsample loses exactness; would need an edge-correction pass.
- Cheap run-length R via CumSum (run isolated) to drop to a single dilation pass —
  blocked: CumSum gives no segment-reset; cummax/cummin not in opset-11, and the
  centered-window-sum conv is contaminated by horizontally-near same-row blocks
  (verified: K=11..19 all fail).
- Collapse the 9-plane bool green-OR: every alternative (fp16 Max-tree after
  per-d Sub, conv-bias normalisation, ReduceMax over channel-stacked gp) costs ≥
  as much (Max/Sub planes are fp16 1800 vs bool 900). Truly 5-way OR = 9 tensors.

## INSIGHT (transferable)
- ⭐ **Variable-radius square dilation `dilate_r(run≥2r)` in ONE MaxPool per r**:
  `MaxPool(Conv(m, ones[1,2r]), kernel[2r+1, 4r], asym pad)` ≥ 2r fuses the
  "full-run detection (erode+spread)" and the "2D dilate-by-r" into a single pool.
  Beats conical (5 iters of pool−1+max) and cumulative-dilation whenever the per-r
  radius classes are needed — those un-fuse the spread, costing strictly more.
- ⭐ **One fp32 1×1 Conv can carry BOTH a colour mask and the in-grid mask**: when
  the input alphabet is tiny (here {bg=0, maroon=9}), weight ch0→1, ch9→2 so
  colf∈{0:off-grid,1:bg,2:maroon} — one 3600B plane replaces a separate
  ReduceMax-occupancy plane.
- ⭐ **Inclusive prefix-OR beats strict** for a "beam below an object" when a
  higher-priority Where overwrites the object's own cells: drop the shift entirely
  (cells above the object have no object above → stay 0; object cells get
  overwritten by the object colour). Saved the slice+pad shift planes.
- FLOOR: a multi-object task that needs BOTH per-object size detection AND
  per-object-size dilation over a full uncroppable canvas floors at ~2·5·900 fp16
  green planes ≈ 18kB; +0.3 is not reachable without a sub-resolution trick.

## 2026-06-28 high-score frontier check

Not a 20+ candidate under the current operator set.  The installed graph already
uses the right family — uint8/QLinearConv run gates plus MaxPool/Max composition.
The semantic wall is variable-radius halo dilation plus downward beam priority
over a full uncroppable canvas.  A single Conv/QLinearConv cannot recover radius,
dilate, beam-fill, and priority-compose with `mem+params <= ~148`.

## 2026-06-30 (S7) — LANDED refit, fresh-gated
Fresh generators are present (/tmp/arc-gen, arc_id db93a21d). The held
reports/candidates/task349_refit_19800.py was fresh-gated: 2500/2500 fresh
instances candidate fail=0 AND candidate==incumbent (the earlier "random 400/400
mismatch" was OFF-DISTRIBUTION garbage, not valid task instances). Bundled 267/267
fail=0. LANDED: mem 26100->19800, params 90, pts 14.827->15.102 (+0.275).
Mechanism: one fp32 1x1 Conv carries maroon+valid; halo detection kernels scaled
x3 to emit colour directly into a Max compose (fuses away bool-cast+Where planes).

# (appended) S8 2026-07-02 — parallel-plane conv-count union (+0.092) ADOPTED, bit-identical
5 per-radius dilation MaxPools (4500B) → ONE QLinearConv gw[1,5,11,20] (per-radius windows as
channels) + gsum>0 union via Min(gsum,3) (u8 Min IS supported in ORT CPU). 5 detector
QLinearConvs → one [5,1,1,12] stack. 15300+1191 vs 18000+90 → 15.197→15.289. Fresh 2500+1500
div 0; 400 random vs deployed onnx div 0. Full walk-einsum priced and REJECTED: radius-gated
growth needs phase-gated shift tensors ≥7-9k params vs 4.5KB removable — parallel per-radius
planes collapse via CONV-CHANNEL UNION, not walks (registry pattern for parallel banks).
Floors: colf 3600 fp32 entry, hp_all 4500.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 16491 -> 15042 (+0.092)
Mechanism: QLinearConv signed renderer.
Gate (fresh_verify, inc/cand fail on 1500-2000): 0/0 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 