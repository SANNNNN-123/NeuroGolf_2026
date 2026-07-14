# task224 — 928ad970

**Rule:** Input is an H×W grid (H,W ≤ 16) holding (a) a small inner rectangle
perimeter in one non-gray colour c, and (b) exactly 4 gray(5) marker pixels —
each sitting one cell OUTSIDE one edge of a larger outer rectangle. Over the 4
markers: top = min(gray_row)+1, bottom = max(gray_row)−1, left = min(gray_col)+1,
right = max(gray_col)−1. Output = input plus the PERIMETER of the outer rectangle
[top..bottom]×[left..right] painted in colour c (gray markers + inner box kept).
Per-cell output colour: 5 where input gray; c where input has colour c OR cell on
outer perimeter; 0 in-grid background; (no channel) outside the H×W grid.
**Current:** was 14.28 pts (public gen:thbdh6332). Now 16.11 pts, label-map+Equal, mem 7162, params 87.
**Target tier:** B (label map + final Equal). Box position is a GLOBAL aggregate of
the 4 gray markers (min/max gray row & col) → non-local → Tier S (single Conv) is
out. Gray markers are 4 scattered points → not row×col separable as a whole →
Tier A out. B is the highest admissible tier; pushed hard below the nominal floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label map, full 30×30, ch0/ch5 slices + 10-ch Mul | B | 22874 | 102 | 14.96 | — | works, mem high (36k Mul + 3 fp32 planes) |
| 2 | drop Mul; Gather ch c; ingrid via 1-D occupancy | B | 17782 | 91 | 15.21 | — | 2× 3600 fp32 planes |
| 3 | single 1×1 Conv [0..9] → colour-value plane (cval) | B | 13874 | 103 | 15.46 | — | Cast(cval)→uint8 IS the label base; drops gray/inner eq-tests |
| 4 | 16×16 working canvas + Pad to 30×30 | B | 11098 | 89 | 15.68 | — | every per-cell plane 900→256 |
| 5 | drop r10/c10; gray-occ from cval, ingrid direct | B | 10250 | 91 | 15.76 | — | |
| 6 | uint8 label base (Cast cval30) + Equal-on-uint8 gray | B | 9038 | 91 | 15.88 | — | |
| 7 | replace Conv with two 16×16 channel slices (ch0,ch5) | B | 7866 | 88 | 16.02 | — | 2×1024 < Conv 3600+cast 900 |
| 8 | drop inner&ingrid And; gray-occ from fp32 slice direct | B | **7162** | **87** | **16.11** | **200/200** | FINAL |

## Best achieved
**16.11 pts @ mem 7162, params 87 — 266/266 stored, fresh 200/200.** Adopted? N
(main adopts). Beats prior 14.28? **Y (+1.83).**

## Irreducible-floor analysis
Dominant intermediates: two fp32 channel slices `gray16`/`bg16` = [1,1,16,16] =
1024 B each (the only per-cell read of colour identity from the input — channel 0
= background, channel 5 = gray; colour c is a scalar), plus the 30×30 uint8 label
`Pad` = 900 B (the output canvas, unavoidable since `output` is 30×30). Remaining
~13 bool/uint8 256-B planes are the perimeter build (horiz/vert/perim) and the
3-level label Where chain. Slice preserves the fp32 input dtype, so the two 1024-B
planes can't be made fp16/uint8 without first casting the input (a [1,10,30,30]
plane ≫ the slices). This is below the nominal Tier-B 16.8 ceiling already.

## OPEN ANGLES (re-attack backlog)
- **Drop bg16 (the 2nd 1024 slice):** find "not background" without reading ch0.
  Inner-box = cell with a non-bg/non-gray colour; if a cheap per-cell "any colour"
  signal can be had at ≤256 B (vs a 3600 ReduceMax-over-channels), one slice goes,
  → ~+0.1 pt. (ReduceMax over channels is 3600 → net worse; needs a cleverer read.)
- **Shrink the Pad (900):** if the active region could be cropped tighter than
  16×16 per-instance (it can't — generator max is exactly 16), or if the final
  Equal could run on a packed canvas. Marginal.
- **Halve the two slices to fp16:** only wins if input is cast to fp16 selectively
  for just channels {0,5} at ≤1024 B total — no opset-11 op does a 2-channel fp16
  gather-cast cheaply. Untried in detail.

## INSIGHT (transferable)
⭐ **Two cheap channel slices can beat a "single elegant Conv".** A 1×1 Conv
[0,1,…,9] over one-hot input gives the full per-cell colour value in one op, but
its output is a full 30×30 fp32 plane (3600 B) plus a uint8 cast (900). When the
rule only distinguishes a FIXED small set of colours per cell (here: background
ch0 + gray ch5, with the variable colour c handled as a scalar), slicing just
those K channels at the bounded working region (K×1024 B at 16×16) is cheaper than
the Conv whenever K is small. Count the channels the per-cell logic actually needs
before reaching for a colour-value Conv.
⭐ **Reduce the fp32 occupancy slice directly — don't cast-to-fp16 first.** A
[1,1,16,16] fp32 mask reduces to a [1,1,16,1] (64 B) 1-D occupancy with one
ReduceMax; an intermediate Cast→fp16 of the 16×16 plane (512 B) is pure waste.
⭐ **An "& ingrid" gate is redundant when a final `Where(ingrid, …, sentinel)`
runs last** — the sentinel overwrites every outside-grid cell regardless, so any
spurious inside-mask truth outside the grid is harmless. Saves a 2-D AND plane.

## S17 (2026-07-06) — dtype-overpay recast (bit-identical safe golf, +dtype_overpay_scan)
task224 row_has: replaced Sign(row_sum)→fp32 with Greater(row_sum,0)→bool + Cast→uint8 (row_sum∈{0,1,2}, threshold clamp needed); ArgMax on {0,1} identical. 2831→2772 (−59).
Gate: evaluate bundled fail=0 + **bit-identical outputs** over all train/test/arc-gen (verified). Safe for both tracks + private LB.
⭐ TRANSFERABLE: only ACTIVATION (node-output) dtype narrowing saves grader bytes — params counted by element-count (dtype-independent). Narrow the PRODUCER (upstream Cast/init dtype), never a post-Cast. Blocked when the plane is derived from / contracted with the free fp32 `input` (Einsum-vs-input, Slice/Conv of input, ScatterND updates vs fp32 data) → those force fp32. See [[neurogolf-fp16-count-plane-recast]].
