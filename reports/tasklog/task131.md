# task131 — 56dc2b01

**Rule:** Grid has one FULL red(2) line (a complete column, or a complete row if xpose) and a
green(3) "creature" (continuous_creature of 8-9 cells, fits in an H×H≤5×5 box) on one side of it.
Output: red line stays put; the green creature is translated (shape preserved) perpendicular to the
line so its bounding-box edge nearest the line sits exactly ONE cell from it; a full CYAN(8) line is
drawn one cell beyond the creature's FAR bbox edge; original green erased. Geometry (red col rc, green
col-bbox [cmin,cmax]): green left of red (cmax<rc) -> shift s=rc-1-cmax, cyan=cmin+s-1; green right ->
s=rc+1-cmin, cyan=cmax+s+1. Horizontal-red (xpose) is the transpose case; flip just relocates green.
Generator bounds: height 4..5, width 16..18, offset 0..3, redline width-6..width-2.

**Current:** 15.28 pts, custom:task131 (prior), mem 14668, params 1940
**Target tier:** A — separable masks + a single 2-D green shape (creature not row⊗col separable, so not
Tier S); the green translate forces one genuine 2-D plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior (M=20 square canvas, 6 agg Convs) | A | 14668 | 1940 | 15.28 | 200/200 | baseline = P |
| 1 | in-grid via ReduceSum (free), R+G one 2-ch Conv/axis | A | 11076 | 1331 | (broke) | 101 | off-grid rows filled bg |
| 2 | +row in-grid mask (Hc), canonical canvas [5,18] | A | 11535 | 1336 | 15.54 | 200/200 | ok, +0.26 marginal |
| 3 | slice line vectors [30]->[18] (halve scalar planes) | A | 10791 | 1327 | 15.60 | 500/500 | ADOPT, +0.315 |
| x | shrink green slice to 8×8 square | — | 8751 | 1331 | (broke) | 97 | flip puts green at col~17, 8 too small |

## Best achieved
15.60 @ mem 10791 params 1326 — adopted? Y (candidate left at src/custom/task131.py). Beats prior 15.28? Y (+0.315).

## Irreducible-floor analysis
Dominant intermediates: Gpc (1296B, [1,1,18,18] fp32 green slice — Slice keeps input fp32 dtype, and the
18×18 square is forced because flip can place the creature anywhere across the WR=18 width so an 8×8 crop
fails) and L (900B, the [1,1,30,30] uint8 label needed for the final Equal->output). After those, ~8
square [18,18] uint8 planes (324B each) for the two orientation transposes (green-canonicalise +
label-uncanonicalise), each needed because xpose maps [H,W]<->[W,H] and a single square holds both. The
canonical canvas is genuinely 5×18 (gen bounds), so all the bulk masks/label run at 90B. Not a hard floor
— see open angles.

## OPEN ANGLES (re-attack backlog)
- Un-flip BEFORE cropping the green: detect flip (green on opposite side of red) and reverse the col
  ramp into the green Gather, so the green source collapses to an 8-wide window -> Gpc 1296->~256 and the
  green transpose planes 324->64 (est. another ~0.1-0.15 pts). Risk: extra flip-direction scalar logic.
- Replace the two 324B orientation transposes by routing orientation into the col/row ramps (build the
  label directly in output coords with a per-axis remap vector) — would remove ~6 square planes.
- The 6 squeeze [30] intermediates (120B each, 720B) survive ahead of the [18] slices; slicing the 4-D
  Conv/ReduceSum outputs to width 18 BEFORE squeeze drops them (~0.05 pts).

## INSIGHT (transferable)
⭐ When a task canonicalises via xpose+flip, the CANONICAL frame's true bounds (here 5×18 from the
generator's height/width ranges) are far smaller than 30×30 — do ALL 2-D assembly on that small canvas
(90B planes) and only pad up at the very end. But the square needed for the orientation transpose is set
by the LONG axis (18), not the creature size, and FLIP defeats any attempt to crop the moved object to
its own small bbox (it can be relocated anywhere across the long axis). Also: per-line aggregates that
sum a colour channel are cheaper as ONE multi-output-channel Conv per axis than N single-channel Convs,
and in-grid (sum over ALL channels) is a zero-param ReduceSum over the free input, never a Conv.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
