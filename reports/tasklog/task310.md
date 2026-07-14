# task310 — c909285e

**Rule:** A `size`x`size` grid (size 20..30) is filled with 3..4 "wires": each
wire (colour, spacing) sets every cell with `(r+1)%spacing==0 OR (c+1)%spacing==0`
to that colour (later wires overwrite earlier). Then a square box of side
`boxlength` (5..8) at (boxrow, boxcol) has its PERIMETER drawn in `boxcolor` (a
colour NOT used by any wire). Output = the box subregion
`grid[boxrow:boxrow+L, boxcol:boxcol+L]` (L=boxlength) — wire colours plus the
boxcolor perimeter — cropped to the top-left of a fresh grid; everything outside
the LxL box is all-channels-off. Box id: wire colours fill whole rows/cols so
their bbox span is ~size (>=15); boxcolor only spans the box (span = L-1 <= 7),
so boxcolor = the non-bg colour with MINIMUM bbox span, and
(boxrow, boxcol, L) = (minrow, mincol, rowspan+1) of that colour.

**Current:** 14.33 pts (manifest) / P=14.84 given, public base net, mem high
**Target tier:** B — data-dependent variable-size crop+translate-to-origin
carrying arbitrary per-cell colours. Tier S/A blocked: output colour per cell is
an arbitrary per-instance value read from a data-dependent window (not
row⊗col-separable, not a fixed linear function of the local one-hot); the crop
size and position are both data-dependent. NOT a BAIL: the output size is
recoverable (L = boxcolor span + 1) and box id collapses to a closed-form 1-D
per-channel argmin span (task036 idiom), no flood-fill / connectivity.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | min-span colour id (task036) + colf colour-index window Gather + LxL gate + Pad + Equal | B | 11916 | 134 | 15.603 | 200/200 | win |

## Best achieved
15.603 @ mem 11916 params 134 — adopted? N (orchestrator gates). Beats prior
14.33 by **+1.27** (vs given P=14.84 by **+0.76**, >= +0.3). GENERALIZES: stored
266/266, ISOLATED fresh 200/200 against freshly-generated instances.

## Irreducible-floor analysis
Dominant intermediates: (a) `colf` [1,1,30,30] fp32 = 3600B — the per-cell
colour-index plane (Conv output must match fp32 input; casting input to fp16
costs an 18000B 10-ch plane). The crop WINDOW position (boxrow,boxcol) is
data-dependent so the full plane must be materialised before the window Gather
(circular dependency, same as task177). (b) two `rowocc`/`colocc`
ReduceMax(input) profiles [1,10,30,1]+[1,10,1,30] = 1200B each — PER-CHANNEL
occupancy is required to find which colour has min span; they stay fp32 because
ReduceMax inherits the fp32 input dtype. (c) `Vr` [1,1,8,30] fp32 = 960B (a
WORK-row window still spans 30 cols; gathering cols-first gives [1,1,30,8], same
cost). (d) `Lfull` [1,1,30,30] uint8 = 900B (full-size for the final 30×30
Equal; Pad rejects bool so cannot Equal-then-Pad). These four are the same
structural floor that pins the task036/task177 crop-and-translate family at
~15.4–15.6.

## OPEN ANGLES (re-attack backlog)
- Cast colf->fp16 and gather the window in fp16 (Vr 960->480, Vw->128): adds an
  1800B fp16 colf plane while saving only ~600B on the window — net LOSS here
  because the windows are already tiny. Only helps when many downstream
  full-canvas ops run on the plane (task377), which this task does not have.
- Shrink the two 1200B per-channel occupancy ReduceMax: would need to identify
  boxcolor without per-channel spans (e.g. a single banded statistic that
  separates "full-grid wire" from "small box"); untried, ~0.1 pt if found.
- Fused GatherND to select colour channel AND window in one shot to skip the
  3600B colf plane — same untried angle as task036/task177.

## INSIGHT (transferable)
⭐ A grid saturated with full-row/full-col "wire" lines plus ONE small marked box
is NOT a detection wall: the box's distinguishing colour is the MINIMUM-bbox-span
non-background colour (wires span ~size, the box spans <=7), recovered by the
task036 per-channel 1-D argmin-span idiom. For a SQUARE data-dependent crop the
output size falls out for free as `span+1` of that colour — no NonZero/ArgMax of
output extent needed. Combine task036's min-span colour id with task177's
colf-window-Gather (carry arbitrary per-cell colours) to handle "crop a marked
region that contains arbitrary colours" in closed form at the ~15.6 B-tier floor.
