# task166 — 6d75e8bb

**Rule:** Input has cyan(8) left-aligned horizontal strips forming a box (rows brow..brow+n-1,
row r filled cols bcol..bcol+lengths[r]-1) plus ONE extra cyan "marker" pixel inside the same
column band; the only colors are bg(0) and cyan(8). The output fills the axis-aligned bounding
rectangle of all cyan cells with red(2) wherever the cell is background, keeping cyan as cyan.
Final flip_horiz/transpose re-orient input+output identically so the bbox statement is invariant.
Verified 0/300 + 0/30000 fresh: `box=bbox(cyan); out=input with (input==0 & box)->2`.

**Current:** 16.20 pts (existing file: full-30x30 fp32 cyan Conv + raw rowhas⊗colhas box), mem 6600, params 22
**Target tier:** A — closed-form separable bbox-fill routed into the FREE Where output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | existing: 30x30 fp32 cyanf + box=rowhas&colhas + notcyan + redmask + Where | A | 6600 | 22 | 16.20 | — | baseline |
| 2 | crop all per-cell work to fixed 13x13 window; pad redmask u8->bool to 30x30 for Where cond | A | 3282 | 30 | 16.89 | 200/200 | adopted-as-best |

## Best achieved
16.89 @ mem 3282 params 30 — adopted? N (per instructions, not self-adopted). Beats prior 16.20? Y (+0.69).

## Irreducible-floor analysis
Dominant intermediates are the TWO unavoidable 30x30 planes at the very end: the Pad output
(uint8, 900B) and the bool `cond` (900B) — the Where condition MUST be 30x30 to broadcast against
the [1,10,30,30] input, and Pad rejects bool so the u8-pad -> bool-cast pair is the cheapest bridge.
Everything else now lives on the 13x13 active window (cyan slice 676B fp32, three bool masks 169B
each). The box = rowhas⊗colhas outer product is exact WITHOUT a prefix/suffix-OR scan because every
box row holds a strip (length>=1) and every box col is hit by the longest strip — no interior gaps.

## OPEN ANGLES (re-attack backlog)
- Collapse the two 30x30 tail planes (1800B) into one: a single-op route from a 13x13 mask to a
  30x30 bool Where-cond without a separate Pad+Cast would cut ~900B (~+0.3 more). Pad-rejects-bool
  is the blocker; an Expand/Scatter or ConstantOfShape placement might beat it but likely costs more.
- Tighten the crop below 13 only if generator bounds prove cyan never reaches row/col 12 — they do
  reach 12 (brow+n-1 <= height-2 <= 12, bcol+maxlen-1 <= width-2 <= 12), so 13 is the proven floor.

## INSIGHT (transferable)
⭐ The "3600B fp32 plane floor" is escapable via BOUNDED-ACTIVE-REGION even when the grid SIZE is
variable: if the generator bounds where the active content can sit (here brow/bcol + run-length caps
pin every cyan cell to rows/cols 0..12), crop the per-cell plane to a FIXED conservative window
(13x13 = 676B) and only Pad the final Where condition back to 30x30. A task sitting exactly at its
public 16.20 with a full-30x30 fp32 plane is the classic signature of a missed crop — re-derive the
content's coordinate bounds from the generator before accepting the floor (16.20 -> 16.89, +0.69).
Also: a bbox is the raw rowhas⊗colhas outer product (no prefix/suffix-OR) whenever every box row and
every box column is guaranteed occupied.
