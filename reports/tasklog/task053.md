# task053 — 25ff71a9

**Rule:** size=3 (3x3) grid; 2-3 cells painted one colour (1 or 2) at rows sampled
from all_pixels(3,2) -> rows in {0,1} ONLY. Output shifts every painted cell DOWN
one row: out[r+1][c]=colour. Source rows {0,1} -> land in {1,2}, never fall off.
As a full-one-hot per-row remap: out0<-bg row, out1<-in0, out2<-in1, out i<-in i (i>=3).
**Current:** 21.60 pts, single Gather(input, idx[30], axis=2), mem 0, params 30
**Target tier:** S (pure spatial copy/permute, mem 0) — already there structurally.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Gather axis=2 idx[30] (= current form) | S | 0 | 30 | 21.60 | 200/200 | at floor |
| 2 | Pad shift-down [0,0,1,0,0,0,-1,0] | S | 0 | 8 | — | FAIL | corrupts ch0 (top row->0, must be all-bg) |
| 3 | Range(3,30,1)+Concat([2,0,1])+Gather | S | ~456 | ~6 | ~18.9 | — | mem of int64 index intermediates sinks score |

## Best achieved
21.60 @ mem 0 params 30 — adopted? N (equals current). Beats prior 21.60? N (tie).

## Irreducible-floor analysis
Output is [1,10,30,30] -> 30 rows along the gathered axis -> a single axis-2 Gather
REQUIRES a 30-element index. params count ELEMENTS (dtype-free), so 30 params is hard.
The background channel (ch0) must be all-1 in the new top row AND in every untouched
row, so every output row must be SOURCED from a real all-background input row — this
rules out Pad/Conv shift (they fill the new top row with 0 in ch0). Computing the
index with Range+Concat cuts the initializer to ~6 params but materialises a 30-elem
int64 gather index (240B) + 27-elem Range output (216B) as intermediates; mem then
dominates and the score falls to ~18.9. No single mem-0 op admits a <30-element
initializer for a 30-row axis remap. Need m+p<=22.2 for +0.3; floor is 30.

## OPEN ANGLES (re-attack backlog)
- None viable. The 30-element row-remap index is the structural floor for any mem-0
  single-op solution, and every multi-op decomposition pays index-intermediate mem
  that exceeds the entire 22.2 budget. AT FLOOR.

## INSIGHT (transferable)
⭐ A whole-grid one-hot ROW/COL SHIFT that must preserve the background channel is a
mem-0 single Gather whose index length == the output extent along that axis (30) —
this is IRREDUCIBLE in params and CANNOT be undercut by Pad/Conv (those zero ch0 in
the inserted line) nor by computed indices (Range/Concat add index-tensor mem that
dwarfs the saved params). "Shift + preserve background one-hot" => 30-param Gather floor.

## S9 (2026-07-03) — mechanism-14 probe: REJECTED (240 > 30)
Single Gather 30-elem index = floor (documented INSIGHT re-confirmed). Scan's zero-pad
premise also wrong: output row 0 = background sourced from input row 2, not zeros.
