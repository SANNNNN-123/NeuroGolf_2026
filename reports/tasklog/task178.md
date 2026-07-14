# task178 — 746b3537

**Rule:** Input is a grid of solid colour bands of varying thickness stacked along
one axis (rows for xpose=0, cols for xpose=1). Output lists the distinct band
colours in order, each collapsed to a single cell, as a single column (xpose=0)
or single row (xpose=1). Generator bounds: width 1..5, 3..5 colours, thicks 1..3,
so band-axis length = sum(thicks) <= 15, cross axis <= 5, output length <= 5.
Consecutive band colours are forced distinct, so every colour change starts a new
output slot and #runs == #colours. The FIRST line along the cross axis (row0 for
xpose=1, col0 for xpose=0) already carries every band colour in order.

**Current:** 16.50 pts, run-length-into-line (first-line read + cumsum dedup),
mem 4791, params 122 (was 15.48 / mem 13569 stored).
**Target tier:** B — closed-form run-length compaction; no per-cell 30x30 plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | stored net (fp32, 5x5 blocks) | B | 13569 | 87 | 15.48 | - | baseline |
| 1 | fp16 pipeline + uint8 5x5 blocks + Pad(uint8,opset11) | B | 7755 | 120 | 16.03 | - | win |
| 2 | same=sum(line*prev) (drop Sub/Abs) | B | 7155 | 120 | 16.11 | - | win |
| 3 | 4D MatMul line@assign4 (drop linemat reshape) | B | 6555 | 112 | 16.20 | - | win |
| 4 | ci=color-index on SMALL plane; shift ci not line | B | 5571 | 122 | 16.35 | - | win |
| 5 | assign = And(cum==s+1, valid) (band cells, drop runstart gate) | B | 5391 | 122 | 16.39 | - | win |
| 6 | ci via no-pad Conv(line, colorvec) (drop Mul+ReduceSum plane) | B | 4791 | 122 | 16.50 | 200/200 | ADOPT |

## Best achieved
16.50 @ mem 4791 params 122 — beats prior 15.48 by +1.02. Fresh 200/200 (isolated).

## Irreducible-floor analysis
Dominant intermediates: two fp32 line Slices `[1,10,1,15]`/`[1,10,15,1]` = 600B each
(1200B total). These are IRREDUCIBLE: any extraction from the free fp32 input via
Slice/Gather/Conv inherits fp32; casting the full input to fp16 costs 18000B; can't
crop the 10 channels (band colours are arbitrary 1-9); band axis is genuinely <=15.
Next: two fp16 `line` casts (300B each, needed one-hot for the compact MatMul) and
`linemm_c` reshape (300B, the only way to get the col line into matmul layout) and
three uint8 `[1,10,5,5]` blocks (250B each: two candidates + the orientation Where).

## OPEN ANGLES (re-attack backlog)
- Eliminate one of the two pipelines by detecting orientation from a cheap scalar
  first, then reading a SINGLE line — blocked by data-dependent slice (symbolic-dim
  trap) since the line position (row0 vs col0) is orientation-dependent.
- Collapse the three uint8 5x5 blocks: select `compact` (both `[1,10,1,5]`) BEFORE
  the orientation-specific reshape — but the row-vs-col PLACEMENT depends on the
  winner, so a post-select reshape still needs both candidate placements.

## INSIGHT (transferable)
- ⭐ "Run-length a band stack into a line": read ONLY the first cross-axis line
  (it already holds every band colour in order); dedup via CumSum of the
  colour-change indicator; assign[cell,slot] = (cumsum==slot+1) AND in-grid; one
  small `[1,10,1,LEN]@[1,1,LEN,SLOTS]` MatMul compacts to one-hot-per-slot; Pad the
  tiny `[1,10,5,5]` uint8 block out to 30x30 as the free output.
- ⭐ Compute a colour-INDEX plane with a no-pad `Conv(line_fp16, colorvec[1,10,1,1])`
  — folds Mul+ReduceSum and avoids the full `[1,10,1,LEN]` product plane. fp16 Conv
  runs fine under ORT_DISABLE_ALL (opset 11).
- ⭐ Since output is scored `>0`, the assign matrix need NOT pick one cell per band:
  gate by `in-grid` instead of `runstart` and let EVERY band cell contribute its
  colour (thickness-weighted magnitude is still `>0`); drops the runstart Mul gate.
- Detect colour change by shifting the SMALL colour-index plane (`[1,1,1,15]`), not
  the full `[1,10,1,15]` one-hot — the slice+pad shift is then ~60B not 300B.
- opset-11 `Pad` (pads as INPUT) accepts uint8; opset-10 `Pad` (attribute form)
  does NOT. Mixing CumSum (opset 11) forces opset 11 anyway, so the uint8 Pad path
  is free. fp16 Pad/MatMul/Conv/Where/Equal/ReduceSum all OK; fp16 CumSum CRASHES
  (keep the tiny cumsum plane fp32).
