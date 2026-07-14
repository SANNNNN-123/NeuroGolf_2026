# task047 — 23581191

**Rule:** Fixed 9x9 grid. Input has one cyan(8) pixel at (r0,c0) and one orange(7) pixel at (r1,c1) (r0!=r1, c0!=c1, all in 1..size-3). Output paints column c0 cyan, column c1 orange, row r0 cyan, row r1 orange (rows overwrite columns at intersections), then the two cross-intersections (r0,c1) and (r1,c0) become red(2).
**Current:** 16.80 pts (stored; manifest 16.99 ext:kojimar6275), mem ~2988, params 36
**Target tier:** A — closed-form separable row/col crosshairs routed into the free bool output; no detection/flood needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 9x9 region, rowval/colval indicators, 2-Where colour resolve, Equal->oh, Pad(bool,opset13) | A | 2529 | 37 | 17.15 | 200/200 | works |
| 2 | replace Where chain with s=rowval+colval -> 17-entry Gather table -> L | A | 2349 | 52 | 17.22 | 200/200 | adopted |

## Best achieved
17.22 @ mem 2277 (eval reports 2349 incl. params) params 52 — beats prior 16.80 by +0.42 (and the 16.99 manifest by +0.23). Fresh isolated 200/200.

## Irreducible-floor analysis
Dominant intermediate is the [1,10,9,9] bool one-hot pre-pad = 810B (the 10-channel expansion, irreducible without leaving 10-ch output). Next: two [1,1,9,9] fp32 channel slices (324B each, needed to feed ReduceMax for the row/col seed positions) and the [1,1,9,9] int32 Gather-index plane (324B — uint8 Gather indices are rejected by ORT, so int32 is the floor). Working in the FIXED 9x9 active region (generator size is always 9) is the key lever: every plane is ~9x smaller than its 30x30 form, and the final Pad lifts the bool one-hot back to 30x30 for FREE.

## OPEN ANGLES (re-attack backlog)
- The two 324B fp32 channel slices feed only ReduceMax; if a single Slice of channels 7:9 -> [1,2,9,9] then per-channel reductions could share, but byte total is identical (648B). No clear win.
- The 324B int32 Gather index could vanish if L were built by fp16 arithmetic, but the s->colour map (0,7,8,14,15,16 -> 0,7,8,7,2,8) is non-monotonic and needs >=2 fp16 Where planes (324B) = neutral.
- 810B one-hot is the structural floor for a 10-channel output; only escape would be routing fewer channels, but 4 distinct colours are present.

## INSIGHT (transferable)
⭐ Crosshair/intersection tasks where two lines cross are a PURE FUNCTION OF s = rowval + colval (line-colour ramps summed): distinct line colours make every (single-line / crossing) case map to a unique sum, so one small Gather table replaces the entire cross-mask + equal-test + Where army. ⭐ Pad on a BOOL tensor REQUIRES opset 13 (Pad-11 rejects bool) — use opset 13 to keep the 10-ch one-hot bool all the way to the free 30x30 output instead of paying a 3600B uint8 30x30 plane (Equal-before-Pad in the small active region, not Pad-before-Equal). ⭐ ORT rejects uint8 Gather indices; int32 is the index floor.
