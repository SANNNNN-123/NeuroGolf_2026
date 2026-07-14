# task300 — be94b721

**Rule:** Grid holds 3..4 small connected sprites, each in a DISTINCT colour
(`random_colors`, one per sprite). Sizes sorted DESCENDING → idx 0 = LARGEST.
Output = sprite-0's shape cropped to its own bbox (origin top-left), monochrome
in sprite-0's colour (ch-0 bg fills interior holes). Since colours are unique
per sprite, the max-pixel-COUNT colour channel (excluding ch0) IS sprite-0, and
every pixel of that colour belongs to sprite-0 → bbox of that channel = answer.

**Status label was "confirmed-infeasible" (no documented reason) → FALSE POSITIVE.**

**Current (prior):** 13.74 pts, gen:thbdh6332, mem 77438, params 108
(public net cast the full 10×30×30 input to fp16 = 18000B + many full fp16 planes).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | max-count channel → single-plane bbox → WORK×WORK shift+Pad one-hot | B | 6176 | 112 | **16.25** | 200/200 | win |

The win = (a) NEVER cast the full input (reduce counts/occupancy on the FREE fp32
input directly); (b) compute the bbox from the SINGLE gathered channel plane
(1-D occupancy, fp16) not a 10-channel profile; (c) shift only a 5×5 window to
the origin, label uint8, Pad(sentinel 10) to 30×30, Equal→free BOOL one-hot.

## Best achieved
16.25 @ mem 6176 params 112. Beats prior 13.74 by **+2.51** (≥0.3 ✓).
GENERALIZES: evaluate 267/267 + ISOLATED fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate: `bplane` = Gather(input, mc, axis=1) [1,1,30,30] fp32 =
3600B — irreducible because the crop window position (min_row,min_col) is
data-dependent on `mc`, which needs the full per-channel counts, so the full
single-channel plane must materialise before the windowed Gather (same circular
dependency as task036). Everything else is small: counts [1,10,1,1], 1-D fp16
occupancy [1,1,30,1]+[1,1,1,30], 5×5 fp16/uint8 work planes.

## OPEN ANGLES (re-attack backlog)
- Collapse `bplane` 3600B: a fused GatherND selecting channel AND windowing in one
  op would skip the full plane — untried (same open angle as task036).
- Drop the second `Where(boxmask,...)` sentinel path if `iseq` alone (Vs>0) already
  excludes out-of-box cells — risky if clipped window indices re-read on-pixels.

## INSIGHT (transferable)
⭐ "extract the LARGEST of several distinct-colour sprites to its bbox" is the
task036 crop-and-translate idiom with max-pixel-COUNT channel selection instead of
min-span. Distinct per-sprite colours make `ArgMax(ReduceSum(input,[2,3]) − bgpen·ch0)`
identify the largest sprite EXACTLY (no flood-fill/connectivity). Compute the bbox
from the SINGLE gathered channel plane (cheaper 1-D reductions than a 10-channel
profile), and never cast the full input — the public net's 18000B fp16 input cast
was pure waste.
