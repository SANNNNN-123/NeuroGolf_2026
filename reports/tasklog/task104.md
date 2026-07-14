# task104 — 4522001f (quadrant-selected 9x9 fixed pattern)

**Rule:** Input is a fixed 3x3 grid encoding one of FOUR cases q in {0,1,2,3}.
Centre is always red(2); exactly ONE of the four corners [0,0],[0,2],[2,0],[2,2]
is green(3) and that corner index IS q. The OUTPUT is always one of FOUR fixed
9x9 green(3)/black(0) patterns selected solely by q (each = two 4x4 green blocks
on diagonal/anti-diagonal corners of the 9x9). Cells outside the top-left 9x9
footprint are all-channels-off.
**Current:** prior 16.73. This session: **17.31 pts, table-lookup mask select +
final Equal, mem 1778, params 400.**
**Target tier:** B (label map + Equal). Tier S/A blocked: the green mask is a
data-dependent pick among 4 fixed 9x9 patterns; it is not a single fixed
permutation of input cells (S) nor a row⊗col separable rectangle (A — the two
diagonal blocks couple r and c).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice g-ch 3x3 → corner-picker Conv qvec[1,4,1,1] → MatMul qvec@Mflat[4,81] → reshape 9x9 → Greater → uint8 L9 → Pad sentinel99 → Equal | B | 1778 | 400 | 17.31 | 200/200 | WIN first try |

## Best achieved
**17.31 @ mem 1778 params 400 — fresh 200/200 (isolated, temp-net).** Beats prior
16.73 by **+0.58**. Adopted? N (build-only per brief).

## Irreducible-floor analysis
Dominant intermediate is the padded uint8 label map L [1,1,30,30] = 900 B,
irreducible because the final Equal must span the full 30x30 output footprint and
ORT Pad rejects bool (so the 9x9→30x30 carrier is uint8, not bool). Everything
else is ≤81 B (3x3 green slice, [1,4] qvec, [1,81] Gflat, 9x9 mask/label). The
input one-hot and the bool output are free.

## OPEN ANGLES (re-attack backlog)
- Drop the 900B L by doing Equal at 9x9 then placing it in the 30x30 output — but
  Pad rejects bool (same wall as task195) and Concat/Scatter of 10 channels costs
  more than 900. No clean sub-900 final assembly found.
- params 400 could shave ~80 (Mflat 4x81 is the bulk; masks are structured) but
  mem dominates the log so payoff is <0.05 pt — not worth it.

## INSIGHT (transferable)
⭐ When the entire output is one of a SMALL FIXED SET of patterns selected by a
discrete scalar, skip all geometry: recover a one-hot selector vector qvec (here a
[4,1,3,3] corner-picker Conv on the colour slice, exactly one entry 1) and read
the output mask straight out of a const lookup table via `qvec @ Mflat[K, H*W]` —
a tiny [1,H*W] intermediate, no per-cell colour plane. The only 30x30 cost is the
canonical uint8 label-map for the free-bool Equal.
